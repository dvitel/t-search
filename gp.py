''' Population based evolutionary loop and default operators, Koza style GP.
    Operators: 
        1. Initialization: ramped-half-and-half
        2. Selection: tournament
        3. Crossover: one-point subtree 
        4. Mutation: one-point subtree
'''

from functools import partial
import math
from typing import Callable, Literal, Optional, Sequence
from time import perf_counter
import numpy as np
import torch
from initialization import RHH, Initialization
from mutation import ClassicPointRandCrossover, ConstOptimization, ConstOptimization1, Deduplicate, Mutation, PointRandCrossover, PointRandMutation, ReplaceWithBestInner
from selection import TournamentSelection
from term import Builder, Builders, Op, Term, Value, Variable, evaluate, get_counts, get_depth, \
                    get_fn_arity, get_inner_terms, match_root, parse_term
from sklearn.base import BaseEstimator, RegressorMixin

from util import stack_rows  

GPSolverStatus = Literal["INIT", "MAX_GEN", "MAX_EVAL", "MAX_ROOT_EVAL", "SOLVED"]

class EvSearchTermination(Exception):
    ''' Reaching maximum of evals, gens, ops etc '''    
    def __init__(self, status: GPSolverStatus, *args):
        super().__init__(*args)
        self.status = status


def fit_0(fitness: torch.Tensor, 
          prev_best_fitness: Optional[torch.Tensor],
          rtol = 1e-04, atol = 1e-03) -> tuple[int | None, bool]:
    best_id = fitness.argmin().item()
    best_fitness = fitness[best_id]
    best_found = False 
    if prev_best_fitness is not None and (prev_best_fitness < best_fitness):
        return None, False
    zero = torch.zeros_like(best_fitness)
    if torch.isclose(best_fitness, zero, rtol = rtol, atol = atol):
        best_found = True
    return best_id, best_found
    
def mse_loss(predictions, target):
    return torch.mean((predictions - target) ** 2, dim=-1)
    
def mse_loss_nan_v(predictions, target, *, nan_error = torch.inf):
    loss = torch.mean((predictions - target) ** 2, dim=-1)
    loss = torch.where(torch.isnan(loss), torch.tensor(nan_error, device=loss.device, dtype=loss.dtype), loss)
    return loss     

def mse_loss_nan_vf(predictions, target, *, 
                    nan_value_fn = lambda m,t: torch.tensor(torch.inf, 
                                                    device = t.device, dtype=t.dtype), 
                    nan_frac = 0.5):
    nan_frac_count = math.floor(target.shape[0] * nan_frac)
    nan_mask = torch.isnan(predictions)
    err_rows: torch.Tensor = nan_mask.sum(dim=-1) > nan_frac_count
    bad_positions = nan_mask & err_rows.unsqueeze(-1)
    fixed_predictions = torch.where(bad_positions, 
                                    nan_value_fn(bad_positions, target),
                                    predictions)
    err_rows.logical_not_()
    fixed_positions = nan_mask & err_rows.unsqueeze(-1)
    fully_fixed_predictions = torch.where(fixed_positions, target, fixed_predictions)
    loss = torch.mean((fully_fixed_predictions - target) ** 2, dim=-1)
    del fully_fixed_predictions, fixed_predictions, fixed_positions, bad_positions, err_rows, nan_mask
    return loss         

def l1_loss(predictions, target):
    return torch.mean(torch.abs(predictions - target), dim=-1)  

def timed(fn: Callable, key: str, metrics: dict) -> Callable:
    ''' Decorator to time function execution '''
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = fn(*args, **kwargs)
        elapsed_time = round((perf_counter() - start_time) * 1000)
        metrics[key] = elapsed_time
        return result
    return wrapper

class GPSolver(BaseEstimator, RegressorMixin):

    def __init__(self, 
                ops: dict[str, Callable],
                fitness_fn: Callable = mse_loss_nan_v,
                fit_condition = partial(fit_0, rtol = 1e-04, atol = 1e-03),
                init: Initialization = RHH(),
                eval_fn = evaluate,
                pipeline: list['Mutation'] = [ TournamentSelection(), PointRandMutation(), PointRandCrossover() ],
                ops_counts: dict[str, tuple[int, int]] = {},
                forbid_patterns: list[str] = [],
                # next is more optimized
                inner_ops_max_counts: dict[str, dict[str, int]] = {},
                immediate_arg_limits: dict[str, dict[str, int]] = {},
                prohibit_ops_on_consts_only: bool = True,
                # commutative_ops: list[str] = [], # by all args
                max_term_depth = 17,
                min_consts: int = 0,
                max_consts: int = 5, # 0 to disable consts in terms
                min_vars: int = 1,
                max_vars: int = 10, # max number of free variables
                max_ops: dict[str, int] = {},
                max_gen: int = 100,
                max_root_evals: int = 100_000, 
                max_evals: int = 1_000_000,
                pop_size: int = 1000, 
                elitism: int = 0, 
                with_inner_evals: bool = False,
                cache_term_props: bool = True,
                cache_terms: bool = True,
                cache_evals: bool = True, # outputs and fitness
                rtol = 1e-04, atol = 1e-03, # NOTE: these are for semantic/outputs comparison, not for fitness, see fit_0
                rnd_seed: Optional[int] = None,
                torch_rnd_seed: Optional[int] = None,
                device = "cpu", dtype = torch.float32,
                ):
        
        self.ops = ops
        self.ops_counts = ops_counts

        self.max_term_depth = max_term_depth
        self.min_vars = min_vars 
        self.max_vars = max_vars
        self.min_consts = min_consts
        self.max_consts = max_consts
        self.max_ops = max_ops
        self.forbid_patterns = forbid_patterns
        self.fpatterns = []
        if len(self.forbid_patterns) > 0:
            self.match_cache = {}
            for p in self.forbid_patterns:
                t, i = parse_term(p)
                assert len(p) == i, f"Invalid pattern: {p}"
                self.fpatterns.append(t)
                
        # self.const_binding: list[torch.Tensor] = []
        self.const_range: torch.Tensor | None = None # detected from y on reset
        # NOTE: variables and consts are stored separately from tree - abstract shapes x * x + c * x + c 
        #       in this approach we have a problem with caching semantics of intermediate terms, as for different c and x, the results are different
        #       solution: make term_output as dictionary with keys (root, term). Root should be a part of all keys to identify concrete selection of c, x
        #       alternative: create subclasses of Term for Vars and Values - this is more explicit approach and better 
        #                    Vars = Term + var id, Values = Term + value Any.
        #                    Do we need (term, occur) in this case? Seems yes.
        self.fitness_fn = fitness_fn
        self.fit_condition = fit_condition
        self.init = init
        self.eval_fn = eval_fn
        self.pipeline = pipeline
        self.max_gen = max_gen
        self.max_root_evals = max_root_evals
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.elitism = elitism
        self.with_inner_evals = with_inner_evals
        self.elite_terms = []
        assert self.elitism < self.pop_size, "Elitism should be less than population size"
        self.cache_term_props = cache_term_props
        self.cache_terms = cache_terms
        self.cache_evals = cache_evals
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.dtype = dtype
        self.prohibit_ops_on_consts_only = prohibit_ops_on_consts_only
        self.inner_ops_max_counts = inner_ops_max_counts
        self.immediate_arg_limits = immediate_arg_limits
        # self.commutative_ops = set(commutative_ops)

        if rnd_seed is None:
            self.rnd = np.random
        else:
            self.rnd = np.random.RandomState(rnd_seed)     

        if torch_rnd_seed is None:
            self.torch_gen = None
        else:
            self.torch_gen = torch.Generator(device=device)
            self.torch_gen.manual_seed(rnd_seed)   

        # next are runtime fields and caches that works across fit calls
        self.target = None 
        # self.free_vars: Optional[Sequence] = None
        self.sign_syntax: dict[tuple[str, ...], Term] = {}
        self.term_repr: dict[Term, Term] = {} # unifies terms to same instance - to use in other caches
        self.term_outputs: dict[Term, torch.Tensor] = {}
        self.term_fitness: dict[Term, torch.Tensor] = {}
        # self.term_counts: dict[Term, np.ndarray] = {}
        self.pos_cache = {}
        self.pos_context_cache = {}
        self.depth_cache = {}
        self.counts_cache = {}
        self.crossover_cache = {} 
        self.inner_term_cache = {}

        self.best_term: Optional[Term] = None
        self.best_outputs: Optional[torch.Tensor] = None
        self.best_fitness: Optional[torch.Tensor] = None
        self.has_solution = False
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.metrics: dict[str, int | float | list[int|float]] = {}
        self.status: GPSolverStatus = "INIT"
        self.start_time: float = 0
        self.const_id = None 
        self.const_tape = None

        self.op_builders = {}
        for op_id, op_fn in self.ops.items():
            op_arity = get_fn_arity(op_fn)
            max_count = None 
            if op_id in self.max_ops:
                max_count = self.max_ops[op_id]
            op_builder = Builder(op_id, self._alloc_op_builder(op_id), op_arity, max_count = max_count)
                                    # commutative = op_id in self.commutative_ops)
            if op_id in self.ops_counts:
                op_min_count, op_max_count = self.ops_counts[op_id]
                op_builder.min_count = op_min_count
                op_builder.max_count = op_max_count

            self.op_builders[op_id] = op_builder

    def _reset_state(self, free_vars: Optional[Sequence] = None, target: Optional[Sequence] = None):
        ''' Called before each fit '''

        # reset caches 
        self.vars: list[Variable] = []
        self.var_binding: dict[str, torch.Tensor] = {}
        # self.const_binding: list[torch.Tensor] = []

        self.target = None 
        self.syntax: dict[tuple[str, ...], Term] = {}
        self.term_outputs: dict[Term, torch.Tensor] = {}
        self.term_fitness: dict[Term, torch.Tensor] = {}
        self.pos_cache = {}
        self.pos_context_cache = {}
        self.counts_cache = {}
        self.depth_cache = {}
        self.crossover_cache = {}

        self.best_term: Optional[Term] = None
        self.best_outputs: Optional[torch.Tensor] = None
        self.best_fitness: Optional[torch.Tensor] = None
        self.has_solution = False
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.metrics: dict[str, int | float | list[int|float]] = {}
        self.status: GPSolverStatus = "INIT"
        self.start_time: float = perf_counter()
        self.gen_metrics = {}
        self.is_fitted_ = False
        builders = {}
        self.elite_terms = []

        if self.max_consts > 0:
            const_builder = Builder("C", self._alloc_const, 0, self.min_consts, self.max_consts)
            builders[Value] = const_builder

        if free_vars is not None and len(free_vars) > 0 and (self.max_vars > 0):
            vars, var_binding = self.get_vars(free_vars)
            self.var_binding = var_binding
            self.vars = vars
            var_builder = Builder("X", self._alloc_var, 0, self.min_vars, self.max_vars)
            builders[Variable] = var_builder

        builders.update(self.op_builders)

        def get_term_builder(term: Term):
            if isinstance(term, Op):
                builder = builders[term.op_id]
            if isinstance(term, Variable):
                builder = builders[Variable]
            if isinstance(term, Value):
                builder = builders[Value]
            return builder
        
        self.builders = Builders(list(builders.values()), get_term_builder)

        arg_limits = {}
        if self.prohibit_ops_on_consts_only:
            for b in self.op_builders.values():
                arg_limits[b] = {builders[Value]: b.arity() - 1}

        for op_id, op_dict in self.immediate_arg_limits.items():
            if op_id not in self.op_builders:
                raise ValueError(f"Operator {op_id} not found in op_builders")
            b = self.op_builders[op_id]
            if b not in arg_limits:
                arg_limits[b] = {}
            for inner_op_id, limit in op_dict.items():
                if inner_op_id not in self.op_builders:
                    raise ValueError(f"Inner operator {inner_op_id} not found in op_builders")
                arg_limits[b][self.op_builders[inner_op_id]] = limit

        self.builders.limit_args(arg_limits)

        context_limits = {}
        for op_id, op_limits in self.inner_ops_max_counts.items():
            if op_id not in self.op_builders:
                continue
            context_limits[self.op_builders[op_id]] = {self.op_builders[inner_op_id]:cnt for inner_op_id, cnt in op_limits.items()}

        self.builders.limit_context(context_limits)

        if target is not None:
            if not torch.is_tensor(target):
                self.target = torch.tensor(target, device = self.device, dtype = self.dtype)
            else:
                self.target = target.to(device = self.device, dtype = self.dtype)
            
            min_value = self.target.min()
            max_value = self.target.max()
            if torch.isclose(min_value, max_value, rtol=self.rtol, atol=self.atol):
                min_value = min_value - 0.1
                max_value = max_value + 0.1
            dist = max_value - min_value
            min_value = min_value - 0.1 * dist
            max_value = max_value + 0.1 * dist
            self.const_range = torch.tensor([min_value, max_value], dtype= self.dtype, device=self.device)
            if torch.is_tensor(free_vars):
                min_fv = torch.min(free_vars)
                max_fv = torch.max(free_vars)
            else:
                min_fv = min(torch.min(xv).item() for xv in free_vars)
                max_fv = max(torch.max(xv).item() for xv in free_vars)
            self.const_range[0] = torch.minimum(self.const_range[0], min_fv)
            self.const_range[1] = torch.maximum(self.const_range[1], max_fv)
        
    def get_vars(self, free_vars):
        vars = []
        var_binding = {}
        for i, xi in enumerate(free_vars):
            v = Variable(f"x{i}")
            if not torch.is_tensor(xi):
                fv = torch.tensor(xi, dtype=self.dtype, device=self.device)
            else:
                fv = xi.to(device = self.device, dtype = self.dtype)        
            vars.append(v)
            var_binding[v.var_id] = fv 
        return vars, var_binding            

    def _alloc_var(self, *_) -> Variable:
        var = self.rnd.choice(self.vars)
        return var
    
    def _alloc_const(self, *_) -> Value: 
        ''' Should we random sample of try some grid? Anyway we tune '''
        if self.const_id is None or self.const_id >= self.pop_size:
            del self.const_tape
            self.const_id = 0
            self.const_tape = self.const_range[0] + \
                                torch.rand(self.pop_size, device=self.device, 
                                            dtype=self.dtype, generator=self.torch_gen) * \
                                                (self.const_range[1] - self.const_range[0])
        value = self.const_tape[self.const_id]
        self.const_id += 1
        # const_id = len(self.const_binding)
        # self.const_binding.append(value)
        # return Value(const_id)
        self.metrics["consts"] = self.metrics.get("consts", 0) + 1
        return Value(value)
    
    def report_evals(self, num_evals: int, num_root_evals: int):
        self.evals += num_evals
        self.root_evals += num_root_evals
        if self.evals > self.max_evals:
            raise EvSearchTermination("MAX_EVAL", "Maximum number of evaluations reached")
        if self.root_evals > self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL", "Maximum number of root evaluations reached")
        
    def eval(self, terms: list[Term]):

        if not self.cache_evals:
            self.term_outputs.clear()
            self.term_fitness.clear()

            
        outputs = []
        for term in terms:
            output = self.eval_fn(term, self.ops, self._get_binding, self._set_binding)
            outputs.append(output)

        new_terms = [t for t in terms if t not in self.term_fitness]
        if self.with_inner_evals:
            full_new_terms = list(new_terms)
            outputs.clear()
            already_added = set()
            for term in new_terms:
                if term not in self.inner_term_cache:
                    inner_terms = get_inner_terms(term)
                    inner_ops = [t for t in inner_terms if not isinstance(t, (Variable, Value))]
                    self.inner_term_cache[term] = inner_ops
                for inner_term in self.inner_term_cache[term]:
                    if inner_term in self.term_fitness or inner_term in already_added:
                        continue
                    output = self.get_cached_output(inner_term)
                    outputs.append(output)
                    new_terms.append(inner_term)
                    already_added.add(inner_term)
            new_terms = full_new_terms

        # assert len(terms) > 0, "No terms to update fitness for"

        new_fitness = self.compute_fitness(new_terms, all_new=True)
        if new_fitness is not None:
            best_id, best_found = self.fit_condition(new_fitness, self.best_fitness)
            if best_id is not None:
                self.best_term = new_terms[best_id]
                self.best_outputs = self.get_cached_output(self.best_term)
                self.best_fitness = new_fitness[best_id]
        
            self.has_solution = best_found

        if self.has_solution:
            self.status = "SOLVED"    

        if self.best_fitness is not None:
            best_fitness = self.best_fitness.unsqueeze(-1) if len(self.best_fitness.shape) == 0 else self.best_fitness
            for fi, fv in enumerate(best_fitness):
                fk = f"fitness_{fi}"
                self.gen_metrics[fk] = fv.item()

        if self.best_term is not None: # best term stats 
            best_depth = get_depth(self.best_term, self.depth_cache)
            best_counts = get_counts(self.best_term, self.builders, self.counts_cache)
            best_size = best_counts.sum().item()
            self.gen_metrics["best_term_depth"] = best_depth
            self.gen_metrics["best_term_size"] = best_size
            self.gen_metrics["best_counts"] = best_counts.tolist()

        if self.elitism > 0:
            fitness = stack_rows([self.term_fitness[t] for t in terms])
            sorted_ids = torch.argsort(fitness, dim=0)
            elite_ids = sorted_ids[:self.elitism].tolist()
            self.elite_terms = [terms[i] for i in elite_ids]

        self.gen_metrics["pop_size"] = len(terms)

        return outputs, fitness
    
    def compute_fitness(self, terms: list[Term], 
                            return_tensor: bool = False,
                            all_new: bool = False) -> torch.Tensor | list[torch.Tensor]:
        if all_new:
            missing_terms = terms
        else:
            missing_terms = [t for t in terms if t not in self.term_fitness]
        new_fitness = None
        if len(missing_terms) > 0:
            outputs = [self.get_cached_output(t) for t in missing_terms]
            predictions = stack_rows(outputs, target_size=self.target.shape[0])
            new_fitness: torch.Tensor = self.fitness_fn(predictions, self.target)
            for t, f in zip(missing_terms, new_fitness):
                self.term_fitness[t] = f
            del predictions
        if all_new:
            return new_fitness
        if (len(missing_terms) == len(terms)) and return_tensor:
            return new_fitness
        fitness = [self.term_fitness[t] for t in terms]
        if return_tensor:
            fitness = stack_rows(fitness)
        return fitness

    def breed(self, population: list[Term], 
                outputs: torch.Tensor | Sequence[torch.Tensor], 
                fitness: torch.Tensor | Sequence[torch.Tensor], 
                size: int) -> list[Term]:
        ''' Pipeline that mutates parents and then applies crossover on pairs. One-point operations '''

        # caches 

        if not self.cache_term_props:
            self.pos_cache.clear()
            self.counts_cache.clear()
            self.pos_context_cache.clear()
            self.depth_cache.clear()

        # validation 1
        # for term in population:
        #     poss = get_positions(term, {})
        #     for pos in poss:
        #         get_pos_constraints(pos, self.builders, {}, {})
        #     pass        

        size = size - len(self.elite_terms)

        children = timed(self.pipeline[0], "selection_time", self.gen_metrics)(self, population, size)

        for mutation_id in range(1, len(self.pipeline)):
            mutation = self.pipeline[mutation_id]
            children = timed(mutation, f"{mutation.name}_time", self.gen_metrics)(self, children)
            self.gen_metrics[mutation.name] = mutation.metrics

        children.extend(self.elite_terms)

        return children
    
    def _validate_term(self, term: Term) -> bool:
        for fpattern in self.fpatterns:
            match = match_root(term, fpattern, prev_matches=self.match_cache)
            if match is not None:
                return False
        return True
            
    def _alloc_op_builder(self, op_id: int) -> Callable:

        hit_key = "syntax_hit"
        miss_key = "syntax_miss"

        if self.cache_terms:
            def _alloc_op(*args):
                signature = (op_id, *args) 
                if signature in self.syntax:
                    key = hit_key
                    term = self.syntax[signature]
                else:
                    key = miss_key
                    term = Op(op_id, args)
                    self.syntax[signature] = term 
                if not self._validate_term(term):
                    self.syntax.pop(signature, None)
                    return None
                self.metrics[key] = self.metrics.get(key, 0) + 1
                return term 
        else:
            def _alloc_op(*args):
                self.metrics[miss_key] = self.metrics.get(miss_key, 0) + 1
                term = Op(op_id, args)
                if self._validate_term(term):
                    return term
                return None
            
        return _alloc_op
    
    def get_cached_output(self, term: Term) -> Optional[torch.Tensor]:
        if isinstance(term, Variable):
            return self.var_binding[term.var_id]
        if isinstance(term, Value):
            # return self.const_binding[term.value]
            return term.value            
        return self.term_outputs.get(term, None)

    
    def _get_binding(self, root: Term, term: Term) -> Optional[torch.Tensor]:        
        res_in_cache = self.get_cached_output(term)      

        if res_in_cache is None:
            self.metrics["eval_cache_miss"] = self.metrics.get("eval_cache_miss", 0) + 1
        else:
            self.metrics["eval_cache_hit"] = self.metrics.get("eval_cache_hit", 0) + 1

        return res_in_cache

    def _set_binding(self, root: Term, term: Term, value: torch.Tensor):
        self.evals += 1
        if root == term:
            self.root_evals += 1
        if self.evals == self.max_evals:
            raise EvSearchTermination("MAX_EVAL")
        if self.root_evals == self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL")
        self.term_outputs[term] = value

    def _checkpoint_metrics(self):
        if len(self.gen_metrics) > 0:
            self.metrics.setdefault("gens", []).append(self.gen_metrics)
            self.gen_metrics = {}

    def _loop(self):   
        population = timed(self.init, "init_time", self.gen_metrics)(self, self.pop_size)
        self.gen_metrics.update(self.init.metrics)
        outputs, fitness = timed(self.eval, "eval_time", self.gen_metrics)(population)
        self._checkpoint_metrics()
        while not self.has_solution and self.gen < self.max_gen and self.evals < self.max_evals and self.root_evals < self.max_root_evals:
            population = self.breed(population, outputs, fitness, self.pop_size) 
            del outputs, fitness 
            outputs, fitness = timed(self.eval, "eval_time", self.gen_metrics)(population)
            self.gen += 1
            self._checkpoint_metrics()

    def _add_final_metrics(self):
        self._checkpoint_metrics()
        self.metrics['gen'] = self.gen
        self.metrics['evals'] = self.evals
        self.metrics['root_evals'] = self.root_evals
        self.metrics["final_time"] = round((perf_counter() - self.start_time) * 1000)
        self.metrics["status"] = self.status
        if self.best_term is not None:
            self.metrics["solution"] = self.best_term

    def check_trivial(self):
        if torch.allclose(self.target, self.target[0], rtol = self.rtol, atol = self.atol):
            self.best_term = Value(self.target[0]) #len(self.const_binding))
            self.best_fitness = torch.tensor(0, device=self.device, dtype=self.dtype)
            self.best_outputs = self.target
            # self.const_binding.append(self.target[0])
            self.status = "SOLVED"
            self.has_solution = True
            return True 
        for x in self.vars:
            x_binding = self.var_binding[x.var_id]
            if torch.allclose(self.target, x_binding, rtol = self.rtol, atol = self.atol):
                self.best_term = x
                self.best_fitness = torch.tensor(0, device=self.device, dtype=self.dtype)
                self.best_outputs = x_binding
                self.status = "SOLVED"
                self.has_solution = True
                return True
        return False 

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> 'GPSolver':
        """
        Fit the solver to the data.

        Args:
            X (array-like): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns the instance itself.
        """
        self._reset_state(free_vars=X, target=y)
        if self.check_trivial():
            return self
        try:
            self._loop()
        except EvSearchTermination as e:
            self.status = e.status        
        self.is_fitted_ = True
        if self.status == "INIT":
            self.status = "MAX_GEN"
        self._add_final_metrics()
        return self
    
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Predict using the trained solver.

        Args:
            X (array-like): Input features.

        Returns:
            array-like: Predicted values.
        """
        if not self.is_fitted_ or self.best_term is None:
            raise RuntimeError("Solver is not fitted yet")
        
        _, var_binding = self.get_vars(X)
        
        def get_binding(root: Term, term: Term, *_) -> Optional[torch.Tensor]:
            if isinstance(term, Variable):
                return var_binding[term.var_id]
            if isinstance(term, Value):
                # return self.const_binding[term.value]
                return term.value
            return None
        
        def set_binding(*_):
            pass 
        
        output = self.eval_fn(self.best_term, self.ops, get_binding, set_binding)
        if output is None:
            raise RuntimeError("Evaluation of the best term returned None, not all terminals may be bound")
        output_numpy = output.cpu().numpy()
        return output_numpy
    
# NOTE: on metrics:
#       1. In contrast to cde-search, we do not collect semantic and syntactic diversity measures of population
#          For algebraic domain provided notions of diversity could be noninformative and should be reconsidered.
#       2. Also, we do not collect children "betterness" as it would require preservation of parent evaluations 
#          till moment of children evaluation. To avoid logic compication we decided to avoid this. Also algebraic domain make "betterness" also vagually defined. 

# IDEA: 1. annealing of tests in test set 
#       2. annealing of constraints 

# PROBLEM: 1. Many const semantics or same var semantics in default Koza GP.
#          2. Constraining minimal number of vars and consts.
#          3. More general constraint specification mechanism should be developed with Tree Tries. 

# TODO: 
#       DONE 1. Testing with caches, probably separate cache enablance. 
#       2. Lexicase selection and its advanced forms 
#          More advanced form of lexicase that considers pair of axes of interaction CS space
#       3. Unification with discrete domains? Can this work with discrete domains?
#       4. Unification with other evo processes in cde-search: NSGA and coevolution.
#       DONE 5. Tuning of constants 
#       6. Syntactic simplifications with axioms (again, need Tree Tries to match rules)
#          No need in tree tries. Rewrites could be done with unification and then replacement.         
#          Do we really need syntactic simplification??? 
#       7. Towards abstract forms (x * x + c * x + c)
#           Reduced to another mutation operator as it replaces any child term with linear combination.
#           Abstract form is just selection of isinstance(term, (Value, Variable)). Generally any child term could be replaced.
#       8. Towards semantic GP (add operators) + propose tuned point operator, using indices
#       9. Math properties and dynamic constraint sets.


#       [BAD IDEA] 10. Gen math expr instead of lisp expr 

#      11. Other metrics??? Add when caches are enabled - syntactic diversity (is there convergance to same syntax)
#      DONE 12. Elitism
#      13. Aging??? 
#      14. Dropout ???
#      15. Distribution control in gen_term based on statistics of past decisions at point of generation -
#           First, We need to have metric to see how gen process produce unique terms, not previously found in cache - should be controlled on term build.
#      DONE 16. We observe that classic crossover frequently fallbacks to reproduction - less point-pairs that required num_children in breed.
#          Therefore, we should noto require more generations from pair than present number of crossover points. 
#          Better to attempt next parents when budget of crossover is not exhausted.
#          Crossover cache hits --> does it make sense to produce same children? On cache hit - should be no child. Or crossover point should be prohibited.
#          Aging controls which points could crossover.
#          Globally crossover should not work with parents, but only with repository of crossover points (root, term, occur)

#      17. Annealing on present ops - max_counts. At what point to add new op? Based on frequency of cache hits or max gens?
#          Which ops should be first? Should we use fft here?

#      Reorganiziations:
#      DONE 18. Crossover fix --> do not do useless reproductions --> breed reorganization.
#      CURRENT 19. Optim of points as separate mutation operator that can evaluate
#      DONE 20. Inner semantics collection and filtering - rethink of current eval.

# NOTE [BAD IDEA]: we should probably go with const identities: 10 constant - so we allocate 10 identities but have different their bindings ??
# PROBLEM: 1. const identity should have max of 1 presence in the term, it seems that it should be this way, or small number???
#          2. On crossover of const identities, bindings should be transfered to children - should or not??? should
# NOTE: this is bad idea (attempted) - no benefit to move from consts list to dict[Term, dict[int, Tensor]] for consts 
#           - it complicates logic and requires additional mandatory binding step. 
#       in current implementation we can collect term consts with one traversal if necessary.


# TODO: 
#      Crossover without fails with iterating poss
#      ConstOptimization with multidim tensor of consts in one go 
#      Solection operators
#      Mutation that is guided by distribution of syntaxes in population ??? 

if __name__ == "__main__":

    from torch_alg import alg_ops, koza_1
    import json
    
    device = "cuda"
    dtype = torch.float16
    rnd_seed = 42

    solver = GPSolver(ops = alg_ops, device = device, dtype = dtype,
                        rnd_seed = rnd_seed, torch_rnd_seed = rnd_seed,
                        max_consts=5, elitism=10,
                        max_ops = {"inv": 5, "neg": 5},
                        with_inner_evals=True,
                        init=RHH(),
                        #(num_tries=1, lr=0.1)],
                        pipeline=[TournamentSelection(), 
                                  PointRandMutation(), 
                                  PointRandCrossover(), 
                                  Deduplicate(), 
                                  ConstOptimization(num_vals = 10, lr=1.0)],
                        # mutations=[PointRandMutation(), PointRandCrossover(), Deduplicate(), ConstOptimization1(lr=1.0)],
                        # commutative_ops=["add", "mul"],
                        forbid_patterns = [
                            # "(inv (inv .))",
                            # "(neg (neg .))",
                            # "(sin (... (sin .)))",
                            # "(cos (... (cos .)))",
                            # "(exp (... (exp .)))",
                            # "(log (... (log .)))",
                            # "(... pow (pow . .))",
                            ],
                        inner_ops_max_counts={
                            "sin": {"sin": 0},
                            "cos": {"cos": 0},
                            "exp": {"exp": 0},
                            "log": {"log": 0},                        
                            "pow": {"pow": 1},
                            "inv": {"inv": 1},
                        },
                        immediate_arg_limits={
                            "inv": {"inv": 0},
                            "neg": {"neg": 0},
                            "log": {"exp": 0},
                            "exp": {"log": 0},
                        },
                        )

    free_vars, target = koza_1.sample_set("train", device = device, dtype = dtype,
                                            generator=solver.torch_gen,
                                            sorted=True)

    solver.fit(free_vars, target)

    with open("gp-metrics.json", "w") as f:
        json.dump(solver.metrics, indent=4, default=str, fp=f)

    # print("Metrics:\n", metrics_json)

    pass
