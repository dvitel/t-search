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
from spatial import InteractionIndex, RTreeIndex, SpatialIndex
from term import Builder, Builders, Op, Term, Value, Variable, evaluate, get_counts, get_depth, \
                    get_fn_arity, get_pos_constraints, get_positions, match_root, \
                    one_point_rand_crossover, one_point_rand_mutation, parse_term, \
                    ramped_half_and_half
from sklearn.base import BaseEstimator, RegressorMixin

#utils

def stack_rows(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    max_size = max(0 if len(ti.shape) == 0 else ti.shape[0] for ti in tensors)
    sz = (len(tensors), ) if max_size == 0 else (len(tensors), max_size)
    res = torch.empty(sz, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, ti in enumerate(tensors):
        res[i] = ti # assuming broadcastable
    return res

# selections
def tournament_selection(size: int, *, 
                         fitness: torch.Tensor | Sequence[torch.Tensor],
                         gen: torch.Generator, tournament_selection_size = 7, **_):
    ''' Fitness is 1d tensor of fitness selected for tournament '''
    should_free = False
    if not torch.is_tensor(fitness):        
        fitness = stack_rows(fitness)
        should_free = True
    selected_ids = torch.randint(fitness.shape[0], (size, tournament_selection_size), dtype=torch.int, device=fitness.device,
                                 generator=gen)
    selected_fitnesses = fitness[selected_ids]
    best_id_id = torch.argmin(selected_fitnesses, dim=-1)
    best_ids = torch.gather(selected_ids, dim=-1, index = best_id_id.unsqueeze(-1)).squeeze(-1)
    del selected_ids, selected_fitnesses, best_id_id
    if should_free:
        del fitness
    return best_ids

def lexicase_selection(size: int, *, 
                       nan_error = torch.inf,
                       outputs: torch.Tensor, target: torch.Tensor, 
                       gen: torch.Generator, **_):
    """ Based on Lee Spector's team: Solving Uncompromising Problems With Lexicase Selection 
        This is iterative version.
    """


    should_free = False
    if not torch.is_tensor(outputs):
        outputs = stack_rows(outputs)
        should_free = True    

    nan_interactions = torch.abs(outputs - target)

    interactions = torch.nan_to_num(nan_interactions, nan=nan_error)
    del nan_interactions
    
    selected_ids = torch.zeros(size, dtype=torch.int, device=outputs.device)

    for pos_i in range(size):
        shuffled_test_ids = torch.randperm(interactions.shape[-1], device=interactions.device,
                                            generator=gen)
        candidate_ids = torch.arange(interactions.shape[0], device=interactions.device) # all candidates
        for test_id in shuffled_test_ids:
            test_min_diff = torch.min(interactions[candidate_ids, test_id])
            candidate_id_ids, = torch.where(interactions[candidate_ids, test_id] == test_min_diff)
            candidate_ids = candidate_ids[candidate_id_ids]
            if len(candidate_ids) == 1:
                break
        if len(candidate_ids) == 1:
            selected_ids[pos_i] = candidate_ids[0]
            continue
        best_id_id = torch.randint(len(candidate_ids), (1,), device=interactions.device,
                                    generator=gen)
        selected_ids[pos_i] = candidate_ids[best_id_id]
    del interactions, shuffled_test_ids
    if should_free:
        del outputs
    return selected_ids


# fitness = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# res = tournament_selection(10, fitness, tournament_selection_size=3)
# pass

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
        metrics.setdefault(key, []).append(elapsed_time)
        return result
    return wrapper

class GPSolver(BaseEstimator, RegressorMixin):

    def __init__(self, 
                ops: dict[str, Callable],
                fitness_fn: Callable = mse_loss_nan_v,
                fit_condition = partial(fit_0, rtol = 1e-04, atol = 1e-03),
                init_args: dict = dict(init_fn = ramped_half_and_half,
                                       init_from_cache = False, # Warn: if True, cache_terms should nbe enabled, violates min constraints
                                       ),
                eval_fn = evaluate,
                breed_args: dict = dict(
                    selection_fn = tournament_selection,
                    mutation_fn = one_point_rand_mutation,
                    crossover_fn = one_point_rand_crossover,
                    mutation_rate = 0.1,
                    crossover_rate = 0.9,
                ),
                ops_counts: dict[str, tuple[int, int]] = {},
                forbid_patterns: list[str] = [],
                # next is more optimized
                inner_ops_max_counts: dict[str, dict[str, int]] = {},
                immediate_arg_limits: dict[str, dict[str, int]] = {},
                prohibit_ops_on_consts_only: bool = True,
                # commutative_ops: list[str] = [], # by all args
                min_consts: int = 0,
                max_consts: int = 5, # 0 to disable consts in terms
                min_vars: int = 1,
                max_vars: int = 10, # max number of free variables
                max_ops: dict[str, int] = {},
                max_gen: int = 100,
                max_root_evals: int = 100_000, 
                max_evals: int = 1_000_000,
                pop_size: int = 1000,
                cache_term_props: bool = False,
                cache_terms: bool = False,
                cache_evals: bool = False, # outputs and fitness
                cache_inner_evals: bool = False,
                cache_crossover: bool = False,
                rtol = 1e-04, atol = 1e-03, # NOTE: these are for semantic/outputs comparison, not for fitness, see fit_0
                rnd_seed: Optional[int] = None,
                torch_rnd_seed: Optional[int] = None,
                device = "cpu", dtype = torch.float32,
                ):
        
        self.ops = ops
        self.ops_counts = ops_counts

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
        self.const_range: tuple[torch.Tensor, torch.Tensor] | None = None # detected from y on reset
        # NOTE: variables and consts are stored separately from tree - abstract shapes x * x + c * x + c 
        #       in this approach we have a problem with caching semantics of intermediate terms, as for different c and x, the results are different
        #       solution: make term_output as dictionary with keys (root, term). Root should be a part of all keys to identify concrete selection of c, x
        #       alternative: create subclasses of Term for Vars and Values - this is more explicit approach and better 
        #                    Vars = Term + var id, Values = Term + value Any.
        #                    Do we need (term, occur) in this case? Seems yes.
        self.fitness_fn = fitness_fn
        self.fit_condition = fit_condition
        self.init_args = init_args
        self.eval_fn = eval_fn
        self.breed_args = breed_args
        self.max_gen = max_gen
        self.max_root_evals = max_root_evals
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.cache_term_props = cache_term_props
        self.cache_terms = cache_terms
        self.cache_evals = cache_evals
        self.cache_inner_evals = cache_inner_evals
        self.cache_crossover = cache_crossover
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
        self.inner_semantics: dict[Term, dict[Term, torch.Tensor]] = {} # not yet in term_fitness and are not roots
        self.term_fitness: dict[Term, torch.Tensor] = {}
        # self.term_counts: dict[Term, np.ndarray] = {}
        self.pos_cache = {}
        self.pos_context_cache = {}
        self.depth_cache = {}
        self.counts_cache = {}
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
        self.start_time: float = 0

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
        self.inner_semantics: dict[Term, dict[Term, torch.Tensor]] = {}
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

        builders = {}

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
            self.const_range = (min_value, max_value - min_value)
        
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
        value = self.const_range[0] + torch.rand((1,), device=self.device, dtype=self.dtype,
                                                    generator=self.torch_gen) * self.const_range[1]
        # const_id = len(self.const_binding)
        # self.const_binding.append(value)
        # return Value(const_id)
        self.metrics["consts"] = self.metrics.get("consts", 0) + 1
        return Value(value)

    def init(self, size: int, *, init_fn: Callable = ramped_half_and_half, init_from_cache = False, **_) -> list[Term]:
        ''' Initialize each term in population 0 with self.init_fn '''
        init_metrics = self.metrics.setdefault("init_metrics", {})
        if self.cache_terms and init_from_cache:
            none_count = 0
            sz = size - len(self.vars)
            while len(self.syntax) < sz:
                term = init_fn(self.builders, rnd=self.rnd, gen_metrics = init_metrics)
                if term is None:
                    none_count += 1
                if none_count == size:
                    break 
            res = list(self.syntax.values())[:sz]
            res.extend(self.vars)
        else:
            res = []
            for _ in range(size):
                term = init_fn(self.builders, rnd=self.rnd, gen_metrics = init_metrics)
                # print(str(term))
                if term is not None:
                    res.append(term)
        return res 
    
    def eval(self, terms: list[Term], *, eval_fn: Callable = evaluate, **_) -> Optional[Term]:
        output_list = []
        for term in terms:
            term_output = eval_fn(term, self.ops, self._get_binding, self._set_binding) 
            output_list.append(term_output)

        # assert len(terms) > 0, "No terms to update fitness for"

        if self.cache_evals: # also means with inner semantics of terms 
            new_term_ids = [tid for tid, term in enumerate(terms) if term not in self.term_fitness]

            best_found = self.has_solution

            if len(new_term_ids) > 0:
                
                all_outputs = [output_list[tid] for tid in new_term_ids]
                all_terms = [terms[tid] for tid in new_term_ids]
                for tid in new_term_ids:
                    for inner_term, inner_outputs in self.inner_semantics.get(terms[tid], {}).items():
                        all_terms.append(inner_term)
                        all_outputs.append(inner_outputs)
                self.inner_semantics.clear()

                predictions = stack_rows(all_outputs)
                new_fitness: torch.Tensor = self.fitness_fn(predictions, self.target)
                for t, f in zip(all_terms, new_fitness):
                    self.term_fitness[t] = f
                best_id, best_found = self.fit_condition(new_fitness, self.best_fitness)
                if best_id is not None:
                    self.best_term = all_terms[best_id]
                    self.best_outputs = predictions[best_id].clone()
                    self.best_fitness = new_fitness[best_id].clone()
                del predictions

            outputs = output_list                    
            fitness = [self.term_fitness[t] for t in terms]

        else:

            outputs = stack_rows(output_list)
            fitness = self.fitness_fn(outputs, self.target) 
            best_id, best_found = self.fit_condition(fitness, self.best_fitness)
            if best_id is not None:
                self.best_term = terms[best_id]
                self.best_outputs = outputs[best_id].clone()
                self.best_fitness = fitness[best_id].clone()    
        
        self.has_solution = best_found
        if self.has_solution:
            self.status = "SOLVED"    

        if self.best_fitness is not None:
            best_fitness = self.best_fitness.unsqueeze(-1) if len(self.best_fitness.shape) == 0 else self.best_fitness
            for fi, fv in enumerate(best_fitness):
                fk = f"fitness_{fi}"
                self.metrics.setdefault(fk, []).append(fv.item())

        if self.best_term is not None: # best term stats 
            best_depth = get_depth(self.best_term, self.depth_cache)
            best_counts = get_counts(self.best_term, self.builders, self.counts_cache)
            best_size = best_counts.sum().item()
            self.metrics.setdefault("best_term_depth", []).append(best_depth)
            self.metrics.setdefault("best_term_size", []).append(best_size)
            self.metrics["best_counts"] = best_counts.tolist()

        return outputs, fitness
    
    def breed(self, population: list[Term], 
                outputs: torch.Tensor | Sequence[torch.Tensor], 
                fitness: torch.Tensor | Sequence[torch.Tensor], 
                size: int, *,
                selection_fn: Callable[[list[Term], int], torch.Tensor] = tournament_selection, 
                mutation_fn: Callable = one_point_rand_mutation, 
                crossover_fn: Callable = one_point_rand_crossover,
                mutation_rate = 0.1, crossover_rate = 0.9, **_) -> list[Term]:
        ''' Pipeline that mutates parents and then applies crossover on pairs. One-point operations '''

        # caches 

        if self.cache_term_props:
            pos_cache = self.pos_cache
            counts_cache = self.counts_cache
            pos_context_cache = self.pos_context_cache
            depth_cache = self.depth_cache
        else:
            pos_cache = {}
            counts_cache = {}
            pos_context_cache = {}
            depth_cache = {}

        if self.cache_crossover:
            crossover_cache = self.crossover_cache
        else:
            crossover_cache = {}

        # validation 1
        # for term in population:
        #     poss = get_positions(term, {})
        #     for pos in poss:
        #         get_pos_constraints(pos, self.builders, {}, {})
        #     pass        

        selection_start = perf_counter()
        selected_ids = selection_fn(size, population = population,
                                    fitness=fitness, outputs=outputs,
                                    target = self.target, gen=self.torch_gen)
        selection_end = perf_counter()
        self.metrics.setdefault("selection_time", []).append(round((selection_end - selection_start) * 1000))

        mutation_mask = torch.rand(size, device=self.device,
                                    generator=self.torch_gen) < mutation_rate
        
        crossover_mask = torch.rand(size // 2, device=self.device,
                                        generator=self.torch_gen) < crossover_rate

        selected_ids_list = selected_ids.tolist()
        mutation_mask_list = mutation_mask.tolist()
        crossover_mask_list = crossover_mask.tolist()
        del selected_ids, mutation_mask, crossover_mask

        parents = [population[i] for i in selected_ids_list]
        mutation_pos = {}
        for i, parent in enumerate(parents):
            if mutation_mask_list[i]:
                mutation_pos.setdefault(parent, []).append(i)

        mutated_parents = list(parents)

        mutation_metrics = {}
        mutation_start = perf_counter()
        for term, term_p in mutation_pos.items():
            mutated_terms = mutation_fn(term = term,
                                builders = self.builders,
                                pos_cache = pos_cache,
                                pos_context_cache = pos_context_cache,
                                counts_cache = counts_cache,
                                rnd=self.rnd, num_children=len(term_p),
                                mutation_metrics = mutation_metrics)
            for i, mterm in zip(term_p, mutated_terms):
                mutated_parents[i] = mterm
        mutation_end = perf_counter()
        mutation_time = round((mutation_end - mutation_start) * 1000)
        self.metrics.setdefault("mutation_time", []).append(mutation_time)
        mutation_metrics["time"] = mutation_time

        children = list(mutated_parents)

        # validation 2
        # for term in children:
        #     poss = get_positions(term, {})
        #     for pos in poss:
        #         get_pos_constraints(pos, self.builders, {}, {})
        # pass        

        crossover_pairs = {}
        for i, should_crossover in enumerate(crossover_mask_list):
            if should_crossover:
                parent1 = mutated_parents[2 * i]
                parent2 = mutated_parents[2 * i + 1]
                crossover_pairs.setdefault((parent1, parent2), []).append(i)

        crossover_metrics = {}
        crossover_start = perf_counter()
        for (parent1, parent2), pair_ids in crossover_pairs.items():
            new_children = crossover_fn(term1 = parent1, term2 = parent2, 
                                builders = self.builders,
                                pos_cache = pos_cache,
                                pos_context_cache = pos_context_cache,
                                counts_cache = counts_cache,
                                crossover_cache = crossover_cache,
                                depth_cache = depth_cache,
                                rnd = self.rnd, num_children=2 * len(pair_ids),
                                crossover_metrics = crossover_metrics)
            for i, ii in enumerate(pair_ids):
                children[2 * ii] = new_children[2 * i]
                children[2 * ii + 1] = new_children[2 * i + 1]
        crossover_end = perf_counter()
        crossover_time = round((crossover_end - crossover_start) * 1000)
        self.metrics.setdefault("crossover_time", []).append(crossover_time)
        crossover_metrics["time"] = crossover_time

        self.metrics.setdefault(f"mutation", []).append(mutation_metrics)

        self.metrics.setdefault(f"crossover", []).append(crossover_metrics)

        # validation 3
        # for term in children:
        #     poss = get_positions(term, {})
        #     for pos in poss:
        #         get_pos_constraints(pos, self.builders, {}, {})
        # pass        

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
    
    def _get_binding(self, root: Term, term: Term) -> Optional[torch.Tensor]:        
        if isinstance(term, Variable):
            return self.var_binding[term.var_id]
        if isinstance(term, Value):
            # return self.const_binding[term.value]
            return term.value

        # next is unnecessary as data never lands into term_outputs
        # if not self.cache_evals:
        #     return None        

        if term in self.term_outputs:
            self.metrics["eval_cache_hit"] = self.metrics.get("eval_cache_hit", 0) + 1
            return self.term_outputs[term]
        else:
            self.metrics["eval_cache_miss"] = self.metrics.get("eval_cache_miss", 0) + 1

        return None 

    def _set_binding(self, root: Term, term: Term, value: torch.Tensor):
        self.evals += 1
        if root == term:
            self.root_evals += 1
        if self.evals == self.max_evals:
            raise EvSearchTermination("MAX_EVAL")
        if self.root_evals == self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVAL")
        if not self.cache_evals:
            return
        self.term_outputs[term] = value
        if self.cache_inner_evals and (term != root):
            self.inner_semantics.setdefault(root, {})[term] = value 

    def _fit_inner(self, init_fn: Callable, eval_fn: Callable, breed_fn: Callable):   
        population = init_fn(self.pop_size)
        while self.gen < self.max_gen and self.evals < self.max_evals and self.root_evals < self.max_root_evals:
            outputs, fitness = eval_fn(population)
            if self.has_solution:                    
                break
            population = breed_fn(population, outputs, fitness, self.pop_size) 
            del outputs, fitness 
            self.gen += 1

    def _add_final_metrics(self):
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
        init_fn = timed(partial(self.init, **self.init_args), 'init_time', self.metrics)
        eval_fn = timed(partial(self.eval, eval_fn = self.eval_fn), 'eval_time', self.metrics)
        breed_fn = timed(partial(self.breed, **self.breed_args), 'breed_time', self.metrics)
        try:
            self._fit_inner(init_fn, eval_fn, breed_fn)
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
#       CURRENT 2. Lexicase selection and its advanced forms 
#       3. Unification with discrete domains? Can this work with discrete domains?
#       4. Unification with other evo processes in cde-search: NSGA and coevolution.
#       5. Tuning of constants 
#       6. Syntactic simplifications with axioms (again, need Tree Tries to match rules)
#       7. Towards abstract forms (x * x + c * x + c)
#       8. Towards semantic GP (add operators) + propose tuned point operator, using indices
#       9. Math properties and dynamic constraint sets.


#       [BAD IDEA] 10. Gen math expr instead of lisp expr 

#      11. Other metrics??? Add when caches are enabled - syntactic diversity (is there convergance to same syntax)
#      12. Elitism??? Aging??? 

# NOTE [BAD IDEA]: we should probably go with const identities: 10 constant - so we allocate 10 identities but have different their bindings ??
# PROBLEM: 1. const identity should have max of 1 presence in the term, it seems that it should be this way, or small number???
#          2. On crossover of const identities, bindings should be transfered to children - should or not??? should
# NOTE: this is bad idea (attempted) - no benefit to move from consts list to dict[Term, dict[int, Tensor]] for consts 
#           - it complicates logic and requires additional mandatory binding step. 
#       in current implementation we can collect term consts with one traversal if necessary.

if __name__ == "__main__":

    from torch_alg import alg_ops, koza_1
    import json
    
    device = "cuda"
    dtype = torch.float16
    rnd_seed = 42

    solver = GPSolver(ops = alg_ops, device = device, dtype = dtype,
                        rnd_seed = rnd_seed, torch_rnd_seed = rnd_seed,
                        cache_terms=True, cache_term_props=True,
                        cache_evals = True, cache_inner_evals=True,
                        cache_crossover=True,
                        init_args = dict(init_from_cache = False),
                        max_consts=5,
                        max_ops = {"inv": 5, "neg": 5},
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
                            "inv": {"inv": 1}
                        },
                        immediate_arg_limits={
                            "inv": {"inv": 0},
                            "neg": {"neg": 0}
                        }
                        # breed_args= dict(
                        #     selection_fn = lexicase_selection,
                        # ),
                        )

    free_vars, target = koza_1.sample_set("train", device = device, dtype = dtype,
                                            generator=solver.torch_gen,
                                            sorted=True)

    solver.fit(free_vars, target)

    with open("gp-metrics.json", "w") as f:
        json.dump(solver.metrics, indent=4, default=str, fp=f)

    # print("Metrics:\n", metrics_json)

    pass
