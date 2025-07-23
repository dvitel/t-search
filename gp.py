''' Population based evolutionary loop and default operators, Koza style GP.
    Operators: 
        1. Initialization: ramped-half-and-half
        2. Selection: tournament
        3. Crossover: one-point subtree 
        4. Mutation: one-point subtree
'''

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
import math
from typing import Any, Callable, Literal, Optional, Sequence, Type
from time import perf_counter
import numpy as np
import torch
from spatial import InteractionIndex, RTreeIndex, SpatialIndex
from term import UNTYPED, Op, Term, TermType, Value, Variable, cache_term, evaluate, get_callable_signature, get_term_pos, one_point_rand_crossover, one_point_rand_mutation, ramped_half_and_half, term_to_str
from sklearn.base import BaseEstimator, RegressorMixin

#utils

def stack_rows(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    max_size = max(ti.shape[0] for ti in tensors)
    res = torch.empty((len(tensors), max_size), dtype=tensors[0].dtype, device=tensors[0].device)
    for i, ti in enumerate(tensors):
        res[i] = ti # assuming broadcastable
    return res

# selections
def tournament_selection(population: list[Term], size: int, *, 
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

def lexicase_selection(population: list[Term], size: int, *, 
                       outputs: torch.Tensor, target: torch.Tensor, 
                       gen: torch.Generator, **_):
    """ Based on Lee Spector's team: Solving Uncompromising Problems With Lexicase Selection """
    # test_ids = np.arange(interactions.shape[1]) # direction is hardcoded 0 - bad, 1 - good
    # default_rnd.shuffle(test_ids)

    # rands = torch.rand((size, interactions.shape[-1]), device=interactions.device)
    # rand_ranks = torch.argsort(rands, dim=-1)

    should_free = False
    if not torch.is_tensor(outputs):        
        outputs = stack_rows(outputs)
        should_free = True    

    interactions = torch.abs(outputs - target)

    shuffled_test_ids = torch.randperm(interactions.shape[-1], device=interactions.device,
                                        generator=gen)
    # candidate_ids = torch.arange(interactions.shape[0], device=interactions.device) # all candidates
    for test_id in shuffled_test_ids:
        test_min_diff = torch.min(interactions[candidate_ids, test_id])
        candidate_id_ids, = torch.where(interactions[candidate_ids, test_id] == test_min_diff)
        candidate_ids = candidate_ids[candidate_id_ids]
        if len(candidate_ids) == 1:
            break
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    best_id_id = torch.randint(len(candidate_ids), (1,), device=interactions.device,
                                generator=gen)
    best_id = candidate_ids[best_id_id]
    del interactions, shuffled_test_ids
    if should_free:
        del outputs
    return best_id


# fitness = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# res = tournament_selection(10, fitness, tournament_selection_size=3)
# pass

class EvSearchTermination(Exception):
    ''' Reaching maximum of evals, gens, ops etc '''    
    pass 


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
    
def mse_loss(predictions, target, *, nan_error = torch.inf):
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
                ops: list[Callable] | dict[str, Callable], # we refer to each func by its position in the list 
                fitness_fn: Callable = partial(mse_loss_nan_vf, nan_frac=0.2), 
                fit_condition = partial(fit_0, rtol = 1e-04, atol = 1e-03),
                init_args: dict = dict(init_fn = ramped_half_and_half),
                eval_args: dict = dict(eval_fn = evaluate),
                breed_args: dict = dict(
                    selection_fn = tournament_selection,
                    mutation_fn = one_point_rand_mutation,
                    crossover_fn = one_point_rand_crossover,
                    mutation_rate = 0.1,
                    crossover_rate = 0.9,
                ),
                max_consts: int = 5, # 0 to disable consts in terms
                max_vars: int = 10, # max number of free variables
                max_gen: int = 100,
                max_root_evals: int = 10_000, 
                max_evals: int = 100_000,
                pop_size: int = 1000,
                with_caches: bool = False, # enables all caches: syntax, semantic, int, fitness
                rtol = 1e-04, atol = 1e-03, # NOTE: these are for semantic/outputs comparison, not for fitness, see fit_0
                rnd_seed: Optional[int] = None,
                torch_rnd_seed: Optional[int] = None,
                device = "cpu", dtype = torch.float32,
                debug = True, # flag will rewrite Term str and repr 
                ):
        
        if type(ops) is dict:
            self.ops = list(ops.values())
            self.ops_names = list(ops.keys())
        else:
            self.ops = ops
            self.ops_names = [ f"f{opi}" if op.__name__ == "<lambda>" else op.__name__ for opi, op in enumerate(ops) ]
        self.all_signatures = [get_callable_signature(op) for op in self.ops]
        self.default_leaves = [((sign, Op, op_id), lambda op_id = op_id:Op(op_id, ())) 
                          for op_id, sign in enumerate(self.all_signatures) if sign.arity() == 0]        
        self.branches = [(sign, Op, op_id) for op_id, sign in enumerate(self.all_signatures) if sign.arity() > 0]
        self.max_consts = max_consts
        self.max_vars = max_vars
        self.debug = debug

        if self.debug:

            solver = self

            def _term_to_str(self: Term):
                return term_to_str(self, name_getter=lambda x: solver._get_name(*x))
            
            Term.__str__ = _term_to_str
            Term.__repr__ = _term_to_str

        self.vars: list[torch.Tensor] = []
        self.var_names: list[str] = []
        self.consts: list[torch.Tensor] = []
        self.const_range: tuple[torch.Tensor, torch.Tensor] | None = None # detected from y on reset
        self.term_consts: dict[Term, list[int]] = {} # for each term stores all present cocnstants
        self.term_vars: dict[Term, list[int]] = {}            # tree root to variable occurances
        self.term_counts: dict[Term, dict[tuple, int]] = {}
        # NOTE: variables and consts are stored separately from tree - abstract shapes x * x + c * x + c 
        #       in this approach we have a problem with caching semantics of intermediate terms, as for different c and x, the results are different
        #       solution: make term_output as dictionary with keys (root, term). Root should be a part of all keys to identify concrete selection of c, x
        #       alternative: create subclasses of Term for Vars and Values - this is more explicit approach and better 
        #                    Vars = Term + var id, Values = Term + value Any.
        #                    Do we need (term, occur) in this case? Seems yes.
        self.count_constraints = {}
        if self.max_consts > 0:            
            const_signature = (UNTYPED, Value)
            self.count_constraints[const_signature] = self.max_consts
            self.default_leaves.append((const_signature, self._alloc_const))
        if self.max_vars > 0:
            var_signature = (UNTYPED, Variable)
            self.count_constraints[var_signature] = self.max_vars
            # self.leaves.append(self._alloc_var)
        self.fitness_fn = fitness_fn
        self.fit_condition = fit_condition
        self.init_args = init_args
        self.eval_args = eval_args
        self.breed_args = breed_args
        self.max_gen = max_gen
        self.max_root_evals = max_root_evals
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.with_caches = with_caches
        # effect of with_caches:
        # 1. Syntax cache: True - forms cache, False - each time new instance
        # 2. Inner semantics: True - preserves, False - only root is considered
        # 3. Semantics cache: True - preserves all seen semantics, 
        #                     False - preserves only recently evaluated roots
        # 4. Fitness cache: True - preserves all seen fitness, 
        #                   False - only recently evaluated roots and best 
        # 5. Depth cache: True - preserves all seen depths,
        #                 False - only last gen 
        # NOTE: later this bool parameter could be split to control each cache
        self.rtol = rtol
        self.atol = atol
        self.device = device
        self.dtype = dtype

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
        self.syntax: dict[tuple[str, ...], Term] = {}
        self.term_outputs: dict[Term, torch.Tensor] = {}
        self.inner_semantics: dict[Term, dict[Term, torch.Tensor]] = {} # not yet in term_fitness and are not roots
        self.term_fitness: dict[Term, torch.Tensor] = {}
        # self.depth_cache: dict[Term, int] = {}

        self.best_term: Optional[Term] = None
        self.best_outputs: Optional[torch.Tensor] = None
        self.best_fitness: Optional[torch.Tensor] = None
        self.has_solution = False
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.metrics: dict[str, int | float | list[int|float]] = {}
        self.leaves = list(self.default_leaves)

    def _get_name(self, term_type: TermType, tp: Type, term_id: Any) -> Optional[str]:
        if tp is Op:
            return self.ops_names[term_id]
        if tp is Value: 
            return self.consts[term_id].item()
        if tp is Variable:
            return self.var_names[term_id]

    def _alloc_const(self) -> Value: 
        ''' Should we random sample of try some grid? Anyway we tune '''
        value = self.const_range[0] + torch.rand((1,), device=self.device, dtype=self.dtype,
                                                    generator=self.torch_gen) * self.const_range[1]
        const_id = len(self.consts)
        self.consts.append(value)
        return Value(const_id)

    def _add_free_vars(self, free_vars: Sequence):
        if self.max_vars == 0:
            return
        for i, xi in enumerate(free_vars):
            if not torch.is_tensor(xi):
                fv = torch.tensor(xi, dtype=self.dtype, device=self.device)
            else:
                fv = xi.to(device = self.device, dtype = self.dtype)
            self.vars.append(fv)
            self.var_names.append(f"x{i}")
            self.leaves.append(((UNTYPED, Variable), lambda i=i: Variable(i)))

    def init(self, size: int, *, init_fn: Callable = ramped_half_and_half, **kwargs) -> list[Term]:
        ''' Initialize each term in population 0 with self.init_fn '''
        res = []
        for _ in range(size):
            term = init_fn(self._cache_term, self.leaves, self.branches, 
                            count_constraints = self.get_count_constraints(),
                            rnd=self.rnd)
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

        if self.with_caches: # also means with inner semantics of terms 
            new_term_ids = [tid for tid, term in enumerate(terms) if term not in self.term_fitness]

            if len(new_term_ids) > 0:
                
                all_outputs = [output_list[tid] for tid in new_term_ids]
                all_terms = [terms[tid] for tid in new_term_ids]
                for tid in new_term_ids:
                    for inner_term, inner_outputs in self.inner_semantics.get(terms[tid]).items():
                        all_terms.append(inner_term)
                        all_outputs.append(inner_outputs)
                self.inner_semantics.clear()

                predictions = stack_rows(all_outputs)
                new_fitness: torch.Tensor = self.fitness_fn(predictions, self.target)
                del predictions
                for t, f in zip(all_terms, new_fitness):
                    self.term_fitness[t] = f
                best_id, best_found = self.fit_condition(new_fitness, self.term_fitness.get(self.best_term, None))
                if best_id is not None:
                    self.best_term = all_terms[best_id]
                    self.best_outputs = predictions[best_id].clone()
                    self.best_fitness = new_fitness[best_id].clone()

            outputs = output_list                    
            fitness = [self.term_fitness[t] for t in terms]

        else:

            outputs = stack_rows(output_list)
            fitness = self.fitness_fn(outputs, self.target) 
            best_id, best_found = self.fit_condition(fitness, self.term_fitness.get(self.best_term, None))
            if best_id is not None:
                self.best_term = terms[best_id]
                self.best_outputs = outputs[best_id].clone()
                self.best_fitness = fitness[best_id].clone()    
        
        self.has_solution = best_found

        return fitness, outputs      
    
    def breed(self, population: list[Term], 
                fitness: torch.Tensor | Sequence[torch.Tensor], 
                outputs: torch.Tensor | Sequence[torch.Tensor], 
                size: int, *,
                selection_fn: Callable[[list[Term], int], torch.Tensor] = tournament_selection, 
                mutation_fn: Callable = one_point_rand_mutation, 
                crossover_fn: Callable = one_point_rand_crossover,
                mutation_rate = 0.1, crossover_rate = 0.9, **_) -> list[Term]:
        ''' Pipeline that mutates parents and then applies crossover on pairs. One-point operations '''

        new_population = []
        all_parents = []

        selected_ids = selection_fn(population, size, 
                                    fitness=fitness, outputs=outputs,
                                    target = self.target, gen=self.torch_gen)

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

        term_pos_cache = {}
        for term, term_p in mutation_pos.items():
            term_poss = get_term_pos(term)
            term_pos_cache[term] = term_poss
            mutated_terms = mutation_fn(self._cache_term,
                                term = term, positions = term_poss,
                                leaves = self.leaves, ops = self.branches,
                                count_constraints = self.get_count_constraints(),
                                count_cache = self.term_counts if self.with_caches else None,
                                rnd=self.rnd, num_children=len(term_p))
            for i, mterm in zip(term_p, mutated_terms):
                mutated_parents[i] = mterm

        children = list(mutated_parents)
        crossover_pairs = {}
        for i, should_crossover in enumerate(crossover_mask_list):
            if should_crossover:
                parent1 = mutated_parents[2 * i]
                parent2 = mutated_parents[2 * i + 1]
                crossover_pairs.setdefault((parent1, parent2), []).append(i)

        depth_cache = {} # temporary collects depths - it is only needed in crossover
        for (parent1, parent2), pair_ids in crossover_pairs.items():
            if parent1 not in term_pos_cache:
                term_pos_cache[parent1] = get_term_pos(parent1)
            if parent2 not in term_pos_cache:
                term_pos_cache[parent2] = get_term_pos(parent2)
            pos1 = term_pos_cache[parent1]
            pos2 = term_pos_cache[parent2]
            new_children = crossover_fn(self._cache_term,
                                term1 = parent1, term2 = parent2, 
                                positions1 = pos1, positions2 = pos2,
                                count_constraints = self.get_count_constraints(),
                                depth_cache = depth_cache,
                                count_cache = self.term_counts if self.with_caches else None,
                                rnd = self.rnd, num_children=2 * len(pair_ids))
            for i, ii in enumerate(pair_ids):
                children[2 * ii] = new_children[2 * i]
                children[2 * ii + 1] = new_children[2 * i + 1]

        return children

    def _reset_state(self, free_vars: Optional[Sequence] = None, target: Optional[Sequence] = None):
        ''' Called before each fit '''

        # reset caches 
        self.vars: list[torch.Tensor] = []
        self.var_names: list[str] = []
        self.consts: list[torch.Tensor] = []
        self.term_consts: dict[Term, list[int]] = {} # for each term stores all present cocnstants
        self.term_vars: dict[Term, list[int]] = {}            # tree root to variable occurances
        self.term_counts: dict[Term, dict[tuple, int]] = {}

        self.target = None 
        self.syntax: dict[tuple[str, ...], Term] = {}
        self.term_outputs: dict[Term, torch.Tensor] = {}
        self.inner_semantics: dict[Term, dict[Term, torch.Tensor]] = {}
        self.term_fitness: dict[Term, torch.Tensor] = {}
        # self.depth_cache: dict[Term, int] = {}

        self.best_term: Optional[Term] = None
        self.best_outputs: Optional[torch.Tensor] = None
        self.best_fitness: Optional[torch.Tensor] = None
        self.has_solution = False
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.metrics: dict[str, int | float | list[int|float]] = {}
        self.leaves = list(self.default_leaves) 

        if free_vars is not None:
            self._add_free_vars(free_vars)
        if target is None:
            self.target = None
        else:
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

    def _cache_term(self, term: Term) -> Term:
        def cache_cb(term: Term, cache_hit: bool):
            key = "syntax_hit" if cache_hit else "syntax_miss"
            self.metrics[key] = self.metrics.get(key, 0) + 1
        if self.with_caches:
            term = cache_term(self.syntax, term, cache_cb)
        else:
            cache_cb(term, False)
        return term     

    def get_count_constraints(self) -> dict[tuple, int]:
        return dict(self.count_constraints)   
    
    def _get_binding(self, root: Term, term: Term) -> Optional[torch.Tensor]:        
        if isinstance(term, Variable):
            return self.vars[term.var_id]
        if isinstance(term, Value):
            return self.consts[term.value_id]

        # next is unnecessary as data never lands into term_outputs
        # if not self.with_caches:
        #     return None        

        if term in self.term_outputs:
            return self.term_outputs[term]

        return None 

    def _set_binding(self, root: Term, term: Term, value: torch.Tensor):
        self.evals += 1
        if root == term:
            self.root_evals += 1
        if self.evals == self.max_evals:
            raise EvSearchTermination("MAX_EVALS")
        if self.root_evals == self.max_root_evals:
            raise EvSearchTermination("MAX_ROOT_EVALS")
        if not self.with_caches:
            return
        self.term_outputs[term] = value
        if term != root:
            self.inner_semantics.setdefault(root, {})[term] = value    
    
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
        init_fn = timed(partial(self.init, **self.init_args), 'init_time', self.metrics)
        eval_fn = timed(partial(self.eval, **self.eval_args), 'eval_time', self.metrics)
        breed_fn = timed(partial(self.breed, **self.breed_args), 'breed_time', self.metrics)
        try:
            population = init_fn(self.pop_size)
            while self.gen < self.max_gen and self.evals < self.max_evals and self.root_evals < self.max_root_evals:
                outputs, fitness = eval_fn(population)
                if self.has_solution:
                    break
                population = breed_fn(population, outputs, fitness, self.pop_size) 
                del outputs, fitness 
                self.gen += 1
        except EvSearchTermination as e:
            pass
        self.is_fitted_ = True
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
        
        term_outputs = self._add_free_vars(free_vars)
        
        def get_binding(term_with_pos: tuple[Term, int]) -> Optional[torch.Tensor]:
            return term_outputs.get(term_with_pos[0], None)
        
        def set_binding(term_with_pos: tuple[Term, int], value: torch.Tensor):
            pass 
        
        output = self.eval_fn(self.best_term, self.ops_map, get_binding, set_binding)
        if output is None:
            raise RuntimeError("Evaluation of the best term returned None, not all terminals may be bound")
        output_numpy = output.cpu().numpy()
        return output_numpy
    
if __name__ == "__main__":

    from torch_alg import alg_ops, koza_1
    
    device = "cuda"
    dtype = torch.float16
    rnd_seed = 42

    solver = GPSolver(ops = alg_ops, device = device, dtype = dtype,
                        rnd_seed = rnd_seed, torch_rnd_seed = rnd_seed)

    free_vars, target = koza_1.sample_set("train", device = device, dtype = dtype,
                                            generator=solver.torch_gen,
                                            sorted=True)


    solver.fit(free_vars, target)

    print("Best term:", term_to_str(solver.best_term))
    pass
