''' Population based evolutionary loop and default operators, Koza style GP.
    Operators: 
        1. Initialization: ramped-half-and-half
        2. Selection: tournament
        3. Crossover: one-point subtree 
        4. Mutation: one-point subtree
'''

from dataclasses import dataclass, field
from functools import partial
import math
from typing import Any, Callable, Optional, Sequence
from time import perf_counter
import numpy as np
import torch
from spatial import InteractionIndex, RTreeIndex, SpatialIndex
from term import Term, TermSignature, build_term, evaluate, get_term_pos, one_point_rand_crossover, one_point_rand_mutation, ramped_half_and_half, sign_from_fn, term_sign, term_to_str
from sklearn.base import BaseEstimator, RegressorMixin

def tournament_selection(gp_solver: 'GPSolver', population: list[Term], size: int, tournament_selection_size = 7):
    ''' Fitness is 1d tensor of fitness selected for tournament '''
    fitness = gp_solver.get_fitness(population)
    selected_ids = torch.randint(fitness.shape[0], (size, tournament_selection_size), dtype=torch.int, device=fitness.device)
    selected_fitnesses = fitness[selected_ids]
    best_id_id = torch.argmin(selected_fitnesses, dim=-1)
    best_ids = torch.gather(selected_ids, dim=-1, index = best_id_id.unsqueeze(-1)).squeeze(-1)
    del fitness
    return best_ids

def lexicase_selection(gp_solver: 'GPSolver', population: list[Term], size: int):
    """ Based on Lee Spector's team: Solving Uncompromising Problems With Lexicase Selection """
    # test_ids = np.arange(interactions.shape[1]) # direction is hardcoded 0 - bad, 1 - good
    # default_rnd.shuffle(test_ids)

    # rands = torch.rand((size, interactions.shape[-1]), device=interactions.device)
    # rand_ranks = torch.argsort(rands, dim=-1)

    outcomes = gp_solver.get_outcomes(population)
    target = gp_solver.target
    if target is None:
        raise RuntimeError("Target is not set for lexicase selection")
    interactions = torch.abs(outcomes - target)

    shuffled_test_ids = torch.randperm(interactions.shape[-1], device=interactions.device)
    # candidate_ids = torch.arange(interactions.shape[0], device=interactions.device) # all candidates
    for test_id in shuffled_test_ids:
        test_min_diff = torch.min(interactions[candidate_ids, test_id])
        candidate_id_ids, = torch.where(interactions[candidate_ids, test_id] == test_min_diff)
        candidate_ids = candidate_ids[candidate_id_ids]
        if len(candidate_ids) == 1:
            break
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    best_id_id = torch.randint(len(candidate_ids), (1,), device=interactions.device)
    best_id = candidate_ids[best_id_id]
    return best_id


# fitness = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# res = tournament_selection(10, fitness, tournament_selection_size=3)
# pass

class EvSearchTermination(Exception):
    ''' Reaching maximum of evals, gens, ops etc '''    
    pass 


def mut_cross2_breed(gp_solver: 'GPSolver', population: list[Term], size: int,
                selection_fn = tournament_selection, mutation_fn = one_point_rand_mutation,
                crossover_fn = one_point_rand_crossover, depth_cache = {},
                mutation_rate: float = 0.1, crossover_rate: float = 0.9) -> list[Term]:
    ''' Pipeline that mutates parents and then applies crossover on pairs '''

    new_population = []
    all_parents = []

    selected_ids = selection_fn(gp_solver, population, size)

    mutation_mask = torch.rand(size, device=population.device) < mutation_rate
    
    crossover_mask = torch.rand(size // 2, device=population.device) < crossover_rate

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
        term_poss = get_term_pos(term, depth_cache)
        term_pos_cache[term] = term_poss
        mutated_terms = mutation_fn(
                            term_builder = gp_solver.term_builder, 
                            term = parent, positions = term_poss,
                            leaf_ops = gp_solver.leaf_signatures, branch_ops = gp_solver.ops_signatures, 
                            rnd=gp_solver.rnd, num_children=len(term_p))
        for i, mterm in zip(term_p, mutated_terms):
            mutated_parents[i] = term

    children = list(mutated_parents)
    crossover_pairs = {}
    for should_crossover in crossover_mask_list:
        if should_crossover:
            parent1 = mutated_parents[2 * i]
            parent2 = mutated_parents[2 * i + 1]
            crossover_pairs.setdefault((parent1, parent2), []).append(i)

    for (parent1, parent2), pair_ids in crossover_pairs.items():
        if parent1 not in term_pos_cache:
            term_pos_cache[parent1] = get_term_pos(parent1, depth_cache)
        if parent2 not in term_pos_cache:
            term_pos_cache[parent2] = get_term_pos(parent2, depth_cache)
        pos1 = term_pos_cache[parent1]
        pos2 = term_pos_cache[parent2]
        new_children = crossover_fn(
                            term_builder = gp_solver.term_builder, 
                            term1 = parent1, term2 = parent2, 
                            positions1 = pos1, positions2 = pos2,
                            rnd = gp_solver.rnd, num_children=2 * len(pair_ids))
        for i, ii in enumerate(pair_ids):
            children[2 * ii] = new_children[2 * i]
            children[2 * ii + 1] = new_children[2 * i + 1]

    return children


def init_one(self):
    return ramped_half_and_half(self.term_builder, self.leaves, self.ops, rnd=self.rnd)

def init_each(init_fn: Callable) -> Callable:
    def _init(gp_solver: 'GPSolver', size: int) -> list[Term]:
        res = []
        for _ in range(size):
            term = init_fn(gp_solver.term_builder, gp_solver.leaf_signatures, gp_solver.ops_signatures, 
                            rnd=gp_solver.rnd)
            if term is not None:
                res.append(term)
        return res
    return _init

def mse_loss(predictions, target):
    """ Mean Squared Error loss function """
    return torch.mean((predictions - target) ** 2, dim=-1)

def l1_loss(predictions, target):
    """ L1 loss function """
    return torch.mean(torch.abs(predictions - target), dim=-1)  

class GPSolver(BaseEstimator, RegressorMixin):

    def __init__(self, 
                # target: torch.Tensor,
                # leaves: dict[str | TermSignature, torch.Tensor],
                # branch_ops: dict[str | TermSignature, Callable],
                ops: list[Callable] | dict[str, Callable] = alg_ops_torch, # we refer to  each func by its position in the list 
                fitness_fn: Callable = mse_loss,
                init_fn: Callable = init_each(ramped_half_and_half),
                eval_fn: Callable = evaluate,
                breed_fn: Callable = mut_cross2_breed,
                # fitness_fns: dict[str, Callable],
                # main_fitness_id: int = 0,
            #    sem_store_class: SpatialIndex = InteractionIndex, # RTsreeIndex,
                max_gen: int = 100,
                max_root_evals: int = 100_000, 
                max_evals: int = 500_000,
                pop_size: int = 1000,
                with_caches: bool = False, # enables all caches: syntax, semantic, int, fitness
                rtol = 1e-04, atol = 1e-03,
                rnd_seed: Optional[int | np.random.RandomState] = None,
                device = "cuda"):
        
        if type(ops) is dict:
            self.ops = list(ops.values())
            ops_names = list(ops.keys())
        else:
            self.ops = ops
            ops_names = [ f"f{opi}" if op.__name__ == "<lambda>" else op.__name__ for opi, op in enumerate(ops) ]
        self.ops = ops
        self.ops_signatures = [sign_from_fn(op, op_name) for op_name, op in zip(ops_names, ops)]
        self.ops_map = {sign: self.ops[i] for i, sign in enumerate(self.ops_signatures)}        
        self.fitness_fn = fitness_fn
        self.init_fn = init_fn
        self.eval_fn = eval_fn
        self.breed_fn = breed_fn
        self.max_gen = max_gen
        self.max_root_evals = max_root_evals
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.with_caches = with_caches
        self.rtol = rtol
        self.atol = atol
        self.device = device

        if rnd_seed is None:
            self.rnd = np.random
        elif isinstance(rnd_seed, np.random.RandomState):
            self.rnd = rnd_seed
        else:
            self.rnd = np.random.RandomState(rnd_seed)        


        # next are runtime fields and caches that works across fit calls
        self.target = None 
        # self.free_vars: Optional[Sequence] = None
        self.syntax: dict[tuple[str, ...], Term] = {}
        self.term_outputs: dict[Term, torch.Tensor] = {}
        self.term_fitness: dict[Term, torch.Tensor] = {}
        self.depth_cache: dict[Term, int] = {}
        self.forest = [] # by default == popoulation, but in archive methods it is an archive

        self._reset_state()
        # self.sem_store: SpatialIndex = sem_store_class(capacity = max_evals, dims = target.shape[0], dtype=target.dtype, 
        #                                  device = target.device, rtol=rtol, atol=atol, target = target)

        # self.delayed_semantics: dict[Term, torch.Tensor] = {} # not yet in the store
        # self.leaves: list[Term] = []
        # self.ops = branch_ops
        # self.fitness_fns = fitness_fns
        # self.main_fitness_id = main_fitness_id
        # self.with_caches = True
        # self.rtol = rtol
        # self.atol = atol
        


        # for signature, value in leaves.items():
        #     term = self.term_builder(signature, [])
        #     self.leaves.append(term)
        #     self.set_binding((term, 0), value)
        # self.process_delayed_semantics()
        # self.with_caches = with_caches

        # self.forest: list[int] = [*leaves] # initial forest are leaves

    def _build_free_vars(self, free_vars: Sequence, term_outputs = None) -> dict[Term, torch.Tensor]:
        term_outputs = term_outputs or {}
        for i, xi in enumerate(free_vars):
            fv_name = f"x{i}"
            term = self.term_builder(fv_name, [])                
            if not torch.is_tensor(xi):
                fv = torch.tensor(xi, dtype=self.device)
            else:
                fv = xi.to(self.device)
            term_outputs[term] = fv 
        return term_outputs

    def _reset_state(self, free_vars: Optional[Sequence] = None, target: Optional[Sequence] = None):
        ''' Called before each fit '''
        self.best_term: Optional[Term] = None
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.metrics: dict[str, int | float | list[int|float]] = {}
        self.leaf_signatures = []
        if free_vars is not None:
            self._build_free_vars(free_vars, self.term_outputs)
        if target is None:
            self.target = None
        else:
            if not torch.is_tensor(target):
                self.target = torch.tensor(target, dtype = self.device)
            else:
                self.target = target.to(self.device)

    def term_builder(self, signature: str | TermSignature, args: list[Term]) -> Term:
        def cache_cb(term: Term, cache_hit: bool):
            key = "syntax_hit" if cache_hit else "syntax_miss"
            self.metrics[key] = self.metrics.get(key, 0) + 1
        if self.with_caches:
            term = build_term(self.syntax, signature, args, cache_cb)
        else:
            term = Term(term_sign(signature, args), args)
            cache_cb(term, False)
        return term        

    def timed(self, fn: Callable, key: str) -> Callable:
        ''' Decorator to time function execution '''
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = fn(*args, **kwargs)
            elapsed_time = perf_counter() - start_time
            self.metrics.setdefault(key, []).append(elapsed_time)
            return result
        return wrapper
    
    def _get_binding(self, term_with_pos: tuple[Term, int]) -> Optional[torch.Tensor]:
        term = term_with_pos[0] # in this implementation we ignore position 
        if term in self.term_outputs:
            return self.term_outputs[term]
        # semantic_id = self.term_to_sid.get(term, None)
        # if semantic_id is None:
        #     return None
        # res = self.sem_store.get_vectors(semantic_id)
        return None 

    def _set_binding(self, term_with_pos: tuple[Term, int], value: torch.Tensor):
        self.evals += 1
        if not self.with_caches:
            return
        term = term_with_pos[0] # ignoring position currently
        self.term_outputs[term] = value

    # def process_delayed_semantics(self):
    #     if len(self.delayed_semantics) == 0:
    #         return set()
    #     terms = list(self.delayed_semantics.keys())
    #     semantics = torch.stack(list(self.delayed_semantics.values()))
    #     sem_ids = self.sem_store.insert(semantics)
    #     new_sem_ids = set()
    #     for sem_id in sem_ids:
    #         if sem_id not in self.sid_to_terms:
    #             new_sem_ids.add(sem_id)
    #     for term, sem_id in zip(terms, sem_ids):
    #         self.term_to_sid[term] = sem_id
    #         self.sid_to_terms.setdefault(sem_id, []).append(term)

    #     self.delayed_semantics = {}
    #     return new_sem_ids

    def eval(self, terms: list[Term]) -> bool:
        terms_for_fitness = []
        for term in terms:
            self.eval_fn(term, self.ops_map, self._get_binding, self._set_binding)
            # sids_set = context.process_delayed_semantics() # semantics of tree term
            # sids.update(sids_set)
            if term not in self.term_fitness:
                terms_for_fitness.append(term)
        if len(terms_for_fitness) == 0:
            return False 
        predictions = torch.stack([self.term_outputs[t] for t in terms_for_fitness ])
        # fitness = torch.zeros(semantics.shape[0], len(self.fitness_fns), dtype=semantics.dtype, device=semantics.device)
        fitness: torch.Tensor = self.fitness_fn(predictions, self.target)
        # for fn_id, fn in enumerate(self.fitness_fns.values()):
        #     fitness[:, fn_id] = fn(self, semantics)
        for term, term_fitness in zip(terms_for_fitness, fitness):
            self.term_fitness[term] = term_fitness

        best_id = fitness.argmin().item()
        best_term = terms_for_fitness[best_id]
        best_fitness = fitness[best_id]
        cur_best_fitness = self.term_fitness.get(self.best_term, None)
        if cur_best_fitness is None or (best_fitness <= self.best_fitness):
            self.best_term = best_term    
        return fitness  
    
    def get_fitness(self, terms: list[Term]) -> torch.Tensor:
        fitness = torch.tensor([self.term_fitness[term] for term in terms], device=terms.device)
        return fitness
    
    def get_outcomes(self, terms: list[Term]) -> torch.Tensor:
        outcomes = torch.stack([self.term_outputs[term] for term in terms], device=terms.device)
        return outcomes
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GPSolver':
        """
        Fit the solver to the data.

        Args:
            X (array-like): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns the instance itself.
        """
        self._reset_state(free_vars=X, target=y)
        try:
            self.forest = self.init_fn(self, self.pop_size)
            while self.gen < self.max_gen and self.evals < self.max_evals and self.root_evals < self.max_root_evals:
                solution = self.eval(population)
                if solution:
                    break        
                population = self.breed_fn(self, population, self.pop_size)  
                self.gen += 1
        except EvSearchTermination as e:
            pass
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained solver.

        Args:
            X (array-like): Input features.

        Returns:
            array-like: Predicted values.
        """
        if not self.is_fitted_ or self.best_term is None:
            raise RuntimeError("Solver is not fitted yet")
        
        term_outputs = self._build_free_vars(free_vars)
        
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

    from torch_alg import alg_ops_torch, koza_1

    free_vars, target = koza_1.sample_set("train")

    solver = GPSolver()

    solver.fit(free_vars, target)

    print("Best term:", term_to_str(solver.best_term))
    pass
