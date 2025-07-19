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
from term import Term, TermSignature, build_term, evaluate, get_term_pos, one_point_rand_crossover, one_point_rand_mutation, ramped_half_and_half, term_sign, term_to_str

def tournament_selection(size: int, fitness: torch.Tensor, tournament_selection_size = 7):
    ''' Fitness is 1d tensor of fitness selected for tournament '''
    selected_ids = torch.randint(fitness.shape[0], (size, tournament_selection_size), dtype=torch.int, device=fitness.device)
    selected_fitnesses = fitness[selected_ids]
    best_id_id = torch.argmin(selected_fitnesses, dim=-1)
    best_id = torch.gather(selected_ids, dim=-1, index = best_id_id.unsqueeze(-1)).squeeze(-1)
    return best_id

def lexicase_selection(size: int, interactions: torch.Tensor):
    """ Based on Lee Spector's team: Solving Uncompromising Problems With Lexicase Selection """
    # test_ids = np.arange(interactions.shape[1]) # direction is hardcoded 0 - bad, 1 - good
    # default_rnd.shuffle(test_ids)

    # rands = torch.rand((size, interactions.shape[-1]), device=interactions.device)
    # rand_ranks = torch.argsort(rands, dim=-1)

    shuffled_test_ids = torch.randperm(interactions.shape[-1], device=interactions.device)
    # candidate_ids = torch.arange(interactions.shape[0], device=interactions.device) # all candidates
    for test_id in shuffled_test_ids:
        test_max_outcome = torch.max(interactions[candidate_ids, test_id])
        candidate_id_ids, = torch.where(interactions[candidate_ids, test_id] == test_max_outcome)
        candidate_ids = candidate_ids[candidate_id_ids]
        if len(candidate_ids) == 1:
            break
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    best_id_id = torch.randint(len(candidate_ids), (1,), device=interactions.device)
    best_id = candidate_ids[best_id_id]
    return best_id


fitness = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
res = tournament_selection(10, fitness, tournament_selection_size=3)
pass

class EvSearchTermination(Exception):
    ''' Reaching maximum of evals, gens, ops etc '''    
    pass 

class GPEvSearch():

    def __init__(self, target: torch.Tensor,
                       leaves: dict[str | TermSignature, torch.Tensor],
                       branch_ops: dict[str | TermSignature, Callable],
                       fitness_fns: dict[str, Callable],
                       main_fitness_id: int = 0,
                    #    sem_store_class: SpatialIndex = InteractionIndex, # RTreeIndex,
                       max_gen: int = 100,
                       max_root_evals: int = 100_000, 
                       max_evals: int = 500_000,
                       pop_size: int = 1000,
                       with_caches: bool = False, # enables all caches: syntax, semantic, int, fitness
                       rtol = 1e-04, atol = 1e-03,
                       mutation_rate: float = 0.1,
                       crossover_rate: float = 0.9,
                       rnd_seed: Optional[int | np.random.RandomState] = None):
        assert len(leaves) > 0, "At least one leaf should be provided"
        self.target = target        
        # self.term_to_sid: dict[Term, int] = {} # term to semantic id        
        # self.sid_to_terms: dict[int, list[Term]] = {} # semantic id to terms

        self.syntax: dict[tuple[str, ...], Term] = {}
        self.term_outputs: dict[Term, torch.Tensor] = {}
        self.term_fitness: dict[Term, torch.Tensor] = {}
        # self.sem_store: SpatialIndex = sem_store_class(capacity = max_evals, dims = target.shape[0], dtype=target.dtype, 
        #                                  device = target.device, rtol=rtol, atol=atol, target = target)

        # self.delayed_semantics: dict[Term, torch.Tensor] = {} # not yet in the store
        self.leaves: list[Term] = []
        self.ops = branch_ops
        self.fitness_fns = fitness_fns
        self.main_fitness_id = main_fitness_id
        self.best_term: Optional[Term] = None
        self.best_fitness: Optional[torch.Tensor] = None
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.mutation_rate: float = mutation_rate
        self.crossover_rate: float = crossover_rate
        self.max_gen: int = max_gen
        self.max_evals: int = max_evals
        self.mex_root_evals: int = max_root_evals
        self.pop_size: int = pop_size
        self.metrics: dict[str, int | float | list[int|float]] = field(default_factory=dict)
        self.with_caches = True
        self.rtol = rtol
        self.atol = atol
        self.depth_cache = {}
        if rnd_seed is None:
            self.rnd = np.random
        elif isinstance(rnd_seed, np.random.RandomState):
            self.rnd = rnd_seed
        else:
            self.rnd = np.random.RandomState(rnd_seed)

        for signature, value in leaves.items():
            term = self.term_builder(signature, [])
            self.leaves.append(term)
            self.set_binding((term, 0), value)
        # self.process_delayed_semantics()
        self.with_caches = with_caches

        self.forest: list[int] = [*leaves] # initial forest are leaves

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
            evaluate(term, self.ops, self._get_binding, self._set_binding)
            # sids_set = context.process_delayed_semantics() # semantics of tree term
            # sids.update(sids_set)
            if term not in self.term_fitness:
                terms_for_fitness.append(term)
        if len(terms_for_fitness) == 0:
            return False 
        semantics = torch.stack([self.term_outputs[t] for t in terms_for_fitness ])
        fitness = torch.zeros(semantics.shape[0], len(self.fitness_fns), dtype=semantics.dtype, device=semantics.device)
        for fn_id, fn in enumerate(self.fitness_fns.values()):
            fitness[:, fn_id] = fn(self, semantics)
        for term, fitness in zip(terms_for_fitness, fitness):
            self.term_fitness[term] = fitness

        best_id = fitness[:, self.main_fitness_id].argmin().item()
        best_term = terms[best_id]
        best_fitness = fitness[best_id]
        if self.best_fitness is None or torch.all(best_fitness <= self.best_fitness):
            self.best_fitness = best_fitness
            self.best_term = best_term    
        return fitness  

    def init_one(self):
        return ramped_half_and_half(self.term_builder, self.leaves, self.ops, rnd=self.rnd)
    
    def init(self, size):
        res = []
        init_fn = partial(init_fn, rnd=self.rnd) # rnd is mandatory in init_fn - we can check signature but anyway
        for _ in range(size):
            term = self.init_one()
            if term is not None:
                res.append(term)
        return res
    
    def select_for_breed(self, population: list[Term], size: int):
        fitness = torch.tensor([self.term_fitness[term][self.main_fitness_id] for term in population], device=population.device)
        selected_ids = tournament_selection(size, fitness, )
        del fitness 
        return selected_ids
    
    def breed(self, population: list[Term], size: int):
        ''' Implements default 1 point crossover and mutation, allows root modification'''

        new_population = []
        all_parents = []

        selected_ids = self.select_for_breed(population, size)

        mutation_mask = torch.rand(size, device=population.device) < self.mutation_rate
        
        crossover_mask = torch.rand(size // 2, device=population.device) < self.crossover_rate

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


        depth_cache = self.depth_cache if self.with_caches else {}

        term_pos_cache = {}
        for term, term_p in mutation_pos.items():
            term_poss = get_term_pos(term, depth_cache)
            term_pos_cache[term] = term_poss
            mutated_terms = one_point_rand_mutation(self.term_builder, parent, term_poss,
                                    self.leaves, self.ops, rnd=self.rnd, 
                                    num_children=len(term_p), 
                                    )
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
            new_children = one_point_rand_crossover(self.term_builder, parent1, parent2, pos1, pos2,
                                     rnd = self.rnd, num_children=2 * len(pair_ids))
            for i, ii in enumerate(pair_ids):
                children[2 * ii] = new_children[2 * i]
                children[2 * ii + 1] = new_children[2 * i + 1]

        return children


    def run(self):
        solution = None
        try:
            self.forest = self.init(self.pop_size)
            while self.gen < self.max_gen and self.evals < self.max_evals and self.root_evals < self.max_root_evals:
                solution = self.eval(population)
                if solution:
                    break        
                population = self.breed(self, population, self.pop_size)  
                self.gen += 1
        except EvSearchTermination as e:
            solution = self.best
        return solution
    
if __name__ == "__main__":

    from torch_alg import alg_ops_torch, koza_1

    free_vars, target = koza_1.sample_set("train")

    process = GPEvSearch(target)        

    process.run()