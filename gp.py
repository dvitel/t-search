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
from typing import Callable, Optional, Sequence
from time import perf_counter
import numpy as np
import torch
from term import Term, TermBindings, TermSignature, build_term, evaluate, ramped_half_and_half


class EvSearchTermination(Exception):
    ''' Reaching maximum of evals, gens, ops etc '''
    pass 
  
class WithTermBindings:
    def __init__(self, context: "GPEvSearch"):
        self.context = context
        self.leaf_bindings = None
        self.bindings = None
    
    def __enter__(self):
        ''' Enter context manager, set bindings '''
        self.leaf_bindings = {self.context.id_to_term[term_id]:self.context.semantics[term_id] for term_id in self.context.leaves}
        self.bindings = TermBindings(self.leaf_bindings, frozen=not self.context.with_caches)
        return self.bindings
    
    def __exit__(self, exc_type, exc_value, traceback):
        ''' Exit context manager, clear bindings '''
        self.context.evals += self.bindings.misses # subtree evals added
        self.leaf_bindings = None
        self.bindings = None

        return False

class GPEvSearch():

    def __init__(self, target: torch.Tensor,
                       leaves: dict[str, torch.Tensor],
                       branch_ops: dict[str | TermSignature, Callable],
                       fitness_fns: list[Callable],
                       max_gen: int = 100,
                       max_root_evals: int = 100000, # 1000 inds per 100 gen
                       max_evals: int = 10000000, # 1000 inds per 100 gen per 100 ops in ind
                       pop_size: int = 1000,
                       with_caches: bool = False, # enables all caches: syntax, semantic, int, fitness
                       rtol = 1e-05, atol = 1e-08,
                       rnd_seed: Optional[int | np.random.RandomState] = None):
        self.target = target
        self.term_to_id: dict[Term, int] = {} 
        self.id_to_term: dict[int, Term] = {}
        self.syntax: dict[tuple[str, ...], Term] = {}
        self.semantics = torch.zeros(max_evals, target.shape[0], dtype=torch.float16, device=target.device)
        self.interactions = torch.zeros(max_evals, target.shape[0], dtype=torch.uint8, device=target.device)
        self.fitness = torch.zeros(max_evals, len(fitness_fns), dtype=torch.float16, device=target.device)
        self.term_id = 0 # next id to allocate
        self.leaves: list[int] = []
        self.ops = branch_ops
        self.fitness_fns = fitness_fns
        self.best_ids: list[int] = [] # best individuals found so far in the forest        
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.max_gen: int = max_gen
        self.max_evals: int = max_evals
        self.mex_root_evals: int = max_root_evals
        self.pop_size: int = pop_size
        self.metrics: dict[str, int | float | list[int|float]] = field(default_factory=dict)
        self.with_caches = with_caches
        # self.sem_cache: dict[Term, torch.Tensor] = dict(leaves)
        # self.int_cache: dict[Term, torch.Tensor] = {}
        # self.fitness_cache = dict[Term, torch.Tensor] = {}
        self.rtol = rtol
        self.atol = atol
        # self.last_outputs: dict[Term, torch.Tensor] = {}
        # self.last_interactions: dict[Term, torch.Tensor] = {}
        # self.last_fitness: dict[Term, torch.Tensor] = {}
        if rnd_seed is None:
            self.rnd = np.random
        elif isinstance(rnd_seed, np.random.RandomState):
            self.rnd = rnd_seed
        else:
            self.rnd = np.random.RandomState(rnd_seed)

        for name, value in leaves.items():
            term = build_term(self.syntax, name) 
            cur_term_id = self._alloc_term(term)
            self.leaves.append(cur_term_id)
            self.semantics[cur_term_id] = value
        self.set_proximity(target, self.leaves) # set proximity for leaves
        self.set_fitness(self.leaves)

        self.forest: list[int] = [*leaves]

    def _alloc_term(self, term: Term) -> int:
        if term in self.term_to_id:
            return self.term_to_id[term]
        cur_term_id = self.term_id 
        self.term_id += 1
        self.term_to_id[term] = cur_term_id
        self.id_to_term[cur_term_id] = term
        return cur_term_id

    def set_proximity(self, target: torch.Tensor, term_ids: list[int]) -> torch.Tensor:
        ''' Important: semantics should be set for term_ids before this call '''
        outputs = self.semantics[term_ids] # get outputs for the terms
        proximity = torch.isclose(outputs, target.unsqueeze(0), rtol=self.rtol, atol=self.atol).to(dtype=torch.uint8) #∣inputi​−otheri​∣≤rtol×∣otheri​∣+atol
        self.interactions[term_ids] = proximity # store interactions in the cache
        pass
    
    def set_fitness(self, term_ids: list[int]):
        semantics = self.semantics[term_ids]
        interactions = self.interactions[term_ids]
        for fitness_id, fitness_fn in enumerate(self.fitness_fns):
            fitness_fn(self.fitness[term_ids, fitness_id], semantics, interactions)
        pass

    def term_builder(self, signature: TermSignature, args: list[Term]) -> Term:
        def cache_cb(term: Term, cache_hit: bool):
            key = "syntax_cache_hit" if cache_hit else "syn_cache_miss"
            self.metrics[key] = self.metrics.get(key, 0) + 1
        if self.with_caches:
            term = build_term(self.syntax, signature, args, cache_cb)
        else:
            term = Term(signature, args)
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
    
    def new_bindings(self) -> TermBindings:
        leaf_bindings = {self.id_to_term[term_id]:self.semantics[term_id] for term_id in self.leaves}
        bindings = TermBindings(leaf_bindings, frozen=not self.with_caches)
        return bindings
    
    # def get_ints(self, terms: list[Term]):
    #     if self.with_int_cache:
    #         context.last_interactions = {}
    #         missed_terms = []
    #         for t in term:
    #             if t in context.int_cache:
    #                 context.last_interactions[t] = context.int_cache[t]
    #             else:
    #                 missed_terms.append(t)
    #         if len(missed_terms) > 0:            
    #             new_interactions = int_fn(context, missed_terms)
    #             for term, ints in zip(missed_terms, new_interactions):
    #                 context.int_cache[term] = ints
    #                 context.last_interactions[term] = ints        
    
    # def init_fn(self, size: int) -> Sequence[Term]:
    #     pass

    # def eval_fn(self, terms: Sequence[Term]) -> torch.Tensor:
    #     pass

    # def analyze_fn(self, population: Sequence[Term], fitness: torch.Tensor) -> Optional[Term]:
    #     pass 

    # def breed_fn(self, population: Sequence[Term], fitness: torch.Tensor) -> Sequence[Term]:
    #     pass

    def run(self,
                init_fn: Callable[["GPEvSearch", int], Sequence[Term]], 
                eval_fn: Callable[["GPEvSearch", Sequence[Term]], torch.Tensor], 
                analyze_fn: Callable[["GPEvSearch", Sequence[Term], torch.Tensor], Optional[Term]], 
                breed_fn: Callable[["GPEvSearch", Sequence[Term], torch.Tensor], Sequence[Term]]):
        ''' Gen evol loop.
            Context parameters: max_gen=inf, max_evals=inf
            Context returns: gen, evals made during process, solution (or None), time metrics + other set params by _fn
        '''
        solution = None
        init_fn, eval_fn, analyze_fn, breed_fn = [self.timed(fn, fn_name) 
                                                for fn, fn_name in [(init_fn, "init_fn"), 
                                                                    (eval_fn, "eval_fn"), 
                                                                    (analyze_fn, "analyze_fn"), 
                                                                    (breed_fn, "breed_fn")]]
        try:
            self.forest = init_fn(self, self.pop_size)
            while self.gen < self.max_gen and self.evals < self.max_evals and self.root_evals < self.max_root_evals:
                fitness = eval_fn(self, population)
                solution = analyze_fn(self, population, fitness)
                if solution:
                    break        
                population = breed_fn(self, population, fitness)  
                self.gen += 1
        except EvSearchTermination as e:
            pass
        return solution

#init op
def init_each(context: GPEvSearch, size: int, init_fn = ramped_half_and_half):
    res = []
    init_fn = partial(init_fn, rnd=context.rnd) # rnd is mandatory in init_fn - we can check signature but anyway
    for _ in range(size):
        term = init_fn(context.term_builder, context.leaf_ops, context.branch_ops)
        if term is not None:
            res.append(term)
    return res

# def compute_fitnesses(fitness_fns, interactions, outputs, population, gold_outputs, derived_objectives = [], derived_info = {}, fitness_prep = np_fitness_prep):
#     fitness_list = []
#     for fitness_fn in fitness_fns:
#         fitness = fitness_fn(interactions, outputs, population = population, gold_outputs = gold_outputs, 
#                                 derived_objectives = derived_objectives, **derived_info)
#         fitness_list.append(fitness) 
#     fitnesses = fitness_prep(fitness_list)
#     return fitnesses   

def eval_terms(context: GPEvSearch, terms: list[Term], int_fn: Callable = test_based_interactions):
    bindings = context.new_bindings()
    context.last_outputs = {}
    for term in terms:
        output = evaluate(term, context.ops, bindings)
        context.last_outputs[term] = output
    if int_fn:
        context.last_interactions = {}
        missed_terms = []
        for t in term:
            if t in context.int_cache:
                context.last_interactions[t] = context.int_cache[t]
            else:
                missed_terms.append(t)
        if len(missed_terms) > 0:            
            new_interactions = int_fn(context, missed_terms)
            for term, ints in zip(missed_terms, new_interactions):
                context.int_cache[term] = ints
                context.last_interactions[term] = ints
    context.last_fitness = {}
    missed_fitness = []
    if context.with_fitness_cache:
        # missed_fitness_terms, missed_outputs = zip(*((t, o) for t, o in zip(terms, outputs) if term not in context.fitness_cache))
        for t in terms:
            if t in context.fitness_cache:
                context.last_fitness[t] = context.fitness_cache[t]
            else:
                missed_fitness.append(t)
    else:
        missed_fitness = terms
    if len(missed_fitness) > 0:
        for fn in context.fitness_fns:
            fn(context, missed_fitness)        
    return context.last_fitness

def analyze_population(context: GPEvContext, population, fitness):
    ''' Get the best program in the population '''
    stats = runtime_context.stats
    fitness_order = np.lexsort(fitnesses.T[::-1])
    best_index = fitness_order[0]
    best_fitness = fitnesses[best_index]
    best = population[best_index]
    stats['best'] = str(best)
    is_best = False 
    if (runtime_context.main_fitness_fn is None) and (len(runtime_context.fitness_fns) > 0):
        main_fitness_fn = runtime_context.fitness_fns[0]
    else:
        main_fitness_fn = runtime_context.main_fitness_fn
    for fitness_idx, fitness_fn in enumerate(runtime_context.fitness_fns):
        if fitness_fn == main_fitness_fn:
            is_best = best_cond(best_fitness[fitness_idx])
        stats.setdefault(fitness_fn.__name__, []).append(best_fitness[fitness_idx])
    if save_stats:
        collect_additional_stats(stats, population, outputs)
        total_best_ch = 0
        total_good_ch = 0 
        total_best_dom_ch = 0
        total_good_dom_ch = 0
        total_bad_ch = 0
        for parents, children in runtime_context.parent_child_relations:
            parent_ints = np.array([ runtime_context.int_cache[n] for n in parents ])
            child_ints = np.array([ runtime_context.int_cache[n] for n in children ])
            best_ch, good_ch, best_dom_ch, good_dom_ch, bad_ch = count_good_bad_children(parent_ints, child_ints)
            total_best_ch += best_ch
            total_good_ch += good_ch
            total_best_dom_ch += best_dom_ch
            total_good_dom_ch += good_dom_ch
            total_bad_ch += bad_ch
        # if total_good_ch > 0 or total_bad_ch > 0:
        stats.setdefault('best_children', []).append(total_best_ch)
        stats.setdefault('good_children', []).append(total_good_ch)
        stats.setdefault('best_dom_children', []).append(total_best_dom_ch)
        stats.setdefault('good_dom_children', []).append(total_good_dom_ch)
        stats.setdefault('bad_children', []).append(total_bad_ch)
    if is_best:
        return population[best_index]
    return None