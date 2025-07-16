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
from spatial import InteractionIndex, RTreeIndex, SpatialIndex
from term import Term, TermSignature, build_term, evaluate, ramped_half_and_half, term_sign


class EvSearchTermination(Exception):
    ''' Reaching maximum of evals, gens, ops etc '''    
    pass 

class GPEvSearch():

    def __init__(self, target: torch.Tensor,
                       leaves: dict[str | TermSignature, torch.Tensor],
                       branch_ops: dict[str | TermSignature, Callable],
                       fitness_fns: list[Callable],
                       sem_store_class: SpatialIndex = InteractionIndex, # RTreeIndex,
                       max_gen: int = 100,
                       max_root_evals: int = 100_000, 
                       max_evals: int = 500_000,
                       pop_size: int = 1000,
                       with_caches: bool = False, # enables all caches: syntax, semantic, int, fitness
                       rtol = 1e-04, atol = 1e-03,
                       rnd_seed: Optional[int | np.random.RandomState] = None):
        assert len(leaves) > 0, "At least one leaf should be provided"
        self.target = target
        self.term_to_sid: dict[Term, int] = {} # term to semantic id
        self.sid_to_terms: dict[int, list[Term]] = {} # semantic id to terms

        self.syntax: dict[tuple[str, ...], Term] = {}
        self.sem_store: SpatialIndex = sem_store_class(capacity = max_evals, dims = target.shape[0], dtype=target.dtype, 
                                         device = target.device, rtol=rtol, atol=atol, target = target)
        # self.fitness: torch.Tensor = torch.zeros((0, len(fitness_fns)), dtype=target.dtype, device=target.device)

        self.delayed_semantics: dict[Term, torch.Tensor] = {} # not yet in the store
        self.leaves: list[Term] = []
        self.ops = branch_ops
        self.fitness_fns = fitness_fns
        self.best: list[Term] = [] # best individuals found so far in the forest        
        self.gen: int = 0
        self.evals: int = 0
        self.root_evals: int = 0
        self.max_gen: int = max_gen
        self.max_evals: int = max_evals
        self.mex_root_evals: int = max_root_evals
        self.pop_size: int = pop_size
        self.metrics: dict[str, int | float | list[int|float]] = field(default_factory=dict)
        self.with_caches = True
        self.rtol = rtol
        self.atol = atol
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
        self.process_delayed_semantics()
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
    
    def get_binding(self, term_with_pos: tuple[Term, int]) -> Optional[torch.Tensor]:
        term = term_with_pos[0] # in this implementation we ignore position 
        if term in self.delayed_semantics:
            return self.delayed_semantics[term]
        semantic_id = self.term_to_sid.get(term, None)
        if semantic_id is None:
            return None
        res = self.sem_store.get_vectors(semantic_id)
        return res 

    def set_binding(self, term_with_pos: tuple[Term, int], value: torch.Tensor):
        self.evals += 1
        if not self.with_caches:
            return
        term = term_with_pos[0] # ignoring position currently
        self.delayed_semantics[term] = value

    def process_delayed_semantics(self):
        if len(self.delayed_semantics) == 0:
            return set()
        terms = list(self.delayed_semantics.keys())
        semantics = torch.stack(list(self.delayed_semantics.values()))
        sem_ids = self.sem_store.insert(semantics)
        new_sem_ids = set()
        for sem_id in sem_ids:
            if sem_id not in self.sid_to_terms:
                new_sem_ids.add(sem_id)
        for term, sem_id in zip(terms, sem_ids):
            self.term_to_sid[term] = sem_id
            self.sid_to_terms.setdefault(sem_id, []).append(term)

        self.delayed_semantics = {}
        return new_sem_ids

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
            solution = self.best
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

def eval_terms(context: GPEvSearch, terms: list[Term]) -> torch.Tensor:
    semantics = []
    sids = set()
    for term in terms:
        term_sem = evaluate(term, context.ops, context.get_binding, context.set_binding)
        sids_set = context.process_delayed_semantics() # semantics of tree term
        sids.update(sids_set)
        semantics.append(term_sem)
    sids = list(sids)
    if len(sids) > 0:
        semantics = context.semantics[sids]
    semantics = torch.stack(semantics, dim=0)
    fitnessT = torch.zeros(len(context.fitness_fns), semantics.size(0), dtype=torch.float16, device=semantics.device)
    for fn_id, fn in enumerate(context.fitness_fns):
        fitnessT[fn_id] = fn(context, semantics)

    fitness = fitnessT.T
    return fitness

def analyze_terms(context: GPEvSearch, terms: list[Term], fitness: torch.Tensor, main_fitness_id = 0) -> Optional[Term]:
    ''' Get the best program in the population '''
    fitness_order = np.lexsort(fitness.T[::-1])
    best_index = fitness_order[0]
    best_fitness = fitness[best_index]
    best = term[best_index]
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

def tournament_selection(population: list[Any], fitnesses: np.ndarray, interactions: np.ndarray, fitness_comp_fn = pick_min, tournament_selection_size = 7, *, runtime_context: RuntimeContext):
    ''' Select parents using tournament selection '''
    selected_ids = default_rnd.choice(len(population), tournament_selection_size, replace=True)
    selected_fitnesses = fitnesses[selected_ids]
    best_id_id = fitness_comp_fn(selected_fitnesses)
    best_id = selected_ids[best_id_id]
    # best = population[best_id]
    return best_id

def lexicase_selection(population: list[Any], fitnesses: np.ndarray, interactions: np.ndarray, *, runtime_context: RuntimeContext):
    """ Based on Lee Spector's team: Solving Uncompromising Problems With Lexicase Selection """
    test_ids = np.arange(interactions.shape[1]) # direction is hardcoded 0 - bad, 1 - good
    default_rnd.shuffle(test_ids)
    candidate_ids = np.arange(len(population))
    for test_id in test_ids:
        test_max_outcome = np.max(interactions[candidate_ids, test_id])
        candidate_id_ids = np.where(interactions[candidate_ids, test_id] == test_max_outcome)[0]
        candidate_ids = candidate_ids[candidate_id_ids]
        if len(candidate_ids) == 1:
            break
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    best_id = default_rnd.choice(candidate_ids)
    return best_id

# NOTE: first we do select and then gen muation tree
# TODO: later add to grow and full type constraints on return type
# IDEA: dropout in GP, frozen tree positions which cannot be mutated or crossovered - for later
def subtree_mutation(node, select_node_leaf_prob = 0.1, tree_max_depth = 17, repl_fn = replace_positions, *, runtime_context: RuntimeContext):
    position, position_id, position_depth = _select_node_id(node, lambda d, n: True, select_node_leaf_prob = select_node_leaf_prob)
    if position is None:
        return node    
    position_func_counts = get_func_counts(position, runtime_context.counts_constraints, runtime_context.counts_cache)
    grow_depth = min(5, tree_max_depth - position_depth)
    if runtime_context.counts_constraints is None:
        grow_counts_constraints = None
    else:
        grow_counts_constraints = {}
        for k, v in runtime_context.counts_constraints.items():
            grow_counts_constraints[k] = v - position_func_counts.get(k, 0)
    new_node = grow(grow_depth = grow_depth, func_list = runtime_context.func_list, terminal_list = runtime_context.terminal_list, 
                    counts_constraints = grow_counts_constraints, grow_leaf_prob = None, node_builder = runtime_context.node_builder)
    # new_node_depth = new_node.get_depth()
    # at_depth, at_node = select_node(leaf_prob, node, lambda d, n: (d > 0) and n.is_of_type(new_node), 
    #                                     lambda d, n: (d + new_node_depth) <= max_depth)
    if new_node is None:
        return node
    res = repl_fn(node, {position_id: new_node}, node_builder = runtime_context.node_builder)
    return res

def no_mutation(node):
    return node
        
def subtree_crossover(parent1: Node, parent2: Node, select_node_leaf_prob = 0.1, tree_max_depth = 17, 
                      repl_fn = replace_positions, *, runtime_context: RuntimeContext):
    ''' Crossover two trees '''
    # NOTE: we can crossover root nodes
    # if parent1.get_depth() == 0 or parent2.get_depth() == 0:
    #     return parent1, parent2
    parent1, parent2 = sorted([parent1, parent2], key = lambda x: x.get_depth())
    # for _ in range(3):
    # at1_at_depth, at1 = select_node(leaf_prob, parent1, lambda d, n: (d > 0), lambda d, n: True)
    at1, at1_id, at1_at_depth = _select_node_id(parent1, lambda d, n: True, select_node_leaf_prob=select_node_leaf_prob)
    if at1_id is None:
        return parent1, parent2
    # at1_at_depth, at1 = parent1.get_node(at1_id)
    at1_depth = at1.get_depth()
    at2, at2_id, at2_at_depth = _select_node_id(parent2, 
                        lambda d, n: n.is_of_type(at1) and at1.is_of_type(n) and ((n.get_depth() + at1_at_depth) <= tree_max_depth) and (at1_at_depth > 0 or d > 0) and ((d + at1_depth) <= tree_max_depth) \
                                            and are_counts_constraints_satisfied_together(n, at1, runtime_context.counts_constraints, runtime_context.counts_cache), 
                        select_node_leaf_prob=select_node_leaf_prob)
    # at2_depth, at2
    # at2_depth, at2 = select_node(leaf_prob, parent2, 
    #                     lambda d, n: (d > 0) and n.is_of_type(at1) and at1.is_of_type(n), 
    #                     lambda d, n: ((d + at1_depth) <= max_depth) and ((n.get_depth() + at1_at_depth) <= max_depth))
    if at2_id is None:
        # NOTE: should not be here
        # continue # try another pos
        return parent1, parent2 
        # return parent1, parent2
    child1 = repl_fn(parent1, {at1_id: at2}, node_builder = runtime_context.node_builder)
    child2 = repl_fn(parent2, {at2_id: at1}, node_builder = runtime_context.node_builder)
    return child1, child2       

def subtree_breed(size, population, fitnesses, interactions,
                    breed_select_fn = tournament_selection, mutation_fn = subtree_mutation, crossover_fn = subtree_crossover,
                    mutation_rate = 0.1, crossover_rate = 0.9, *, runtime_context: RuntimeContext):
    new_population = []
    runtime_context.parent_child_relations = []
    if runtime_context.select_fitness_ids is not None and fitnesses is not None:
        fitnesses = fitnesses[:, runtime_context.select_fitness_ids]
    collect_parents = ("syntax" in runtime_context.breeding_stats) or ("semantics" in runtime_context.breeding_stats)
    all_parents = []
    while len(new_population) < size:
        # Select parents for the next generation
        parent1_id = breed_select_fn(population, fitnesses, interactions, runtime_context = runtime_context)
        parent2_id = breed_select_fn(population, fitnesses, interactions, runtime_context = runtime_context)
        parent1 = population[parent1_id]
        parent2 = population[parent2_id]
        if default_rnd.rand() < mutation_rate:
            child1 = mutation_fn(parent1, runtime_context = runtime_context)
        else:
            child1 = parent1
        if default_rnd.rand() < mutation_rate:
            child2 = mutation_fn(parent2, runtime_context = runtime_context)
        else:
            child2 = parent2
        if default_rnd.rand() < crossover_rate:
            child1, child2 = crossover_fn(child1, child2, runtime_context = runtime_context)   
        runtime_context.parent_child_relations.append(([parent1, parent2], [child1, child2]))
        if collect_parents:
            all_parents.extend((parent1, parent2))
        new_population.extend([child1, child2])
    for parent in all_parents:
        if 'syntax' in runtime_context.breeding_stats:
            runtime_context.breeding_stats['syntax'][parent] = runtime_context.breeding_stats['syntax'].get(parent, 0) + 1 
        if 'semantics' in runtime_context.breeding_stats:
            parent_ints = tuple(runtime_context.int_cache[parent])
            runtime_context.breeding_stats['semantics'][parent_ints] = runtime_context.breeding_stats['semantics'].get(parent_ints, 0) + 1 
    return new_population