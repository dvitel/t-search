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
from term import Term, TermSignature, build_term, evaluate, ramped_half_and_half, term_sign


class EvSearchTermination(Exception):
    ''' Reaching maximum of evals, gens, ops etc '''    
    pass 

class GPEvSearch():

    def __init__(self, target: torch.Tensor,
                       leaves: dict[str | TermSignature, torch.Tensor],
                       branch_ops: dict[str | TermSignature, Callable],
                       fitness_fns: list[Callable],
                       max_gen: int = 100,
                       max_root_evals: int = 100000, # 1000 inds per 100 gen
                       max_evals: int = 10000000, # 1000 inds per 100 gen per 100 ops in ind
                       pop_size: int = 1000,
                       with_caches: bool = False, # enables all caches: syntax, semantic, int, fitness
                       rtol = 1e-04, atol = 1e-03,
                       rnd_seed: Optional[int | np.random.RandomState] = None):
        # self.target = target
        # self.term_to_tid: dict[Term, int] = {} 
        # self.tid_to_term: dict[int, Term] = {}
        assert len(leaves) > 0, "At least one leaf should be provided"
        self.term_to_sid: dict[Term, int] = {} # term to semantic id
        self.sid_to_terms: list[list[Term]] = [] # semantic id to terms
        self.sid_to_iid: list[int] = [] # semantic id to interaction id
        self.iid_to_sids: list[list[int]] = [] # interaction id to semantic ids
        # self.sid_to_fid: list[int] = [] # semantic id to fitness id
        # self.fid_to_sids: list[list[int]] = [] # fitness id to semantic ids
        self.syntax: dict[tuple[str, ...], Term] = {}

        # IMPORTANT: semantics is used as cache of values, does not store computational graph
        #            when optimization is used, separate tensors and tensors on the path to the root should be new
        #            cache could be used only for values that are not on computational path root-variation point
        self.semantics = torch.zeros(max_evals, target.shape[0], dtype=torch.float16, device=target.device)
        self.semantics[0] = target
        # delayed semantics collect the group to be inserted at some point (after one tree for instance)
        self.delayed_semantics: dict[Term, torch.Tensor] = {}
        # self.sem_id = 1 # next id to allocate for semantics 
        self.epsilons = None #torch.zeros_like(target) # for interactions 0 1 split
        self.interactions = torch.zeros(max_evals, target.shape[0], dtype=torch.uint8, device=target.device)
        self.interactions[0] = 1
        self.sid_to_iid = [0]
        self.iid_to_sids = [[0]]
        # # self.int_id = 0 # next id to allocate for interactions
        # self.fitness = torch.zeros(max_evals, len(fitness_fns), dtype=torch.float16, device=target.device)
        # self.sid_to_fid = [0]
        # self.fid_to_sids = [[0]]
        # # self.fit_id = 0
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

        for signature, value in leaves.items():
            term = self.term_builder(signature, [])
            self.leaves.append(term)
            self.set_binding((term, 0), value)
            # self.semantics[cur_term_id] = value
        self.process_delayed_semantics()
        self.with_caches = with_caches
        # self.set_proximity(target, self.leaves) # set proximity for leaves
        # self.set_fitness(self.leaves)

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

    # def set_proximity(self, target: torch.Tensor, term_ids: list[int]) -> torch.Tensor:
    #     ''' Important: semantics should be set for term_ids before this call '''
    #     outputs = self.semantics[term_ids] # get outputs for the terms
    #     proximity = torch.isclose(outputs, target.unsqueeze(0), rtol=self.rtol, atol=self.atol).to(dtype=torch.uint8) #∣inputi​−otheri​∣≤rtol×∣otheri​∣+atol
    #     self.interactions[term_ids] = proximity # store interactions in the cache
    #     pass
    
    # def set_fitness(self, term_ids: list[int]):
    #     semantics = self.semantics[term_ids]
    #     interactions = self.interactions[term_ids]
    #     for fitness_id, fitness_fn in enumerate(self.fitness_fns):
    #         fitness_fn(self.fitness[term_ids, fitness_id], semantics, interactions)
    #     pass

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
        # if not self.with_caches:
        #     return None 
        term = term_with_pos[0] # in this implementation we ignore position 
        if term in self.delayed_semantics:
            return self.delayed_semantics[term]
        semantic_id = self.term_to_sid.get(term, None)
        if semantic_id is None:
            return None
        res = self.semantics[semantic_id]
        return res 

    def set_binding(self, term_with_pos: tuple[Term, int], value: torch.Tensor):
        self.evals += 1
        if not self.with_caches:
            return
        term = term_with_pos[0] # ignoring position currently
        # semantic_id = self.term_to_sid.get(term, None)
        # assert semantic_id is None, "Term already has a binding. Term reevaluaton?"

        # search for same semantics through iteractions index
        self.delayed_semantics[term] = value

    def _rebuild_interactions(self):
        # cur_sem_size = len(self.sid_to_iid)
        # if possible_delayed is None:
        #     semantics = self.semantics[:cur_sem_size]
        # else:
        #     semantics = torch.stack([ self.semantics[:cur_sem_size], possible_delayed ])
        semantics = self.semantics[:len(self.sid_to_iid)]
        distances = torch.abs(semantics - self.semantics[0])
        self.epsilons = torch.mean(distances, dim=0)
        ints = (distances < self.epsilons.unsqueeze(0)).to(dtype=torch.uint8)
        unique_interactions, unique_indices = torch.unique(ints, sorted=True, dim=0, return_inverse = True) # O(n log(n)) - we allow for index rebuild
        iid_to_sids = [[] for _ in range(unique_interactions.size(0))]
        sid_to_iid = []
        for sem_id, int_id in enumerate(reversed(unique_indices)):
            iid_id_host = int_id.item()
            iid_to_sids[iid_id_host].append(sem_id)        
            sid_to_iid.append(iid_id_host)

        self.interactions[:unique_interactions.size(0)] = unique_interactions[::-1]
        self.sid_to_iid = sid_to_iid
        self.iid_to_sids = iid_to_sids

        del distances, unique_interactions, unique_indices

    def _get_unique_interactions(self, semantics: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        distances = torch.abs(semantics - self.semantics[0])
        ints = (distances < self.epsilons.unsqueeze(0)).to(dtype=torch.uint8)
        unique_interactions, unique_indices = torch.unique(ints, sorted=True, dim=0, return_inverse = True) # O(n log(n)) - we allow for index rebuild
        iid_to_sids = [[] for _ in range(unique_interactions.size(0))]
        sid_to_iid = []
        for sem_id, int_id in enumerate(reversed(unique_indices)):
            iid_id_host = int_id.item()
            iid_to_sids[iid_id_host].append(sem_id)        
            sid_to_iid.append(iid_id_host)
        return unique_interactions, sid_to_iid, iid_to_sids


    def _get_unique_semantics(self, semantics: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        unique_semantics, unique_indices = torch.unique(semantics, sorted=True, dim=0, return_inverse = True) # O(n log(n)) - also expensive for lareg semantics tensor
        sid_to_pids = [[] for _ in range(unique_semantics.size(0))]
        pid_to_sid = []
        for prog_id, sem_id in enumerate(unique_indices):
            sid_id_host = sem_id.item()
            sid_to_pids[sid_id_host].append(prog_id)        
            pid_to_sid.append(sid_id_host)    
        return unique_semantics, pid_to_sid, sid_to_pids
    
    def get_close_indices(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        ''' Check if two semantics are equal
            x of size (n1, m), y - (n2, m). Result z is (n1, n2) with 1 where they are close
        '''
        z = torch.isclose(x.unsqueeze(1), y.unsqueeze(0), rtol=self.rtol, atol=self.atol).all(dim=2).to(dtype=torch.uint8)

        indices = torch.nonzero(z, as_tuple=False)

        return [(idx[0].item(), idx[1].item()) for idx in indices]
    
    def get_same_indices(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        ''' Check if two semantics are equal
            x of size (n1, m), y - (n2, m). Result z is (n1, n2) with 1 where they are equal
        '''
        z = (x.unsqueeze(1) == y.unsqueeze(0)).all(dim=2).to(dtype=torch.uint8)

        indices = torch.nonzero(z, as_tuple=False)

        return [(idx[0].item(), idx[1].item()) for idx in indices]

    def process_delayed_semantics(self):
        if len(self.delayed_semantics) == 0:
            return set()
        terms = list(self.delayed_semantics.keys())
        semantics = torch.stack(list(self.delayed_semantics.values()))
        unique_semantics, pid_to_sid, sid_to_pids = self._get_unique_semantics(semantics)
        if self.epsilons is None: #start, we have only target semantics 
            close_indices = self.get_close_indices(unique_semantics, self.semantics[:1])
            if len(close_indices) > 0: # we have target semantics in the current semantics
                loc_sids = [x for x, _ in close_indices]
                loc_pids = [pid for sid in loc_sids for pid in sid_to_pids[sid]]
                self.best = [terms[pid] for pid in loc_pids]
                raise EvSearchTermination("Best individuals found at start: {}".format(self.best))
            else:
                cur_sid_id = len(self.sid_to_iid)
                self.semantics[cur_sid_id:cur_sid_id + unique_semantics.size(0)] = unique_semantics
                self.term_to_sid.update({term: cur_sid_id + sid for term, sid in zip(terms, pid_to_sid)})
                self.sid_to_terms.extend([[terms[pid] for pid in pids] for pids in sid_to_pids])
                self._rebuild_interactions()
        else: # starting search for interactions bin
            unique_interactions, sid_to_iid, iid_to_sids = self._get_unique_interactions(unique_semantics)
            same_indices = self.get_same_indices(unique_interactions, self.interactions[:len(self.iid_to_sids)])
            glob_iid_to_loc_sids = {}
            loc_iid_to_glob_iid_map = dict(same_indices)
            for (local_iid, glob_iid) in same_indices:
                glob_iid_to_loc_sids.setdefault(glob_iid, []).extend(iid_to_sids[local_iid])
            loc_sid_to_glob_sid_map = {}
            for glob_iid, loc_sids in glob_iid_to_loc_sids.items():
                glob_sids = self.iid_to_sids[glob_iid]
                local_semantics = unique_semantics[loc_sids]
                glob_semantics = self.semantics[glob_sids]
                close_indices = self.get_close_indices(local_semantics, glob_semantics)
                # here we establish semantic equivalence between terms 
                loc_sid_to_glob_sid_map.update({loc_sids[loc_sid_i]:glob_sids[glob_sid_i] for loc_sid_i, glob_sid_i in close_indices})
            # new_sid_to_iid = []
            start_sid = len(self.sid_to_iid)
            start_iid = len(self.iid_to_sids)
            new_loc_sids = []
            new_loc_iids = []
            for loc_sid, loc_iid in enumerate(sid_to_iid):
                if loc_sid in loc_sid_to_glob_sid_map: # sid match noop - iid will also match 
                    continue
                new_loc_sids.append(loc_sid)
                cur_sid = len(self.sid_to_iid)
                loc_sid_to_glob_sid_map[loc_sid] = cur_sid
                if loc_iid in loc_iid_to_glob_iid_map: # iid matched - add glob_sid 
                    cur_iid = loc_iid_to_glob_iid_map[loc_iid]
                    self.iid_to_sids[cur_iid].append(cur_sid)
                else:
                    cur_iid = len(self.iid_to_sids)
                    self.iid_to_sids.append([cur_sid]) # new iid
                    new_loc_iids.append(loc_iid)
                self.sid_to_iid.append(cur_iid)
            self.semantics[start_sid:len(self.sid_to_iid)] = unique_semantics[new_loc_sids]
            self.interactions[start_iid:len(self.iid_to_sids)] = unique_interactions[new_loc_iids]
            term_to_sid = {term: loc_sid_to_glob_sid_map[loc_sid] for term, loc_sid in zip(terms, pid_to_sid)}
            self.term_to_sid.update(term_to_sid)
            if len(self.sid_to_iid) > len(self.sid_to_terms):
                self.sid_to_terms += [[]] * (len(self.sid_to_iid) - len(self.sid_to_terms))
            for loc_sid, pids in enumerate(sid_to_pids):
                sid_terms = [terms[pid] for pid in pids]
                glob_sid = loc_sid_to_glob_sid_map[loc_sid]
                self.sid_to_terms[glob_sid].extend(sid_terms)

        self.delayed_semantics = {}
        return set([self.term_to_sid[term] for term in terms])

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

# def compute_fitnesses(fitness_fns, interactions, outputs, population, gold_outputs, derived_objectives = [], derived_info = {}, fitness_prep = np_fitness_prep):
#     fitness_list = []
#     for fitness_fn in fitness_fns:
#         fitness = fitness_fn(interactions, outputs, population = population, gold_outputs = gold_outputs, 
#                                 derived_objectives = derived_objectives, **derived_info)
#         fitness_list.append(fitness) 
#     fitnesses = fitness_prep(fitness_list)
#     return fitnesses   

def eval_terms(context: GPEvSearch, terms: list[Term]) -> torch.Tensor:
    semantics = []
    sids = set()
    for term in terms:
        term_sem = evaluate(term, context.ops, context.get_binding, context.set_binding)
        sids_set = context.process_delayed_semantics() # semantics of tree term
        sids.update(sids_set)
        semantics.append(term_sem)
    sids = list(sids)
    if context.with_caches:
        semantics = context.semantics[sids]
    else:
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