''' Module for different mutation and crossover operators '''

from dataclasses import dataclass
from itertools import cycle, product
from typing import Optional, Sequence

import torch

from spatial import VectorStorage
from term import Term, TermPos, Value, get_depth, get_inner_terms, get_pos_constraints, get_pos_sibling_counts, get_positions, grow, is_valid, replace_pos, replace_pos_protected, shuffle_positions
from torch_alg import OptimState, OptimState, get_pos_optim_state, optimize_consts, optimize_positions

from typing import TYPE_CHECKING

from util import Operator, stack_rows

if TYPE_CHECKING:
    from gp import GPSolver  # Import only for type checking

class Mutation(Operator):

    def _mutate(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]:
        ''' Should be implemented in subclasses '''
        return population # noop by default - reproduction
    
    def __call__(self, solver: 'GPSolver', population: Sequence[Term]): 
        ''' Use to trigger mutation, _mutate should not be called directly '''
        self.on_start()
        children =  self._mutate(solver, population)
        return children
    
    # def configure(self, solver: 'GPSolver'):
    #             # self, *, builders: Builders,
    #             #  pos_cache: dict[Term, list[TermPos]],
    #             #  pos_context_cache: dict[Term, dict[tuple[Term, int], TermGenContext]],
    #             #  counts_cache: dict[Term, np.ndarray], 
    #             #  depth_cache: dict[Term, int],
    #             #  device: str, torch_gen: torch.Generator, rnd: np.random.RandomState): 
    #     self.builders = solver.builders
    #     self.pos_cache = solver.pos_cache
    #     self.pos_context_cache = solver.pos_context_cache
    #     self.counts_cache = solver.counts_cache
    #     self.depth_cache = solver.depth_cache
    #     self.device = solver.device
    #     self.torch_gen = solver.torch_gen
    #     self.rnd = solver.rnd
    
class PointRandMutation(Mutation):

    def __init__(self, name = "_1p_rand_m", *, rate : float = 0.1, 
                    leaf_proba: Optional[float] = 0.1, max_grow_depth = 5):
        super().__init__(name)
        self.rate = rate
        self.leaf_proba = leaf_proba
        self.max_grow_depth = max_grow_depth

    def _one_point_rand_mutation(self, solver: 'GPSolver', term: Term, num_children: int) -> list[Term]:
        
        # metrics
        success = 0
        fail = 0
        
        positions = get_positions(term, solver.pos_cache)

        if len(positions) == 0:
            fail += 1
            return []
        
        pos_contexts = solver.pos_context_cache.setdefault(term, {})

        ordered_pos_ids = shuffle_positions(positions, 
                                        select_node_leaf_prob = self.leaf_proba, 
                                        rnd = solver.rnd)

        mutants = []
        prev_same_count = 0
        prev_len = -1
        for pos_id in cycle(ordered_pos_ids):
            if len(mutants) >= num_children:
                break
            if prev_len == len(mutants):
                prev_same_count += 1
                if prev_same_count > len(ordered_pos_ids):
                    break
            else:
                prev_same_count = 0
                prev_len = len(mutants)
            position: TermPos = positions[pos_id]
            start_context = get_pos_constraints(position, solver.builders, solver.counts_cache, pos_contexts)
            arg_counts = get_pos_sibling_counts(position, solver.builders)

            new_term = grow(grow_depth = min(self.max_grow_depth, 
                                             solver.max_term_depth - position.at_depth), 
                            builders = solver.builders, start_context = start_context, 
                            arg_counts = arg_counts,
                            gen_metrics = self.metrics, rnd = solver.rnd)                
            
            mutated_term = replace_pos(position, new_term, solver.builders)
            if mutated_term is not None:       
                # val_poss = get_positions(mutated_term, {})
                # for val_pos in val_poss:
                #     get_pos_constraints(val_pos, builders, {}, {})
                # pass        
                mutants.append(mutated_term)
                success += 1
            else:
                fail += 1

        repr = 0
        if len(mutants) < num_children:
            repr = num_children - len(mutants)
            mutants += [term] * (num_children - len(mutants))

        self.metrics["success"] = self.metrics.get("success", 0) + success
        self.metrics["fail"] = self.metrics.get("fail", 0) + fail
        self.metrics["repr"] = self.metrics.get("repr", 0) + repr
            
        return mutants 

    def _mutate(self, solver: 'GPSolver', population: Sequence[Term]): 

        size = len(population)

        mutation_mask = torch.rand(size, device=solver.device,
                                    generator=solver.torch_gen) < self.rate
        
        mutation_mask_list = mutation_mask.tolist()

        mutation_pos = {}
        for i, parent in enumerate(population):
            if mutation_mask_list[i]:
                mutation_pos.setdefault(parent, []).append(i)

        children = list(population)
        for term, term_p in mutation_pos.items():
            mutated_terms = self._one_point_rand_mutation(solver,
                                term = term, num_children=len(term_p))
            for i, mterm in zip(term_p, mutated_terms):
                children[i] = mterm
        
        return children
    
class PointRandCrossover(Mutation):
    ''' In contrast to ClassicPointRandCrossover, it tries to satisfy requested number of crossovers '''

    def __init__(self, name: str = "_1p_rand_x", *, frac : float = 0.9, 
                    leaf_proba: Optional[float] = 0.1, num_tries: int = 1,
                    exclude_values: bool = True):
        super().__init__(name)
        self.frac = frac
        self.leaf_proba = leaf_proba
        self.exclude_values = exclude_values 
        self.num_tries = num_tries
        self.crossover_cache: dict[tuple[Term, Term, int, Term], Term] = {}    

    def _mutate(self, solver: 'GPSolver', population: Sequence[Term]):

        # metrics
        success = 0 
        fails = 0
        retries = 0

        max_count = int(self.frac * len(population))

        children = list(population)

        term_ids = solver.rnd.permutation(len(population))        
        for term_i in term_ids:
            if max_count <= 0:
                break
            term1 = population[term_i]

            positions1 = get_positions(term1, solver.pos_cache)
            if self.exclude_values:
                positions1 = [pos for pos in positions1 if not isinstance(pos.term, Value)]            

            if len(positions1) == 0:
                retries += 1
                continue

            pos_contexts = solver.pos_context_cache.setdefault(term1, {})

            pos1 = solver.rnd.choice(positions1)

            tries_count = self.num_tries
            child = None
            while tries_count > 0:

                other_id = solver.rnd.choice(term_ids)
                term2 = population[other_id]

                positions2 = get_positions(term2, solver.pos_cache)
                if self.exclude_values:
                    positions2 = [pos for pos in positions2 if not isinstance(pos.term, Value)]

                if len(positions2) == 0:
                    tries_count -= 1
                    retries += 1
                    continue

                pos2 = solver.rnd.choice(positions2)
                # for pos in solver.rnd.permutation(positions2):
                if pos2.term == pos1.term:
                    tries_count -= 1
                    retries += 1
                    continue
                if (term1, pos1.term, pos1.occur, pos2.term) in self.crossover_cache:
                    tries_count -= 1
                    retries += 1
                    continue

                child = replace_pos_protected(pos1, pos2.term, solver.builders,
                                              depth_cache=solver.depth_cache,
                                              counts_cache=solver.counts_cache,
                                              pos_context_cache=pos_contexts,
                                              max_term_depth=solver.max_term_depth)
                
                if child is None:
                    tries_count -= 1
                    retries += 1
                    continue                
                break 

            if child is not None:
                self.crossover_cache[(term1, pos1.term, pos1.occur, pos2.term)] = child
                children[term_i] = child
                success += 1
                max_count -= 1
            else:
                fails += 1

        self.metrics["success"] = self.metrics.get("success", 0) + success
        self.metrics["fails"] = self.metrics.get("fails", 0) + fails
        self.metrics["retries"] = self.metrics.get("retries", 0) + retries

        return children

class ClassicPointRandCrossover(Mutation):

    def __init__(self, name: str = "_1p_rand_x", *, rate : float = 0.9, 
                    leaf_proba: Optional[float] = 0.1,
                    exclude_values: bool = True):
        super().__init__(name)
        self.rate = rate
        self.leaf_proba = leaf_proba
        self.exclude_values = exclude_values 
        self.crossover_cache: dict[tuple[Term, Term, int, Term], Term] = {}

    def _one_point_rand_crossover(self, solver: 'GPSolver', term1: Term, term2: Term, num_children: int) -> list[Term]:    

        # metrics
        same_subtree = 0
        success = 0 
        fail = 0
        cache_hit = 0

        positions1 = get_positions(term1, solver.pos_cache)
        positions2 = get_positions(term2, solver.pos_cache)

        num_pairs = len(positions1) * len(positions2)
        if num_pairs > 0:
            term1_pos_contexts = solver.pos_context_cache.setdefault(term1, {})
            term2_pos_contexts = solver.pos_context_cache.setdefault(term2, {})

            pos_ids1 = shuffle_positions(positions1,
                                        select_node_leaf_prob = self.leaf_proba, 
                                        rnd = solver.rnd)
            
            if self.exclude_values:
                pos_ids1 = [pos_id for pos_id in pos_ids1 if not isinstance(positions1[pos_id].term, Value)]

            pos_ids2 = shuffle_positions(positions2,
                                        select_node_leaf_prob = self.leaf_proba, 
                                        rnd = solver.rnd)
            
            if self.exclude_values:
                pos_ids2 = [pos_id for pos_id in pos_ids2 if not isinstance(positions2[pos_id].term, Value)]

        else:
            pos_ids1 = []
            pos_ids2 = []

        children = []

        num_points = min(len(pos_ids1) * len(pos_ids2), num_children)

        for pos_id1, pos_id2 in product(pos_ids1, pos_ids2):
            pos1: TermPos = positions1[pos_id1]
            pos2: TermPos = positions2[pos_id2]
            if pos1.term == pos2.term:
                same_subtree += 2
                continue

            if (term1, pos1.term, pos1.occur, pos2.term) in self.crossover_cache:
                children.append(self.crossover_cache[(term1, pos1.term, pos1.occur, pos2.term)])
                cache_hit += 1
            else:
                
                new_child = replace_pos_protected(pos1, pos2.term, solver.builders,
                                                  depth_cache=solver.depth_cache,
                                                  counts_cache=solver.counts_cache,
                                                  pos_context_cache=term1_pos_contexts,
                                                  max_term_depth=solver.max_term_depth)                

                if new_child is not None:
                    # val_poss = get_positions(new_child, {})
                    # for val_pos in val_poss:
                    #     get_pos_constraints(val_pos, builders, {}, {})
                    # pass
                    children.append(new_child)
                    self.crossover_cache[(term1, pos1.term, pos1.occur, pos2.term)] = new_child
                    success += 1
                else:
                    fail += 1

            if len(children) >= num_children:
                break

            if (term2, pos2.term, pos2.occur, pos1.term) in self.crossover_cache:
                children.append(self.crossover_cache[(term2, pos2.term, pos2.occur, pos1.term)])
                cache_hit += 1
            else:
                
                new_child = replace_pos_protected(pos2, pos1.term, solver.builders,
                                                  depth_cache=solver.depth_cache,
                                                  counts_cache=solver.counts_cache,
                                                  pos_context_cache=term2_pos_contexts,
                                                  max_term_depth=solver.max_term_depth)
                
                if new_child is not None:
                    # val_poss = get_positions(new_child, {})
                    # for val_pos in val_poss:
                    #     get_pos_constraints(val_pos, builders, {}, {})
                    # pass                
                    children.append(new_child)
                    self.crossover_cache[(term2, pos2.term, pos2.occur, pos1.term)] = new_child
                    success += 1
                else:
                    fail += 1
            
            if len(children) >= num_children:
                break    

        repr = 0
        if len(children) < num_children:
            repr = num_children - len(children)
            left_children = [term1] * (num_children - len(children))
            for i in range(1, len(left_children), 2):
                left_children[i] = term2
            children += left_children

        self.metrics["same_subtree"] = self.metrics.get("same_subtree", 0) + same_subtree
        self.metrics["success"] = self.metrics.get("success", 0) + success
        self.metrics["fail"] = self.metrics.get("fail", 0) + fail
        self.metrics["cache_hit"] = self.metrics.get("cache_hit", 0) + cache_hit
        self.metrics["children"] = self.metrics.get("children", 0) + len(children)
        self.metrics["repr"] = self.metrics.get("repr", 0) + repr
        self.metrics["num_points"] = self.metrics.get("num_points", 0) + num_points

        return children        

    def _mutate(self, solver: 'GPSolver', population: Sequence[Term]):

        size = len(population)

        crossover_mask = torch.rand(size // 2, device=solver.device,
                                        generator=solver.torch_gen) < self.rate
        
        crossover_mask_list = crossover_mask.tolist()
        crossover_pairs = {}
        for i, should_crossover in enumerate(crossover_mask_list):
            if should_crossover:
                parent1 = population[2 * i]
                parent2 = population[2 * i + 1]
                crossover_pairs.setdefault((parent1, parent2), []).append(i)        

        children = list(population)
        for (parent1, parent2), pair_ids in crossover_pairs.items():
            new_children = self._one_point_rand_crossover(solver, term1 = parent1, term2 = parent2, 
                                                         num_children=2 * len(pair_ids))
            for i, ii in enumerate(pair_ids):
                children[2 * ii] = new_children[2 * i]
                children[2 * ii + 1] = new_children[2 * i + 1]

        return children
        
class ConstOptimization(Mutation):
    ''' Adjust consts to correspond to the given target '''

    def __init__(self, name = "const_opt", *, 
                 frac = 0.2, 
                 num_vals: int = 1,
                 max_tries: int = 1,
                 num_evals: int = 10, lr = 1.0,
                 rtol: float = 1e-5, atol: float = 1e-5):
        super().__init__(name)
        self.frac = frac
        self.num_vals = num_vals
        self.max_tries = max_tries
        self.num_evals = num_evals
        self.lr = lr
        self.rtol = rtol
        self.atol = atol
        self.term_values_cache: dict[Term, list[Value]] = {}
        self.optim_term_cache: dict[Term, Term] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}

    def _optimize_consts(self, solver: 'GPSolver', term: Term, term_loss: torch.Tensor) -> Optional[Term]:
        # start_opt = perf_counter()
        optim_res = optimize_consts(term, term_loss, solver.target, solver.builders,
                                    solver.ops, solver._get_binding,
                                    solver.const_range, 
                                    solver.eval_fn, solver.fitness_fn,
                                    num_vals = self.num_vals,
                                    max_tries=self.max_tries,
                                    max_evals=self.num_evals,
                                    lr = self.lr,
                                    rtol = self.rtol, atol = self.atol,
                                    torch_gen=solver.torch_gen,
                                    term_values_cache=self.term_values_cache,
                                    optim_term_cache=self.optim_term_cache,
                                    optim_state_cache=self.optim_state_cache)
        # end_opt = perf_counter()
        # dur = round((end_opt - start_opt) * 1000)
        # if dur > 100:
        #     print(f"O: {dur}, {optim_res.num_root_evals}, {term}\n\t{optim_res.term}")
        if optim_res is not None:
            optim_state, num_evals, num_root_evals = optim_res
            solver.report_evals(num_evals, num_root_evals)                        

            # print(f"<<< {optim_res.optim_state.final_term} | {term_loss:.2f} --> {optim_res.optim_state.best_loss.item():.2f} >>>")
            if term_loss < optim_state.best_loss[0]:
                return None # can happen when we exhaust all attempts of optimization 
            return optim_state.best_term
        return None

    def _mutate(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]:
        children = list(population)
        new_terms = []
        max_count = int(len(population) * self.frac)
        if max_count == len(population):
            term_ids = range(len(population))
        else:
            term_ids = solver.rnd.permutation(len(population))
        for term_id in term_ids:
            if max_count <= 0:
                break
            term = population[term_id]
            if term not in solver.term_outputs:            
                continue # optimizing only evaluated and good terms
            term_loss = solver.output_fitness[solver.term_outputs[term]]
            optimized_term = self._optimize_consts(solver, term, term_loss)
            if optimized_term is not None:
                children[term_id] = optimized_term
                new_terms.append(optimized_term)
                max_count -= 1
        if len(new_terms) > 0:
            solver.eval(new_terms)
        return children    
    
class Deduplicate(Mutation):
    ''' Removes duplicate syntaxes from the population '''

    def __init__(self, name: str = "dedupl"):
        super().__init__(name)    

    def _mutate(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]:
        present_terms = set()
        children = []
        for term in population:
            if term not in present_terms:
                children.append(term)
                present_terms.add(term)
        return children
    
class ReplaceWithBestInner(Mutation):
    ''' Replaces each term with its inner term with best fitness '''

    def __init__(self, name: str = "best_inner", *, frac: float = 0.5,
                    inner_cnt: float = 3, with_self: bool = True):
        super().__init__(name)
        self.term_best_inner_term_cache: dict[Term, Term] = {}
        self.frac = frac
        self.inner_cnt = inner_cnt
        self.with_self = with_self

    def _mutate(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]:
        
        if not solver.cache_evals:
            return population
        
        children = []

        max_count = int(len(population) * self.frac)
        if max_count == len(population):
            term_ids = range(len(population))
        else:
            term_ids = solver.rnd.permutation(len(population))

        alerady_added = set()
        for term_id in term_ids:
            term = population[term_id]
            if max_count <= 0:
                if term not in alerady_added:
                    children.append(term)
                    alerady_added.add(term)
                continue
            if term not in self.term_best_inner_term_cache:
                inner_terms = get_inner_terms(term)
                # self.term_inner_terms_cache[term] = inner_terms
                present_terms, present_fitness = solver.get_terms_fitness(inner_terms)
                if len(present_fitness) > 0:
                    inner_fitness = torch.stack(present_fitness, dim=0)
                    sort_ids = torch.argsort(inner_fitness) 
                    best_ids = sort_ids[:self.inner_cnt]
                    best_inners = [present_terms[i] for i in best_ids.tolist()]
                    if len(present_terms) == len(inner_terms):
                        self.term_best_inner_term_cache[term] = best_inners
                    del inner_fitness
                else:
                    best_inners = [term]
            else:
                best_inners = self.term_best_inner_term_cache[term]
            for t in best_inners:
                if t not in alerady_added:
                    children.append(t)
                    alerady_added.add(t)
            if self.with_self and term not in alerady_added:
                children.append(term)
                alerady_added.add(term)
            max_count -= 1

        return children
        
@dataclass 
class OptimizedPos:
    term_pos: list[TermPos]
    cur_id: int
            
class PointOptimization(Mutation):
    ''' Adjust arbirary point of the term by searching best vectors at points ''' 
    
    def __init__(self, name = "point_opt", *, 
                 frac = 0.2, num_vals: int = 1, max_tries: int = 1,
                 num_evals: int = 10, lr = 1.0, delta: float = 0.1,
                 num_best: int = 5,
                 loss_threshold: Optional[float] = 0.1,
                 rtol: float = 1e-5, atol: float = 1e-5):
        super().__init__(name)
        self.frac = frac
        self.num_vals = num_vals
        self.max_tries = max_tries
        self.num_evals = num_evals
        self.lr = lr
        self.delta = delta
        self.rtol = rtol
        self.atol = atol
        self.num_best = num_best
        self.tries_pos: dict[Term, OptimizedPos] = {}
        self.optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}
        self.loss_threshold = loss_threshold

    def _select_pos_optim_state(self, solver: 'GPSolver', term: Term) -> Optional[tuple[TermPos, OptimState]]:
        if term not in self.tries_pos:
            positions = get_positions(term, solver.pos_cache)
            valid_positions = [pos for pos in positions if pos not in solver.invalid_term_outputs]
            # DECISION 1: how to pick term pos?? shuffle, sorted by depth?
            # positions.sort(key=lambda pos: pos.at_depth) # start with shallowest positions
            positions = solver.rnd.permutation(valid_positions)
            self.tries_pos[term] = OptimizedPos(term_pos=positions, cur_id=0)

        term_pos = self.tries_pos[term]
        end_id = term_pos.cur_id - 1 
        if end_id < 0:
            end_id = len(term_pos.term_pos) - 1
        optim_state = None
        # pos_to_remove = set()
        while term_pos.cur_id <= end_id:
            cur_pos = term_pos.term_pos[term_pos.cur_id % len(term_pos.term_pos)]
            term_pos.cur_id += 1
            optim_state = get_pos_optim_state(term, (cur_pos,), 
                                optim_term_cache = self.optim_term_cache, 
                                optim_state_cache = self.optim_state_cache,
                                builders = solver.builders,
                                num_vals = self.num_vals,
                                output_size = solver.target.shape[0],
                                max_tries = self.max_tries,
                                dtype = solver.dtype, device = solver.device)
            # DECISION 2: how many optim attempts? what starting point to take?
            if optim_state is None:
                # pos_to_remove.add((cur_pos.term, cur_pos.occur))
                continue
            if optim_state.max_tries <= 0:
                # point was already optimized and is in the hole index
                continue # try next pos // or should we try next term??? return None
                # return None ???
            break 

        # if len(pos_to_remove) > 0:
        #     term_pos.term_pos = [pos for pos in term_pos.term_pos if (pos.term, pos.occur) not in pos_to_remove]
        #     if len(term_pos.term_pos) == 0:
        #         del self.tries_pos[term]
        #         return None 
        #     term_pos.cur_id = term_pos.cur_id % len(term_pos.term_pos)
        return cur_pos, optim_state
    
    def _optimize_pos_inner(self, solver: 'GPSolver', term: Term,  pos: TermPos, optim_state: OptimState) -> list[Term]:
        pos_output = solver.get_cached_output(pos.term)
        output_range = torch.stack([pos_output, solver.target], dim=0)
        range_mins = torch.minimum(output_range[0], output_range[1])
        range_maxs = torch.maximum(output_range[0], output_range[1])
        output_range[0] = range_mins - self.delta
        output_range[1] = range_maxs + self.delta
        num_evals, num_root_evals = \
            optimize_positions(optim_state, solver.target,
                solver.ops, solver._get_binding,
                output_range, 
                solver.eval_fn, solver.fitness_fn,
                pos_outputs=(pos_output,),
                num_vals = self.num_vals,
                max_tries=self.max_tries,
                max_evals=self.num_evals,
                num_best = self.num_best,
                lr = self.lr,
                rtol = self.rtol, atol = self.atol,
                torch_gen=solver.torch_gen)
        solver.report_evals(num_evals, num_root_evals)
        fit_ids, = torch.where(optim_state.best_loss < self.loss_threshold)
        if len(fit_ids) > 0: # good semantics to add to the hole index

            holes = []
            semantics = []

            while pos is not None:
                holes.append(pos)
                semantics.append(optim_state.best_binding[pos.term][fit_ids])
                pos = pos.parent

            new_terms = solver.register_holes(holes, semantics)
            
            return new_terms 
        
        return [] # no good terms that match pos semantics

    def _optimize_pos(self, solver: 'GPSolver', term: Term) -> list[Term]:
        
        cur_pos, optim_state = self._select_pos_optim_state(solver, term)        
        
        if optim_state is None:
            return [] 
        
        final_terms = self._optimize_pos_inner(solver, term, cur_pos, optim_state)

        return final_terms 

    def _mutate(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]:
        children = []
        max_count = int(len(population) * self.frac)
        new_terms = []
        if max_count == len(population):
            term_ids = range(len(population))
        else:
            term_ids = solver.rnd.permutation(len(population))
        for term_id in term_ids:
            if max_count <= 0:
                break
            term = population[term_id]
            if term not in solver.term_outputs:            
                continue # optimizing only evaluated and good terms
            optimized_terms = self._optimize_pos(solver, term)
            new_terms.extend(optimized_terms)
            if len(optimized_terms) > 0:
                children.extend(optimized_terms)
                max_count -= len(optimized_terms)
            else:
                children.append(term)
        if len(new_terms) > 0:
            solver.eval(new_terms)
        return children