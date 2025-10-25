''' Module for different mutation and crossover operators '''

from dataclasses import dataclass, field
from itertools import cycle, product
from typing import Generator, Literal, Optional, Sequence

import numpy as np
import torch

from initialization import UpToDepth
from spatial import VectorStorage
from term import Op, Term, TermPos, Value, get_depth, get_inner_terms, get_pos_constraints, get_pos_sibling_counts, get_positions, grow, is_valid, replace_pos, replace_pos_protected, shuffle_positions
from term_spatial import TermVectorStorage
from torch_alg import OptimPoint, OptimState, OptimState, get_pos_optim_state, optimize_consts, optimize_positions

from typing import TYPE_CHECKING

from torch_alg_inv import DesiredSemantics, get_desired_semantics, invert, alg_inv
from util import OperatorInitMixin, TermsListener, Operator, l2_distance, stack_rows, stack_rows_2d

if TYPE_CHECKING:
    from gp import GPSolver, Builders  # Import only for type checking

class PopulationMutation(Operator):
    ''' Abstract base for all population mutation operators '''
    pass 

class TermMutation(PopulationMutation): 
    ''' Abstract base. Mutates population one term at a time (1-to-1 mapping pattern to repr or mutated child)'''
    def __init__(self, name, *, rate : float = 0.1, **kwargs):
        super().__init__(name, **kwargs)
        self.rate = rate
        self.cur_parents = None

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        ''' Abstract. Mutates one term in the context of parents and already generated children ''' 
        pass # to be implemented in subclasses

    def exec(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]: 
        ''' 
            Some mutations could return None, we would like to reattempt if small number was mutated t guarantee mutated_size.
            However, we still stick to only one pass through population.
        '''

        self.cur_parents = population

        success = 0
        fail = 0     
        repr_cnt = 0    

        size = len(population)
        mutated_size = int(self.rate * size)
        permuted_term_ids = solver.rnd.permutation(size)         
        children = [] 

        for term_id in permuted_term_ids:
            term = population[term_id]
            if mutated_size <= 0: 
                children.append(term)
                repr_cnt += 1
            else: 
                child = self.mutate_term(solver, term)
                if child is not None:
                    success += 1
                    children.append(child)
                    mutated_size -= 1
                else:
                    fail += 1
                    children.append(term)

        self.metrics["success"] = self.metrics.get("success", 0) + success
        self.metrics["fail"] = self.metrics.get("fail", 0) + fail
        self.metrics["repr"] = self.metrics.get("repr", 0) + repr_cnt
        
        return children

class TermCrossover(TermMutation): 
    ''' Abstract base. Two parents crossover. Asymmetric implementation, child is produced from first parent '''

    def crossover_terms(self, solver: 'GPSolver', term: Term, other_term: Term) -> Term | None:
        ''' Abstract. Uses term as based and material from other_term to form a child ''' 
        pass # to be implemented in subclasses

    def select_mate(self, solver: 'GPSolver', term: Term) -> Term: 
        ''' Picks mate for given term. Default: random '''
        term = solver.rnd.choice(self.cur_parents)
        return term

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:        
        other_term = self.select_mate(solver, term)
        child = self.crossover_terms(solver, term, other_term)
        return child
    
def shuffled_position_flow(positions: list[TermPos], leaf_proba: float | None = None, rnd: np.random.RandomState = np.random) -> Generator[TermPos]:
    if len(positions) == 0:
        return
    ordered_pos_ids = shuffle_positions(positions, 
                                    select_node_leaf_prob = leaf_proba, 
                                    rnd = rnd)        
    for pos_id in ordered_pos_ids:
        yield positions[pos_id]

def random_position_flow(positions: list[TermPos], rnd: np.random.RandomState = np.random) -> Generator[TermPos]:
    if len(positions) == 0:
        return
    
    while True:
        pos_id = rnd.randint(len(positions))
        yield positions[pos_id]


class PositionMutation(TermMutation):
    ''' Abstract base. Mutates specific position inside a term. '''

    def __init__(self, name, *, max_pos_tries: int = 1e6, leaf_proba: Optional[float] = 0.1, **kwargs):
        super().__init__(name, **kwargs)
        self.max_pos_tries = max_pos_tries
        self.leaf_proba = leaf_proba

    def select_positions(self, solver: 'GPSolver', term: Term) -> Generator[TermPos]:   
        positions = get_positions(term, solver.pos_cache)
        return shuffled_position_flow(positions, self.leaf_proba, solver.rnd)

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        ''' Abstract. Mutates term at the given position. '''
        pass # to be implemented in subclasses    

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        ''' Mutates one term in the context of parents and already generated children ''' 
        
        positions = self.select_positions(solver, term)
        
        pos_try = 0
        for position in positions:
            if pos_try >= self.max_pos_tries:
                break
            pos_try += 1
            mutated_term = self.mutate_position(solver, term, position)
            if mutated_term is not None:       
                return mutated_term
            
        return None 
    
class PositionCrossover(TermCrossover):
    ''' Abstract base. Crossovers selected positions of two terms '''

    def __init__(self, name, *, max_pos_tries: int = 1e6, leaf_proba: Optional[float] = 0.1, 
                                exclude_values: bool = True, **kwargs):
        super().__init__(name, **kwargs)
        self.max_pos_tries = max_pos_tries    
        self.leaf_proba = leaf_proba
        self.exclude_values = exclude_values

    # def select_term_positions(self, solver: 'GPSolver', term: Term) -> Generator[TermPos]:
    #     ''' Selects positions of base term '''
    #     positions = get_positions(term, solver.pos_cache)
    #     return (p for p in positions)    

    def crossover_positions(self, solver: 'GPSolver', term: Term, position: TermPos, other_term: Term, other_position: TermPos) -> Term | None:
        ''' Abstract. Exchanges terms at positions. '''
        pass # to be implemented in subclasses        

    def default_position_flow(self, solver: 'GPSolver', term: Term) -> Generator[TermPos]:
        positions = get_positions(term, solver.pos_cache)
        if self.exclude_values:
            positions = [pos for pos in positions if not isinstance(pos.term, Value)]
        flow = shuffled_position_flow(positions, self.leaf_proba, solver.rnd)
        return flow
    
    def select_position_pairs(self, solver: 'GPSolver', term: Term, other_term: Term) -> Generator[tuple[TermPos, TermPos]]:
        for pos1 in self.default_position_flow(solver, term):
            for pos2 in self.default_position_flow(solver, other_term):
                if pos1.term == pos2.term:
                    continue
                yield pos1, pos2

    def crossover_terms(self, solver: 'GPSolver', term: Term, other_term: Term) -> Term | None:

        positions = self.select_position_pairs(solver, term, other_term)
        
        pos_try = 0
        for position, other_position in positions:
            if pos_try >= self.max_pos_tries:
                break
            pos_try += 1
            mutated_term = self.crossover_positions(solver, term, position, other_term, other_position)
            if mutated_term is not None:       
                return mutated_term
            
        return None         

    
class RPM(PositionMutation):
    ''' One Random Position Mutation '''

    def __init__(self, name = "RPM", *, max_grow_depth = 5, **kwargs):
        super().__init__(name, **kwargs)
        self.max_grow_depth = max_grow_depth

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        pos_contexts = solver.pos_context_cache.setdefault(term, {})
        start_context = get_pos_constraints(position, solver.builders, solver.counts_cache, pos_contexts)
        arg_counts = get_pos_sibling_counts(position, solver.builders)

        new_term = grow(grow_depth = min(self.max_grow_depth, 
                                            solver.max_term_depth - position.at_depth), 
                        builders = solver.builders, start_context = start_context, 
                        arg_counts = arg_counts,
                        gen_metrics = self.metrics, rnd = solver.rnd)                
        
        mutated_term = replace_pos(position, new_term, solver.builders)
        return mutated_term

class RPX(PositionCrossover):
    ''' One Random Position Crossover '''

    def __init__(self, name: str = "_1p_rand_x", **kwargs):
        super().__init__(name, **kwargs)
        self.crossover_cache: dict[tuple[Term, Term, int, Term], Term] = {}    

    def crossover_positions(self, solver: 'GPSolver', term: Term, position: TermPos, 
                                                        other_term: Term, other_position: TermPos) -> Term | None:

        crossover_key = (term, position.term, position.occur, other_position.term)
        if crossover_key in self.crossover_cache:
            child = self.crossover_cache[crossover_key]
            return child 

        child = replace_pos_protected(position, other_position.term, solver.builders,
                                        depth_cache=solver.depth_cache,
                                        counts_cache=solver.counts_cache,
                                        pos_context_cache=solver.pos_context_cache.setdefault(term, {}),
                                        max_term_depth=solver.max_term_depth)
        
        return child
      
class CO(TermMutation):
    ''' Const Optimization, Adjust consts to correspond to the given target. '''

    def __init__(self, name = "const_opt", *, 
                 frac = 0.2, 
                 num_vals: int = 1,
                 max_tries: int = 1,
                 num_evals: int = 10, lr = 1.0,
                 loss_threshold: Optional[float] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.frac = frac
        self.num_vals = num_vals
        self.max_tries = max_tries
        self.num_evals = num_evals
        self.lr = lr
        self.term_values_cache: dict[Term, list[Value]] = {}
        self.optim_term_cache: dict[Term, Term] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}
        self.loss_threshold = loss_threshold

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        ''' Optimizes all constants inside the term '''
        
        term_loss, *_ = solver.eval(term, return_outputs="list").outputs
        
        optim_res = optimize_consts(term, term_loss, solver.fitness_fn, solver.builders,
                                    solver.ops, solver._get_binding,
                                    solver.const_range, 
                                    eval_fn = solver.eval_fn,
                                    num_vals = self.num_vals,
                                    max_tries=self.max_tries,
                                    max_evals=self.num_evals,
                                    lr = self.lr, loss_threshold = (solver.best_fitness if self.loss_threshold is None else self.loss_threshold),
                                    torch_gen=solver.torch_gen,
                                    term_values_cache=self.term_values_cache,
                                    optim_term_cache=self.optim_term_cache,
                                    optim_state_cache=self.optim_state_cache)

        if optim_res is not None:
            optim_state, num_evals, num_root_evals = optim_res
            solver.report_evals(num_evals, num_root_evals)                        

            # print(f"<<< {optim_res.optim_state.final_term} | {term_loss:.2f} --> {optim_res.optim_state.best_loss.item():.2f} >>>")
            if optim_state.best_loss is not None and (term_loss < optim_state.best_loss[0]):
                return None # can happen when we exhaust all attempts of optimization 
            return optim_state.best_term
        return None             
    
class Dedupl(PopulationMutation):
    ''' Removes duplicate syntaxes from the population '''

    def __init__(self, name: str = "dedupl", **kwargs):
        super().__init__(name, **kwargs)    

    def exec(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]:
        present_terms = set()
        children = []
        for term in population:
            if term not in present_terms:
                children.append(term)
                present_terms.add(term)
        return children
    
class ReplWithBestInner(TermMutation):
    ''' Replaces each term with its inner term with best fitness '''

    def __init__(self, name: str = "best_inner", **kwargs):
        super().__init__(name, **kwargs)
        self.term_best_inner_term_cache: dict[Term, Term] = {}

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        if term in self.term_best_inner_term_cache:
            child = self.term_best_inner_term_cache[term]
            return child 
        inner_terms = get_inner_terms(term)
        # self.term_inner_terms_cache[term] = inner_terms
        inner_fitness = solver.eval(inner_terms, return_fitness="tensor").fitness
        best_id = torch.argmin(inner_fitness).item()
        best_inner = inner_terms[best_id]
        self.term_best_inner_term_cache[term] = best_inner
        del inner_fitness
        return best_inner
    
        # sort_ids = torch.argsort(inner_fitness) 
        # best_ids = sort_ids[:self.inner_cnt]
        # best_inners = [present_terms[i] for i in best_ids.tolist()]
        # if len(present_terms) == len(inner_terms):
        #     self.term_best_inner_term_cache[term] = best_inners
        # del inner_fitness        
        

from syntax import sp_alg_ops_f, sp_alg_ops_b, sp_simplify

class Reduce(TermMutation): 
    ''' Syntactic Simplifier based on domain axioms '''

    def  __init__(self, name: str = "syn_simpl", *,
                    to_ops: dict = sp_alg_ops_f, from_ops: dict = sp_alg_ops_b,
                    check_validity: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self.to_ops = to_ops
        self.from_ops = from_ops
        self.check_validity = check_validity
    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        new_term = sp_simplify(term, to_dict = self.to_ops, from_dict=self.from_ops,
                               alloc_val = lambda value: solver.const_builder.fn(value = value),
                               alloc_var = lambda var_id: solver.var_builder.fn(var_id = var_id),
                               alloc_op = lambda op_id: lambda *args: solver.op_builders.get(op_id).fn(*args))
        if self.check_validity and not is_valid(new_term, builders=solver.builders, counts_cache=solver.counts_cache):
            return None
        return new_term

@dataclass 
class OptimizedPos:
    term_pos: list[TermPos]
    cur_id: int

@dataclass 
class TermSemantics:
    term: Term # term or sketch (w OptimPoint)
    sid: int # id in spatial index 
    # normalization: semantics(Term) = std * index(sid) + mean 
    std: torch.Tensor # scaling coefficient 
    mean: torch.Tensor # shift coefficient

@dataclass 
class HoleSemantics:    
    root_term: Term
    pos: TermPos
    sid: int 
    std: torch.Tensor
    mean: torch.Tensor
            
class PO(PositionMutation, OperatorInitMixin, TermsListener):
    ''' Position Optimization, adjust selected position with optimizer ''' 
    
    def __init__(self, name = "point_opt", *, num_vals: int = 1, max_tries: int = 1,
                 num_evals: int = 10, lr = 1.0, delta: float = 0.1,
                 num_best: int = 5,
                 loss_threshold: Optional[float] = None,
                 sem_atol: float = 1e-5,
                 collect_inner_binding: bool = True,
                 index_type = VectorStorage,
                 normalize_semantics: bool = True,
                 syn_simplify: Optional[Reduce] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.num_vals = num_vals
        self.max_tries = max_tries
        self.num_evals = num_evals
        self.lr = lr
        self.delta = delta
        self.sem_atol = sem_atol
        self.num_best = num_best
        self.tries_pos: dict[Term, OptimizedPos] = {}
        self.optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term | None] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}
        self.optim_point_pos_cache: dict[Term, TermPos] = {}
        self.loss_threshold = loss_threshold
        self.collect_inner_binding = collect_inner_binding

        self.index_type = index_type
        self.term_index: VectorStorage | None = None         
        self.hole_index: VectorStorage | None = None
        self.term_semantics: dict[Term, TermSemantics] = {}
        self.semantic_terms: dict[int, TermSemantics] = {}
        self.semantic_holes: dict[int, dict[tuple[Term, Term, int, int], HoleSemantics]] = {} 
        self.zero: torch.Tensor | None = None
        self.one: torch.Tensor | None = None
        self.normalize_semantics = normalize_semantics
        self.syn_simplify = syn_simplify

    def op_init(self, solver: 'GPSolver'):
        if self.term_index is not None:
            del self.term_index
        self.term_index: VectorStorage = \
            self.index_type(capacity = solver.max_evals // 2, dims = solver.target.shape[0], 
                dtype = solver.dtype, device = solver.device,
                rtol = 0, atol = self.sem_atol)
        if self.hole_index is not None:
            del self.hole_index
        self.hole_index: VectorStorage = \
            self.index_type(capacity = solver.max_evals // 2, dims = solver.target.shape[0], 
                dtype = solver.dtype, device = solver.device,
                rtol = 0, atol = self.sem_atol)
        
        self.term_semantics: dict[Term, TermSemantics] = {}
        self.semantic_terms: dict[int, TermSemantics] = {}
        self.semantic_holes: dict[int, dict[tuple[Term, Term, int, int], HoleSemantics]] = {} 
        self.zero = torch.zeros((1,), dtype = solver.dtype, device = solver.device)
        self.one = torch.ones((1,), dtype = solver.dtype, device = solver.device)

        if self.normalize_semantics:
            if "add" not in solver.ops or "mul" not in solver.ops or solver.max_consts == 0:
                print(f"Warning: normalization was disabled as there are no operations (add, mul) or consts to revert it")
                self.normalize_semantics = False # normalization requires add, mul in solver.ops
        
        # if solver.max_consts > 0 and self.normalize_semantics:
        if self.normalize_semantics:
            zero_ids = self.term_index.insert(torch.zeros_like(solver.target).unsqueeze(0))
            zero_id = zero_ids[0]
            zero_const = Value(self.zero)
            zero_semantics = TermSemantics(term=zero_const, sid=zero_id, std=self.zero, mean=self.zero)
            self.term_semantics[zero_const] = zero_semantics
            self.semantic_terms[zero_id] = zero_semantics

    def select_positions(self, solver: 'GPSolver', term: Term) -> Generator[TermPos]:
        if term not in self.tries_pos:
            positions = get_positions(term, solver.pos_cache)
            positions = [pos for pos in positions if pos not in solver.invalid_term_outputs]
            # DECISION 1: how to pick term pos?? shuffle, sorted by depth?
            positions.sort(key=lambda pos: pos.at_depth) # start with shallowest positions
            # positions = solver.rnd.permutation(positions)
            self.tries_pos[term] = OptimizedPos(term_pos=positions, cur_id=0)

        term_pos = self.tries_pos[term]
        end_id = term_pos.cur_id - 1 
        if end_id < 0:
            end_id = len(term_pos.term_pos) - 1
        cur_pos = None
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
                optim_state = None
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

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        
        optim_term = self.optim_term_cache.get((term, position))
        optim_state = self.optim_state_cache.get(optim_term)
        if optim_state is None:
            return None
        
        pos_output, *_ = solver.eval(position.term, return_outputs="list").outputs
        output_range = stack_rows([pos_output, solver.target], target_size=solver.target.shape[0])
        range_mins = torch.minimum(output_range[0], output_range[1])
        range_maxs = torch.maximum(output_range[0], output_range[1])
        output_range[0] = range_mins - self.delta
        output_range[1] = range_maxs + self.delta

        num_evals, num_root_evals = \
            optimize_positions(optim_state, solver.fitness_fn,
                solver.ops, solver._get_binding,
                output_range, 
                solver.eval_fn,
                pos_outputs=(pos_output,),
                num_vals = self.num_vals,
                max_evals=self.num_evals,
                num_best = self.num_best,
                lr = self.lr, loss_threshold = (solver.best_fitness if self.loss_threshold is None else self.loss_threshold),
                collect_inner_binding = self.collect_inner_binding,
                torch_gen=solver.torch_gen)
        
        solver.report_evals(num_evals, num_root_evals)
        if optim_state.best_loss is None: 
            return None    
        # good semantics to add to the hole index

        holes_w_semantics: list[tuple[Term, TermPos, torch.Tensor]] = []

        if self.collect_inner_binding:

            if optim_state.optim_term not in self.optim_point_pos_cache:
                optim_term_poss = get_positions(optim_state.optim_term, solver.pos_cache)
                optim_point_pos = next(pos for pos in optim_term_poss if isinstance(pos.term, OptimPoint))
                self.optim_point_pos_cache[optim_state.optim_term] = optim_point_pos

            optim_point_pos = self.optim_point_pos_cache[optim_state.optim_term]
            
            # now we have pos (in term) and optim_point_pos (in optim_term)
            # we can build chains in both terms to the root 

            cur_pos = position
            cur_optim_pos = optim_point_pos
            while cur_pos.term != term:
                cur_binding = optim_state.best_binding[cur_optim_pos.term]
                holes_w_semantics.append((term, cur_pos, cur_binding))
                cur_pos = cur_pos.parent
                cur_optim_pos = cur_optim_pos.parent
        else: # we collected only point binding 
            cur_binding = optim_state.best_binding[optim_state.optim_points[0]]
            holes_w_semantics.append((term, position, cur_binding))

        new_terms = self.register_holes(solver, holes_w_semantics)
        
        return new_terms 

    def _query_index(self, idx: VectorStorage, 
                            query: torch.Tensor,
                            qtype: Literal["point", "range"] = "point",
                            deltas = [0.001, 0.01, 0.1],) -> dict[int, list[int]]:
        ''' Either point query or more complelx iterative range query 
            Returns map: query id to found ids in index (list)
        '''
        
        if qtype == "point":
            found_ids = idx.query_points(query, rtol=0, atol=1e-1) #atol or self.atol, rtol=rtol or self.rtol)
            res = {qid:[v] for qid, v in enumerate(found_ids) if v >= 0}
            return res 
        
        # qtype == "range":

        # const_val = self.find_any_const(query)
        # if const_val is not None:
        #     return [Value(const_val)]
    
        res = {}
        for delta in deltas:
            for qid, q in enumerate(query):
                range = torch.stack([q - delta, q + delta], dim=0)
                found_ids = idx.query_range(range)
                if len(found_ids) > 0:
                    res[qid] = found_ids
            if len(res) > 0:
                break
            
        return res        
    
    def fill_hole(self, solver: 'GPSolver', 
                    hole_semantics: HoleSemantics, term_semantics: TermSemantics) -> Optional[Term]:
        if hole_semantics.pos.term == term_semantics.term:
            return None 
        

        if self.normalize_semantics:

            # denormalize --> though we have much by normalized semantics, mean and std is different 
            #  we create a term that would match hole_semantics 
            # t* = hs * (t - tm) / ts + hm = (hs / ts) * t + hm - (hs / ts) * tm
            # new_term = k * term_semantics.term + b 
            #       where k = hole_semantics.std / term_semantics.std 
            #             b = hole_semantics.mean - hole_semantics.std / term_semantics.std * term_semantics.mean

            term_std_zero = torch.isclose(term_semantics.std, self.zero, rtol=0, atol=1e-2)
            hole_std_zero = torch.isclose(hole_semantics.std, self.zero, rtol=0, atol=1e-2)
            if term_std_zero and hole_std_zero: # const adjustment hs / ts = 0 / 0 = 1
                k = self.one #torch.ones_like(hole_semantics.std)
                b = hole_semantics.mean - term_semantics.mean
            elif term_std_zero:
                return None # cannot adjust const to hole 
            else:
                k = hole_semantics.std / term_semantics.std
                b = hole_semantics.mean - (hole_semantics.std / term_semantics.std) * term_semantics.mean
            k_is_one = torch.isclose(k, self.one, rtol=0, atol=1e-2)
            b_is_zero = torch.isclose(b, self.zero, rtol=0, atol=1e-2)
            if k_is_one and b_is_zero:
                hole_term = term_semantics.term
            elif k_is_one:
                # NOTE: only one constant is allowed in term_index - Value(0)
                if isinstance(term_semantics.term, Value):
                    hole_term = Value(b)
                else:
                    hole_term = solver.op_builders["add"].fn(term_semantics.term, solver.const_builder.fn(value = b))
            elif b_is_zero:
                hole_term = solver.op_builders["mul"].fn(solver.const_builder.fn(value = k), term_semantics.term)
            else:
                if isinstance(term_semantics.term, Value):
                    hole_term = Value(b)
                else:
                    hole_term = solver.op_builders["add"].fn(
                                    solver.op_builders["mul"].fn(solver.const_builder.fn(value = k), term_semantics.term),
                                    solver.const_builder.fn(value = b))
        else:
            hole_term = term_semantics.term
        
        new_term = replace_pos_protected(hole_semantics.pos, hole_term, solver.builders,
                                        depth_cache=solver.depth_cache,
                                        counts_cache=solver.counts_cache,
                                        pos_context_cache=solver.pos_context_cache.setdefault(hole_semantics.root_term, {}),
                                        max_term_depth=solver.max_term_depth)
        
        if new_term is not None and self.syn_simplify is not None:
            new_term = self.syn_simplify.mutate_term(solver, new_term, [], [])
        return new_term
        
    def register_terms(self, solver: 'GPSolver', terms: list[Term], semantics: torch.Tensor) -> list[Term]: 
        if len(terms) == 0:
            return []
        if self.normalize_semantics:
            means = torch.mean(semantics, dim=1, keepdim=False)
            stds = torch.std(semantics, dim=1, keepdim=False)
            const_mask = torch.isclose(stds, self.zero, rtol=0, atol=1e-2)
            nonconst_mask = ~const_mask
            nonconst_ids, = torch.where(nonconst_mask)
            if nonconst_ids.numel() == 0:
                return
            final_means = means[nonconst_ids]
            final_stds = stds[nonconst_ids]
            normalized_semantics = (semantics[nonconst_ids] - final_means.unsqueeze(-1)) / final_stds.unsqueeze(-1)
            nonconst_terms: list[Term] = [terms[i] for i in nonconst_ids.tolist()]
        else:
            normalized_semantics = semantics
            nonconst_terms = terms
            final_means = [self.zero] * len(terms)
            final_stds = [self.one] * len(terms)
        semantic_ids = self.term_index.insert(normalized_semantics)
        for term, semantic_id, mean, std in zip(nonconst_terms, semantic_ids, final_means, final_stds):
            term_semantics = TermSemantics(term=term, sid=semantic_id, std=std, mean=mean)
            self.term_semantics[term] = term_semantics
            if semantic_id in self.semantic_terms: # pick smallest term as representative
                cur_t = self.semantic_terms[semantic_id].term
                cur_t_depth = get_depth(cur_t, solver.depth_cache)
                t_depth = get_depth(term, solver.depth_cache)
                if t_depth < cur_t_depth:
                    self.semantic_terms[semantic_id] = term_semantics
            else:
                self.semantic_terms[semantic_id] = term_semantics
        # searching for nearby holes 
        found_hole_ids = self._query_index(self.hole_index, normalized_semantics)
        closest_pairs = [(hole_sem, self.semantic_terms[semantic_ids[qid]]) 
                        for qid, hids in found_hole_ids.items()
                        for hid in hids 
                        for hole_sem in self.semantic_holes.get(hid, {}).values()]
        new_terms = []
        present_terms = set()
        for hole_sem, term_sem in closest_pairs:
            new_term = self.fill_hole(solver, hole_sem, term_sem)
            if new_term is not None and new_term not in present_terms:
                present_terms.add(new_term)
                new_terms.append(new_term)
        return new_terms

    def register_holes(self, solver: 'GPSolver', holes: list[tuple[Term, TermPos, torch.Tensor]]) -> list[Term]:
        ''' Adds hole and its semantics to index and outputs currently present fillings '''
        if len(holes) == 0:
            return []
        semantics = stack_rows_2d([s for _, _, s in holes], target_size=solver.target.shape[0])

        if self.normalize_semantics:
            means = torch.mean(semantics, dim=1, keepdim=True)
            stds = torch.std(semantics, dim=1, keepdim=True)
            const_mask = torch.all(torch.isclose(semantics, means, rtol=1e-2, atol=1e-2), dim=-1)
            # nonconst_mask = ~const_mask
            # nonconst_ids, = torch.where(nonconst_mask)
            normalized_semantics = (semantics - means) / stds
            normalized_semantics[const_mask] = self.zero
            stds[const_mask] = self.zero

            # nonconst_terms: list[Term] = [terms[i] for i in nonconst_ids.tolist()]
            # semantic_ids = self.term_index.insert(normalized_semantics)
        else:
            normalized_semantics = semantics
            means = [self.zero] * semantics.shape[0]
            stds = [self.one] * semantics.shape[0]

        all_hole_ids = self.hole_index.insert(normalized_semantics)

        cur_start = 0
        hole_semantics_map = {}
        for (root_term, hole_pos, hs) in holes:
            hole_query_ids = list(range(cur_start, cur_start + hs.shape[0]))
            cur_start += hs.shape[0]
            for qid in hole_query_ids:
                hole_sem_id = all_hole_ids[qid]
                mean = means[qid]
                std = stds[qid]
                hole_semantics = HoleSemantics(root_term=root_term, pos=hole_pos, sid=hole_sem_id, std=std, mean=mean)
                hole_semantics_map[qid] = hole_semantics
                sem_sketches = self.semantic_holes.setdefault(hole_sem_id, {})
                sem_sketches[(root_term, hole_pos.term, hole_pos.occur, hole_sem_id)] = hole_semantics

        query_ids = self._query_index(self.term_index, normalized_semantics)

        cur_start = 0
        new_terms = []
        present_terms = set()
        for (root_term, hole_pos, hole_semantics) in holes:
            hole_query_ids = list(range(cur_start, cur_start + hole_semantics.shape[0]))
            cur_start += hole_semantics.shape[0]
            present_tuples = set()
            for qid in hole_query_ids:
                hole_semantics = hole_semantics_map[qid]
                for term_sid in query_ids.get(qid, []):
                    term_semantics = self.semantic_terms[term_sid]
                    tuple_key = (hole_semantics.root_term, hole_semantics.pos.term, hole_semantics.pos.occur, term_semantics.term)
                    if tuple_key in present_tuples:
                        continue
                    present_tuples.add(tuple_key)
                    new_term = self.fill_hole(solver, hole_semantics, term_semantics)
                    if new_term is not None and new_term not in present_terms:
                        present_terms.add(new_term)
                        new_terms.append(new_term)
        del normalized_semantics
        return new_terms        
    
class SDM(RPM):
    ''' Semantically Driven Mutation Beadle and Johnson (2008, 2009b) '''
    def __init__(self, name = "SDM", *, min_d = 1e-1, max_d=1e+2, **kwargs):
        super().__init__(name, **kwargs)
        self.min_d = min_d
        self.max_d = max_d

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        mutated_term = super().mutate_position(solver, term, position)
        if mutated_term is None: 
            return None
        
        # check semantic difference
        parent_sem, mutated_term_sem, *_ = solver.eval([term, mutated_term], return_outputs="list").outputs
        dist = l2_distance(parent_sem, mutated_term_sem)
        if dist < self.min_d or dist > self.max_d:
            return None        

        return mutated_term 

class SGM(TermMutation, OperatorInitMixin):
    ''' Implementing Semantic Geometric Mutation from Moraglio 2012 
        Parent program is lineary combined with random term 

        p' = p + r * (t1 - t2)
        r - random const 
        t1, t2 - random terms 
    '''
    def __init__(self, name = "SGM", *, max_grow_depth = 5, num_tries = 2, epsilon = 0.02, 
                    check_validity: bool = False,
                    simplifier: Reduce | None = None,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.num_tries = num_tries
        self.max_grow_depth = max_grow_depth
        self.epsilon = epsilon
        self.minus_one: Term | None = None
        self.check_validity = check_validity
        self.simplifier = simplifier

    def op_init(self, solver):
        self.minus_one = solver.const_builder.fn(value = -1.0)

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:

        mutated_term = None
        
        for _ in range(self.num_tries):
            t1 = grow(grow_depth = self.max_grow_depth,
                        builders = solver.builders, start_context = None, 
                        arg_counts = None,
                        gen_metrics = self.metrics, rnd = solver.rnd) 

            t2 = grow(grow_depth = self.max_grow_depth,
                        builders = solver.builders, start_context = None, 
                        arg_counts = None,
                        gen_metrics = self.metrics, rnd = solver.rnd)               
                
            neg_t2 = solver.op_builders["mul"].fn(self.minus_one, t2)
            t1_minus_t2 = solver.op_builders["add"].fn(t1, neg_t2)
            r = solver.const_builder.fn(value = solver.rnd.rand() * self.epsilon)
            delta_term = solver.op_builders["mul"].fn(r, t1_minus_t2)
            mutated_term = solver.op_builders["add"].fn(term, delta_term)
            if self.simplifier is not None:
                mutated_term = self.simplifier.mutate_term(solver, [mutated_term])
            if self.check_validity and not is_valid(mutated_term, builders=solver.builders, counts_cache=solver.counts_cache):
                mutated_term = None
            if mutated_term is not None:
                break

        return mutated_term 
    
def get_best_semantics(desired: DesiredSemantics, undesired: list[DesiredSemantics], all_semantics: torch.Tensor,):
    assert len(desired) > 0

    if any(d is None for d in desired): # unsat desired at position 
        return None

    # if all(len(d) == 0 for d in desired): # any term will work - shou
    #     return None 

    forbidden_mask = torch.zeros((all_semantics.shape[1],), dtype=torch.bool, device=all_semantics.device)

    for forbidden in undesired:

        if any(d is None for d in forbidden):
            continue # unsat undesired - skip
        
        forbidden_close_mask = torch.ones((all_semantics.shape[1],), dtype=torch.bool, device=all_semantics.device)

        for test_id, forbit_values in enumerate(forbidden):
            if len(forbit_values) == 0:
                continue 
            sem_values = all_semantics[:, test_id].unsqueeze(-1) # (num_terms, 1)
            forbidden_tensor = torch.tensor(list(forbit_values), dtype=all_semantics.dtype, device=all_semantics.device)
            diffs = torch.abs(sem_values - forbidden_tensor.unsqueeze(0)) # (num_terms, num_forbidden)
            close_mask = torch.any(diffs < 1e-5, dim=1) # (num_terms,)
            forbidden_close_mask &= close_mask
            del forbidden_tensor
            if not torch.any(forbidden_close_mask):
                break

        forbidden_mask |= forbidden_close_mask

    # test_ids = [i for i, d in enumerate(desired) if len(d) > 0]
    # selected_semantics = all_semantics[:, test_ids]
    sem_score = torch.zeros((all_semantics.shape[0],), dtype=all_semantics.dtype, device=all_semantics.device)
    for test_id, allowed_values in enumerate(desired):
        if len(allowed_values) == 0:
            continue 
        sem_values = all_semantics[:, test_id].unsqueeze(-1) # (num_terms, 1)
        allowed_tensor = torch.tensor(list(forbit_values), dtype=all_semantics.dtype, device=all_semantics.device)
        diffs = torch.abs(sem_values - allowed_tensor.unsqueeze(0)) # (num_terms, num_allowed)
        sem_score += torch.min(diffs, dim=1).values # (num_terms,) 

    sem_score[forbidden_mask] = torch.inf

    best_sem_id = torch.argmin(sem_score).item()

    if sem_score[best_sem_id] == torch.inf:
        return None

    return best_sem_id
    
# applying classic one point random crossover and mutation until there is semantic difference
class SDX(RPX):
    ''' Semantically Driven Crossover Beadle and Johnson (2008, 2009b) '''
    def __init__(self, name = "SDX", *, min_d = 1e-1, max_d=1e+2, **kwargs):
        super().__init__(name, **kwargs)
        self.min_d = min_d
        self.max_d = max_d

    def crossover_positions(self, solver: 'GPSolver', term: Term, position: TermPos, 
                                                        other_term: Term, other_position: TermPos) -> Term | None:
        mutated_term = super().crossover_positions(solver, term, position, other_term, other_position)
        if mutated_term is None: 
            return None
        
        # check semantic difference
        term1_sem, term2_sem, mutated_term_sem, *_ = solver.eval([term, other_term, mutated_term], return_outputs="list").outputs
        dist1 = l2_distance(term1_sem, mutated_term_sem)
        dist2 = l2_distance(term2_sem, mutated_term_sem)
        if dist1 < self.min_d or dist1 > self.max_d or dist2 < self.min_d or dist2 > self.max_d:
            return None       

        return mutated_term 

class SGX(TermCrossover, OperatorInitMixin):
    ''' Implementing Semantic Geometric Crossover from Moraglio 2012 
        Linear combination of programs
    '''
    def __init__(self, name = "SGX", *, max_grow_depth = 5, num_tries = 2, epsilon = 1.0, 
                    check_validity: bool = False,
                    simplifier: Reduce | None = None,
                    min_d: float | None = 1e-2,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.num_tries = num_tries
        self.max_grow_depth = max_grow_depth
        self.epsilon = epsilon
        self.minus_one: Term | None = None
        self.check_validity = check_validity
        self.simplifier = simplifier

    def op_init(self, solver):
        self.minus_one = solver.const_builder.fn(value = -1.0)        

    def crossover_terms(self, solver: 'GPSolver', term: Term, other_term: Term) -> Term | None:

        mutated_term = None
        
        t1 = term  
        t2 = other_term       

        for _ in range(self.num_tries):


            neg_t2 = solver.op_builders["mul"].fn(self.minus_one, t2)
            t1_minus_t2 = solver.op_builders["add"].fn(t1, neg_t2)
            r = solver.const_builder.fn(value = solver.rnd.rand() * self.epsilon)
            delta_term = solver.op_builders["mul"].fn(r, t1_minus_t2)
            mutated_term = solver.op_builders["add"].fn(term, delta_term)
            if self.simplifier is not None:
                mutated_term = self.simplifier.mutate_term(solver, [mutated_term])
            if self.check_validity and not is_valid(mutated_term, builders=solver.builders, counts_cache=solver.counts_cache):
                mutated_term = None

            if self.min_d is not None: # check effectiveness of the operator
                term1_sem, term2_sem, mutated_term_sem, *_ = solver.eval([term, other_term, mutated_term], return_outputs="list").outputs
                dist1 = l2_distance(term1_sem, mutated_term_sem)
                dist2 = l2_distance(term2_sem, mutated_term_sem)
                if dist1 < self.min_d or dist2 < self.min_d:
                    mutated_term = None
            if mutated_term is not None:
                break

        return mutated_term 

# class LGX(Mutation): 
#     ''' Replace parent subprograms with library of known programs '''
#     pass 

# # class LGM(Mutation):
# #     pass 

# class AGX(Mutation):
#     ''' Inverse execution of parent '''
#     pass 

# class SSGX(Mutation):
#     ''' Subtree Semantic Geometric Crossover from Nguyen et al. (2016)
#         pick most similar to parent subtree and then use SGX 
#     '''
#     pass

@dataclass 
class InversionCache: 
    term_semantics: dict[Term, DesiredSemantics] = field(default_factory=dict)
    term_subtree_semantics: dict[Term, dict[tuple[Term, int], tuple[DesiredSemantics, list[DesiredSemantics]]]] = field(default_factory=dict)

class CM(PositionMutation, OperatorInitMixin, TermsListener):
    ''' Competent Mutation from Dr. Kraviec and Pawlak
        Parent program is lineary combined with random term 
    '''
    def __init__(self, name = "CM", *, 
                    index: TermVectorStorage, 
                    inv_cache: InversionCache,
                    index_init_depth: int | None = None, 
                    dynamic_index: bool = False,
                    index_max_size: int = 1e10,
                    op_invs = alg_inv,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.index = index # used as library of semantics 
        self.inv_cache = inv_cache
        self.index_init_depth = index_init_depth # if None, dynamic library - uses any available term. 
        self.dynamic_index = dynamic_index
        self.index_max_size = index_max_size
        self.desired_target: DesiredSemantics | None = None
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache

    def op_init(self, solver):
        ''' Initializes desired combinatorial semantics and Library of programs '''
        self.desired_target = get_desired_semantics(solver.target)
        if self.index_init_depth is not None and self.index.len_sem() == 0: 
            init_op = UpToDepth(self.index_init_depth, force_pop_size=False)
            lib_terms = init_op(solver, pop_size=self.index_max_size)
            semantics = solver.eval(lib_terms, return_outputs="tensor").outputs
            self.index.insert(lib_terms, semantics) 
            del semantics
        pass 

    def register_terms(self, solver, terms, semantics):
        if self.dynamic_index and self.index.len_terms() < self.index_max_size:
            self.index.insert(terms, semantics)
        return []

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        
        if (position.term, position.occur) not in self.desired_at_pos:
            return None
        
        desired, undesired = self.desired_at_pos[(position.term, position.occur)]

        all_semantics = self.index.get_semantics()

        best_sem_id = get_best_semantics(desired, undesired, all_semantics)

        if best_sem_id is None:
            return None
        
        best_term = self.index.get_repr_term(best_sem_id)
        
        mutated_term = replace_pos_protected(position, best_term, solver.builders, 
                            depth_cache=solver.depth_cache,
                            counts_cache=solver.counts_cache,
                            pos_context_cache=solver.pos_context_cache.setdefault(term, {}),
                            max_term_depth=solver.max_term_depth)
                                             
        return mutated_term

    
    def mutate_term(self, solver: 'GPSolver', term: Term, parents: Sequence[Term], children: Sequence[Term]) -> Term | None:

        term_sem, *_ = solver.eval(term, return_outputs="list").outputs
        if term not in self.inv_cache.term_semantics:
            self.inv_cache.term_semantics[term] = get_desired_semantics(term_sem)

        self.desired_at_pos = invert(term, self.desired_target, [self.inv_cache.term_semantics[term]], 
                                     lambda args: solver.eval(args, return_outputs="list").outputs, 
                                     self.inv_cache.term_semantics, self.op_invs)
        
        child = super().mutate_term(solver, term, parents, children)

        del self.desired_at_pos

        return child 

# applying classic one point random crossover and mutation until there is semantic difference
class CX(TermCrossover, OperatorInitMixin, TermsListener):
    ''' Competent crossover operator '''

    def __init__(self, name = "CX", *, 
                    index: TermVectorStorage, 
                    inv_cache: InversionCache,
                    index_init_depth: int | None = None, 
                    dynamic_index: bool = False,
                    index_max_size: int = 1e10,
                    op_invs = alg_inv,
                    max_tries: int = 2,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.index = index # used as library of semantics 
        self.inv_cache = inv_cache
        self.index_init_depth = index_init_depth # if None, dynamic library - uses any available term. 
        self.dynamic_index = dynamic_index
        self.index_max_size = index_max_size
        self.op_invs = op_invs
        self.desired_at_pos = {} # temp cache
        self.max_tries = max_tries

    def op_init(self, solver):
        ''' Initializes desired combinatorial semantics and Library of programs '''
        if self.index_init_depth is not None and self.index.len_sem() == 0: 
            init_op = UpToDepth(self.index_init_depth, force_pop_size=False)
            lib_terms = init_op(solver, pop_size=self.index_max_size)
            semantics = solver.eval(lib_terms, return_outputs="tensor").outputs
            self.index.insert(lib_terms, semantics) 
            del semantics
        pass 

    def register_terms(self, solver, terms, semantics):
        if self.dynamic_index and self.index.len_terms() < self.index_max_size:
            self.index.insert(terms, semantics)
        return []

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        child = CM.mutate_position(self, solver, term, position)
        return child

    def crossover_terms(self, solver: 'GPSolver', term: Term, other_term: Term) -> Term | None:

        term_sem, other_term_sem, *_ = solver.eval([term, other_term], return_outputs="list").outputs

        if term not in self.term_curr:
            self.inv_cache.term_semantics[term] = get_desired_semantics(term_sem)
        if other_term not in self.term_curr:
            self.inv_cache.term_semantics[other_term] = get_desired_semantics(other_term_sem)        

        mid_point = 0.5 * (term_sem + other_term_sem)
        mid_desired = get_desired_semantics(mid_point)


        self.desired_at_pos = invert(term, mid_desired, [self.inv_cache.term_semantics[term], self.inv_cache.term_semantics[other_term]], 
                                     lambda args: solver.eval(args, return_outputs="list").outputs, 
                                     self.inv_cache.term_semantics, self.op_invs)
        
        child = PositionMutation.mutate_term(self, solver, term)

        del self.desired_at_pos

        return child