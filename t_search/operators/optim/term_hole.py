from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import prod
from typing import Literal, Optional

import torch

from t_search.base import ServiceBase
from t_search.evaluators.const_optimizer import ConstOptimizer
from t_search.evaluators.optimization import OptimPoint
from t_search.evaluators.term_spatial import HoleVectorStorage, TermVectorStorage
from t_search.operators.listeners import EvalListener
from t_search.operators.mutation import TermMutation
from t_search.syntax import Term, TermPos
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import Op, Value

@dataclass(order=True)
class PriorityPair:
    priority: float
    term: Term = field(compare=False)
    hole: tuple[Term, TermPos] = field(compare=False)

@dataclass(frozen=False)
class HoleFilling:
    term: Term
    skeletons: list[Term]

class TermHolePairs(EvalListener, ServiceBase):
    ''' For new terms search for sketches, for new sketches search for terms.  '''

    def __init__(self, *, 
                    target: torch.Tensor,
                    syntax: Syntax,
                    # evaluator: Evaluator,
                    # fitness: Fitness,
                    term_index: TermVectorStorage,
                    hole_index: HoleVectorStorage,
                    const_optimizer: ConstOptimizer | None = None,
                    syn_simplifier: TermMutation | None = None,
                    start_delta: float = 1e-5,
                    multiplier: float = 10,
                    num_steps: int = 3,
                    num_closest: int = 3,
                    small_variation: float = 1e-5,
                    small_value: float = 1e-5,
                    min_l2_for_instant_build: float = 0.01,
                    max_pair_queue_size: int = 1000,
                    const_fit_type: Literal["none", "linear", "optimize"] = "optimize",
                    debug: bool = False
                    ):

        self.target = target
        self.zero = torch.zeros((1,), dtype = target.dtype, device = target.device)
        self.one = torch.ones((1,), dtype = target.dtype, device = target.device)    

        self.term_index: TermVectorStorage = term_index
        self.syn_simplifier = syn_simplifier
        
        self.hole_index: HoleVectorStorage = hole_index

        # assert hole_index.normalizer is term_index.normalizer, "Both indexes must share normalizer"
        self.start_delta = start_delta
        self.multiplier = multiplier
        self.num_steps = num_steps
        self.num_closest = num_closest
        self.syntax = syntax
        self.small_variation = small_variation
        self.small_value = small_value

        self.term_hole_pairs: list[PriorityPair] = [] # priority queue

        self.min_l2_for_instant_build = min_l2_for_instant_build
        self.max_pair_queue_size = max_pair_queue_size        
        self.const_fit_type = const_fit_type
        self.debug = debug
        self.const_optimizer = const_optimizer

        pass 

    def init(self):
        # NOTE: we have to add to term index at least one constant to discover constant terms for holes
        if self.syntax.max_consts > 0:
            zero_term = self.syntax.get_const(value=0.0)
            # WARNINING: next fake eval works outsied evaluator object - hack that would avoid circular dependencies 
            # Instead, Value(0) term may have different semantics for different evaluators 
            fake_eval = torch.zeros_like(self.target) # NOTE: for now we assume it is ok 
            self.register_terms([zero_term], fake_eval.unsqueeze(0))
            del fake_eval
            pass 
        pass        

    def on_eval(self, terms: list[Term], semantics: torch.Tensor):
        ''' New terms appear, queue themfor later optimization '''
        self.register_terms(terms, semantics)
       
    def register_terms(self, terms: list[Term], term_params: torch.Tensor) -> None:
        if len(terms) == 0:
            return []

        new_terms = self.term_index.insert(terms, term_params)
        if len(new_terms) == 0:
            return

        normalized_terms, normalized_semantics = self.term_index.get_new_normalized(new_terms)

        if len(normalized_terms) == 0:
            # print("WARN: no normalized terms found during registering terms")
            return
        # searching for nearby holes 
        found_holes = self.hole_index.query_closest(normalized_semantics, 
                                     start_delta=self.start_delta, 
                                     multiplier=self.multiplier, 
                                     num_steps=self.num_steps,
                                     num_closest=self.num_closest)
        del normalized_semantics

        for term, holes in zip(normalized_terms, found_holes):
            for (l2, hole) in holes:
                # if self.debug:
                #     print(f"Pair {term} --> {hole[0]} at {hole[1].term}, {hole[1].occur} with L2 {l2:.4f}")
                #     print(f"     from querying term [register_terms]")
                heappush(self.term_hole_pairs, PriorityPair(l2, term, hole))
        
        pass

    def register_holes(self, holes: list[tuple[Term, TermPos]], hole_params: torch.Tensor) -> None:
        ''' Adds hole and its semantics to index and outputs currently present fillings '''
        if len(holes) == 0:
            return []
        
        new_holes = self.hole_index.insert(holes, hole_params)
        
        if len(new_holes) == 0:
            return

        normalized_holes, normalized_semantics = self.hole_index.get_new_normalized(new_holes)

        if len(normalized_holes) == 0:
            # print("WARN: no normalized holes found during registering holes")
            return
        
        found_terms = self.term_index.query_closest(normalized_semantics, 
                                     start_delta=self.start_delta, 
                                     multiplier=self.multiplier, 
                                     num_steps=self.num_steps,
                                     num_closest=self.num_closest)
        del normalized_semantics

        for hole, terms in zip(normalized_holes, found_terms):
            for (l2, term) in terms:
                # if self.debug:
                #     print(f"Pair {term} --> {hole[0]} at {hole[1].term}, {hole[1].occur} with L2 {l2:.4f}")
                #     print(f"     from querying holes [register_holes]")                
                heappush(self.term_hole_pairs, PriorityPair(l2, term, hole))

        pass 

    # TODO 1: simplify linear combination - DONE 
    # TODO 1.1: Add const optimization to the pipeline
    # TODO 2: operator that gradually introduces new skeletons from Up2D way. (exploraton)
    # TODO 3: do we need const replacement operator?
    # TODO 4: think of final pipeline, what should be the population? 
    # TODO 5: separate random seed/gen for dataset - ensure they are always sampled in the same way for different pipelines
    # TODO 6: early exit in optimization when bellow threshold?? - or not 

    def simplify_linear_comb(self, term: Term, k: float, b: float) -> tuple[Term, float, float]:
        ''' 
            Input term: k1 * s + b1
            Output term: (k * k1) * s + (k * b1 + b) where k * k1 and k * b1 + b are new constants
        '''
        if isinstance(term, Op) and (term.op_id == "add"):
            args = term.get_args()
            add_consts, add_other = [], []
            for a in args:
                (add_consts if isinstance(a, Value) else add_other).append(a)
            if len(add_consts) > 0 and len(add_other) == 1:
                new_b = sum(a.value for a in add_consts) * k + b
                other = add_other[0]
                if isinstance(other, Op) and (other.op_id == "mul"):
                    mul_args = other.get_args()
                    # assert len(mul_args) == 2
                    mul_consts, mul_other = [], []
                    for ma in mul_args:
                        (mul_consts if isinstance(ma, Value) else mul_other).append(ma)
                    if len(mul_consts) > 0 and len(mul_other) == 1:
                        new_k = prod(a.value for a in mul_consts) * k
                        return (mul_other[0], new_k, new_b)
                else:
                    return (other, k, new_b)
        elif isinstance(term, Op) and (term.op_id == "mul"):
            mul_args = term.get_args()
            # assert len(mul_args) == 2
            mul_consts, mul_other = [], []
            for ma in mul_args:
                (mul_consts if isinstance(ma, Value) else mul_other).append(ma)
            if len(mul_consts) > 0 and len(mul_other) == 1:
                new_k = prod(a.value for a in mul_consts) * k
                return (mul_other[0], new_k, b)        
        return (term, k, b)
                                
    def fill_hole(self, term: Term, hole_root: Term, hole_pos: TermPos) -> Optional[HoleFilling]:
        # NOTE: we commented the followign as rescaling would create different term
        # if hole_pos.term == term:
        #     return None 

        term_semantics = self.term_index.get_semantics_for_term(term)
        hole_semantics = self.hole_index.get_semantics_for_term((hole_root, hole_pos))

        # this is part of the path to block in the optimization
        new_skeletons = [
            # OptimPoint(0) # regarding term position itself
        ] # to avoid optimization of some created points again

        # we compute k * ts + b that is closest to hs
        # (k * ts + b - hs)^2 --> min
        # Sx = sum(ts), Sy = sum(hs), Sxx = sum(ts^2), Sxy = sum(ts * hs)

        # (k * x + b - y)^2 --> min
        # 2 (k * x + b - y) * x = 0   

        if self.const_fit_type == "none":
            hole_term = term 
        elif self.const_fit_type == "linear":
            Sx = term_semantics.sum()
            Sy = hole_semantics.sum()
            Sxx = (term_semantics * term_semantics).sum()
            Sxy = (term_semantics * hole_semantics).sum()
            n = term_semantics.shape[0]
            n_Covar_xy = (n * Sxy - Sx * Sy)
            n_Var_x = (n * Sxx - Sx * Sx)
            if n_Var_x / n < self.small_variation: # no variation of x - x == c - searching for best b:
                b = Sy / n # approximate with constant 
                hole_term = self.syntax.get_const(value=b)
            else:
                k = n_Covar_xy / n_Var_x
                b = (Sy - k * Sx) / n

                simple_term, k, b = self.simplify_linear_comb(term, k, b)
                
                if torch.abs(k) < self.small_value:
                    # approximate with constant 
                    hole_term = self.syntax.get_const(value=b)
                elif torch.abs(b) < self.small_value:
                    # approximate with scaling only
                    if torch.abs(k - 1.0) < self.small_value:
                        hole_term = simple_term
                    else:
                        value_k = self.syntax.get_const(value=k)
                        hole_term = self.syntax.get_op("mul", value_k, simple_term)
                        new_skeletons.append(self.syntax.get_op("mul", value_k, OptimPoint(0)))
                elif torch.abs(k - 1.0) < self.small_value: # b is not small here
                    # approximate with shifting only
                    value_b = self.syntax.get_const(value=b)
                    hole_term = self.syntax.get_op("add", simple_term, value_b)
                    new_skeletons.append(self.syntax.get_op("add", OptimPoint(0), value_b))
                else: # general case 
                    value_k = self.syntax.get_const(value=k)
                    value_b = self.syntax.get_const(value=b)       
                    hole_term = self.syntax.get_op("add", 
                                                    self.syntax.get_op("mul", value_k, simple_term),
                                                    value_b)
                    new_skeletons.append(self.syntax.get_op("add", 
                                                            self.syntax.get_op("mul", value_k, OptimPoint(0)),
                                                            value_b))
                    new_skeletons.append(self.syntax.get_op("add", OptimPoint(0), value_b)) 
        elif self.const_fit_type == "optimize" and self.const_optimizer is not None:
            # 1. create dummy constants 1 * t + 0 
            # 2. apply term simplification (1 * (a * s + b) + 0) --> a * s + b
            # 3. apply const_optimzer to fit all constants 
            # 4. simplify (~1) * s --> s and s + (~0) --> s

            fit_subterm0, k, b = self.simplify_linear_comb(term, 1.0, 0.0)
            k_value = self.syntax.get_const(value=k)
            b_value = self.syntax.get_const(value=b)
            fit_subterm1 = self.syntax.get_op("mul", k_value, fit_subterm0)
            fit_term = self.syntax.get_op("add", fit_subterm1, b_value)
            # fit_subterm0_occur = -1
            # fit_subterm1_occur = -1
            # def replace_with_fit_term(t: Term, occur: int) -> Optional[Term]:
            #     nonlocal fit_subterm0_occur, fit_subterm1_occur
            #     if t == fit_subterm0:
            #         fit_subterm0_occur = occur 
            #     elif t == fit_subterm1:
            #         fit_subterm1_occur = occur
            #     elif t == hole_pos.term and occur == hole_pos.occur:
            #         return fit_term           
            # new_term = self.syntax.replace_fn(hole_root, replace_with_fit_term)
            # NOTE: change hole pos to remove (mul ? c) and (add ? c).
            while True:
                if hole_pos.parent is not None and \
                    isinstance(hole_pos.parent.term, Op) and \
                    ((hole_pos.parent.term.op_id == "mul") or (hole_pos.parent.term.op_id == "add")) and \
                    any(isinstance(a, Value) for a in hole_pos.parent.term.get_args()):
                    hole_pos = hole_pos.parent
                    continue                            
                break  

            new_term = self.syntax.replace_position(hole_root, hole_pos, fit_term)
            if new_term is None:
                return None
            # fit_subterm0_occur += 1 
            # fit_subterm1_occur += 1
            optimized_term = self.const_optimizer.optimize(new_term)
            assert optimized_term is not None, "Const optimizer must return valid term"
            
            # val_id = 0
            # def fix_skeleton_fn(t: Term, *_) -> Optional[Term]:
            #     nonlocal val_id
            #     if isinstance(t, Value):
            #         const_val = consts[val_id]
            #         val_id += 1
            #         return const_val
            #     return t
            # fixed_skeleton = self.syntax.replace_fn(new_skeleton_term, fix_skeleton_fn)
            def replace_identities_fn(t: Term, *_) -> Optional[Term]:
                if isinstance(t, Op):
                    if t.op_id == "mul":
                        args = t.get_args()
                        mul_consts, mul_other = [], []
                        for ma in args:
                            (mul_consts if isinstance(ma, Value) else mul_other).append(ma)
                        if len(mul_consts) > 0 and len(mul_other) == 1:
                            new_k = prod(a.value for a in mul_consts)
                            if torch.abs(new_k - 1.0) < self.small_value:
                                return self.syntax.replace_fn(mul_other[0], replace_identities_fn)
                    elif t.op_id == "add":
                        args = t.get_args()
                        add_consts, add_other = [], []
                        for a in args:
                            (add_consts if isinstance(a, Value) else add_other).append(a)
                        if len(add_consts) > 0 and len(add_other) == 1:
                            new_b = sum(a.value for a in add_consts)
                            if torch.abs(new_b) < self.small_value:
                                return self.syntax.replace_fn(add_other[0], replace_identities_fn)
                pass 
            final_term = self.syntax.replace_fn(optimized_term, replace_identities_fn)
            return HoleFilling(final_term, []) # no skeletons here as we optimized constants
        else: 
            raise ValueError(f"Unknown const_fit_type: {self.const_fit_type}") 
        
        if hole_term is None:
            return None
        
        new_term = self.syntax.replace_position(hole_root, hole_pos, hole_term)
        if new_term is None:
            return None
        
        skeletons = [self.syntax.replace_position(hole_root, hole_pos, s, with_validation=False) for s in new_skeletons]        

        if self.syn_simplifier is None:
            return HoleFilling(new_term, skeletons)
        
        new_simplified = self.syn_simplifier.mutate_term(new_term) or new_term
        
        return HoleFilling(new_simplified, skeletons)
    

    def get_best_term_hole_pairs(self, max_pairs: int) -> list[tuple[Term, tuple[Term, TermPos]]]:
        res: list[tuple[Term, tuple[Term, TermPos]]] = []
        while len(res) < max_pairs and len(self.term_hole_pairs) > 0:
            pp = heappop(self.term_hole_pairs)
            res.append((pp.term, pp.hole))
        return res   

    def has_pairs(self) -> bool:
        return len(self.term_hole_pairs) > 0   
    
    def get_best_hole_filling(self, force_pick: bool = False) -> tuple[HoleFilling, PriorityPair] | None:
        while (len(self.term_hole_pairs) > 0) and \
              (force_pick or \
               (self.term_hole_pairs[0].priority < self.min_l2_for_instant_build) or \
               (len(self.term_hole_pairs) > self.max_pair_queue_size)):

            pp = heappop(self.term_hole_pairs)
            filled = self.fill_hole(pp.term, pp.hole[0], pp.hole[1])
            if filled is not None:

                # asserts that filled holes produce better outcomes than original 
                # self.evaluator.eval(filled)
                # new_fitness = self.fitness.get_fitness(filled)
                # old_fitness = self.fitness.get_fitness(pp.hole[0])
                # assert new_fitness < old_fitness, "Filling must improve fitness"

                return (filled, pp)
        return None, None