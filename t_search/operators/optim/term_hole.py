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

# @dataclass(order=True)
# class PriorityPair:
#     priority: float
#     term: Term = field(compare=False)
#     hole: tuple[Term, TermPos] = field(compare=False)

@dataclass(order=True)
class HoleFilling:
    priority: float
    l2: float
    id: int
    term: Term
    found_term: Term 
    hole_root: Term 
    hole_pos: TermPos
    term_semantics: torch.Tensor
    hole_semantics: torch.Tensor
    # skeletons: list[Term]

def dn(v: torch.Tensor):
    c = v - v.mean()
    n = c / c.norm() 
    return n

def ds(v:torch.Tensor):
    c = v - v.mean()
    s = c / v.std()
    return s

def dl2(a, b):
    return ((a - b) ** 2).sum()

def covar(a, b):
    a_mean = a.mean()
    b_mean = b.mean()
    return ((a - a_mean) * (b - b_mean)).mean()

def dstat(*s:HoleFilling):
    return [ 
        {
            "dnl2": dl2(dn(hf.term_semantics), dn(hf.hole_semantics)),
            "dsl2": dl2(ds(hf.term_semantics), ds(hf.hole_semantics)),
            "covalN": covar(dn(hf.term_semantics), dn(hf.hole_semantics)),
            "covalS": covar(ds(hf.term_semantics), ds(hf.hole_semantics)),
            "l2": hf.l2, 
            "loss": hf.priority
        }
        for hf in s 
    ]

class TermHolePairs(EvalListener, ServiceBase):
    ''' For new terms search for sketches, for new sketches search for terms.  '''

    def __init__(self, *, 
                    target: torch.Tensor,
                    syntax: Syntax,
                    # evaluator: Evaluator,
                    # fitness: Fitness,
                    term_index: TermVectorStorage,
                    hole_index: HoleVectorStorage,
                    const_optimizer: ConstOptimizer,
                    start_delta: float = 1e-5,
                    multiplier: float = 10,
                    num_steps: int = 3,
                    num_closest: int = 3,
                    small_variation: float = 1e-5,
                    small_value: float = 1e-5,
                    good_filling_loss: float = 0.01,
                    max_filling_queue_size: int = 1000,
                    debug: bool = False
                    ):

        self.target = target
        self.zero = torch.zeros((1,), dtype = target.dtype, device = target.device)
        self.one = torch.ones((1,), dtype = target.dtype, device = target.device)    

        self.term_index: TermVectorStorage = term_index
        
        self.hole_index: HoleVectorStorage = hole_index

        # assert hole_index.normalizer is term_index.normalizer, "Both indexes must share normalizer"
        self.start_delta = start_delta
        self.multiplier = multiplier
        self.num_steps = num_steps
        self.num_closest = num_closest
        self.syntax = syntax
        self.small_variation = small_variation
        self.small_value = small_value

        self.hole_fillings: list[HoleFilling] = []

        self.good_filling_loss = good_filling_loss
        self.max_filling_queue_size = max_filling_queue_size 
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
        # filtered_terms = [t for t in terms if self.syntax.get_num_consts(t) == 0]
        self.register_terms(terms, semantics)
        pass
       
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
                self.add_fill_hole(term, hole[0], hole[1], l2)        
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
                self.add_fill_hole(term, hole[0], hole[1], l2)
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
                                
    def add_fill_hole(self, term: Term, hole_root: Term, hole_pos: TermPos, l2: float) -> Optional[HoleFilling]:

        fit_subterm0, k, b = self.simplify_linear_comb(term, 1.0, 0.0)
        k_value = self.syntax.get_const(value=k)
        b_value = self.syntax.get_const(value=b)
        fit_subterm1 = self.syntax.get_op("mul", k_value, fit_subterm0)
        fit_term = self.syntax.get_op("add", fit_subterm1, b_value)
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

        optimized = self.const_optimizer.optimize(new_term, with_loss=True)
        assert optimized.term is not None, "Const optimizer must return valid term"
        
        hole_filling = HoleFilling(optimized.loss, l2, id(optimized.term), optimized.term,
                            term, hole_root, hole_pos,
                            term_semantics=self.term_index.get_semantics_for_term(term, denormalize=True),
                            hole_semantics=self.hole_index.get_semantics_for_term((hole_root, hole_pos), denormalize=True)
                            ) # no skeletons here as we optimized constants
        
        if self.debug and len(self.hole_fillings) > 0 and \
            ((self.hole_fillings[0].priority < hole_filling.priority and self.hole_fillings[0].l2 > hole_filling.l2) or \
                (self.hole_fillings[0].priority > hole_filling.priority and self.hole_fillings[0].l2 < hole_filling.l2)):
            print(f"loss/l2: new {hole_filling.priority:.4f}/{hole_filling.l2:.4f} vs best {self.hole_fillings[0].priority:.4f}/{self.hole_fillings[0].l2:.4f}")
            print(f">> {hole_filling.term}")
            pass
        existing = next((hf for hf in self.hole_fillings if hf.term == hole_filling.term), None)
        if existing is not None:
            print("Duplicate hole filling detected!")
            pass
        heappush(self.hole_fillings, hole_filling)        

    def has_fillings(self) -> bool:
        return len(self.hole_fillings) > 0   
    
    def get_best_hole_filling(self, force_pick: bool = False) -> HoleFilling | None:
        while (len(self.hole_fillings) > 0) and \
              (force_pick or \
               (self.hole_fillings[0].priority < self.good_filling_loss) or \
               (len(self.hole_fillings) > self.max_filling_queue_size)):

            filling = heappop(self.hole_fillings)
            return filling
        return None