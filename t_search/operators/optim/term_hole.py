from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from math import prod
from typing import Callable, Literal, Optional

import torch

from t_search.base import ServiceBase
from t_search.evaluators.const_optimizer import ConstOptimizer
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.term_spatial import HoleVectorStorage, TermVectorStorage
from t_search.operators.listeners import EvalListener
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
    id: int # sequential id in a sequence of created fillings
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
                    fitness: Fitness,
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

        self.filling_id = 0
        self.hole_fillings: list[HoleFilling] = []
        self.present_fillings: set[Term] = set()

        self.good_filling_loss = good_filling_loss
        self.max_filling_queue_size = max_filling_queue_size 
        self.debug = debug
        self.const_optimizer = const_optimizer
        self.fitness = fitness

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
            # if self.debug: # outputing duplicates that were removed
            #     for t in terms:
            #         if t not in new_terms:
            #             print(f"[register_terms] duplicate: {t}")
            #     pass 
            return

        normalized_terms, normalized_semantics = self.term_index.get_new_normalized(new_terms)

        if len(normalized_terms) == 0:
            # print("WARN: no normalized terms found during registering terms")
            # if self.debug: # same semantics alreay represented by some simple terms: 
            #     for t in new_terms:
            #         print(f"[register_terms] different repr for: {t}")
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
                if isinstance(term, Value) and isinstance(hole[1].term, Value):      
                    # if self.debug:
                    #     print(f"\tskipping known constant: {hole[0]}@({hole[1].term}, {hole[1].occur})")
                    continue                
                self.add_fill_hole(term, hole[0], hole[1], l2)        
        pass

    def register_holes(self, holes: list[tuple[Term, TermPos]], hole_params: torch.Tensor) -> None:
        ''' Adds hole and its semantics to index and outputs currently present fillings '''
        if len(holes) == 0:
            return []
        
        new_holes = self.hole_index.insert(holes, hole_params)
        
        if len(new_holes) == 0:
            # if self.debug: # outputing duplicates that were removed
            #     for h in holes:
            #         if h not in new_holes:
            #             print(f"[register_holes] duplicate: {h[0]}@({h[1].term}, {h[1].occur})")
            #     pass             
            return

        normalized_holes, normalized_semantics = self.hole_index.get_new_normalized(new_holes)

        if len(normalized_holes) == 0:
            # print("WARN: no normalized holes found during registering holes")
            # if self.debug: # same semantics alreay represented by some simple holes: 
            #     for h in new_holes:
            #         print(f"[register_holes] different repr for: {h[0]}@({h[1].term}, {h[1].occur})")
            return
        
        found_terms = self.term_index.query_closest(normalized_semantics, 
                                     start_delta=self.start_delta, 
                                     multiplier=self.multiplier, 
                                     num_steps=self.num_steps,
                                     num_closest=self.num_closest)
        del normalized_semantics

        for hole, terms in zip(normalized_holes, found_terms):
            for (l2, term) in terms: 
                if isinstance(term, Value) and isinstance(hole[1].term, Value):      
                    # if self.debug:
                    #     print(f"\tskipping known constant: {hole[0]}@({hole[1].term}, {hole[1].occur})")
                    continue
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
    
    def reduce_consts(self, term: Term, 
                        ops: dict[str, Callable] = {"add": lambda vs: sum(v for v in vs), "mul": lambda vs: prod(v for v in vs)},
                        identities: dict[str, Callable] = {}
                      ) -> Term:
        ''' add/mul for binary is tansformed to one of varying arity and then all constants are combined
            then, we return to binary ops. Top-down transfomration.
        '''
        if not isinstance(term, Op) or (term.op_id not in ops):
            return term
        all_terms = deque([term])
        final_args = []
        while len(all_terms) > 0:
            current = all_terms.popleft()
            if isinstance(current, Op) and (current.op_id == term.op_id):
                for a in current.get_args():                    
                    all_terms.append(a)
            else:
                reduced_current = self.reduce_consts(current, ops=ops, identities=identities)
                final_args.append(reduced_current)
        const_terms, non_const_terms = [], []
        for a in final_args:
            (const_terms if isinstance(a, Value) else non_const_terms).append(a)
        # if len(const_terms) == 0: # nothing to reduce - leave as it was 
        #     return term
        # final_const = const_terms[0]
        final_const = None 
        if len(const_terms) == 1:
            final_const = const_terms[0]
        elif len(const_terms) > 1:
            reduce_fn = ops[term.op_id]
            new_const = reduce_fn([c.value for c in const_terms])
            final_const = new_const if isinstance(new_const, Value) else self.syntax.get_const(value=new_const)
        if (final_const is not None) and ((len(non_const_terms) == 0) or (term.op_id not in identities) or (not identities[term.op_id](final_const.value))):
            non_const_terms.append(final_const)
        if len(non_const_terms) == 1:
            return non_const_terms[0]
        new_term = self.syntax.get_op(term.op_id, *non_const_terms)
        return new_term
        
    def mul_identity(self, v) -> bool:
        return (v - self.syntax.one_value.value) < self.small_value
    
    def add_identity(self, v) -> bool:
        return (v - self.syntax.zero_value.value) < self.small_value
        
    def custom_rule_mul_add_one(self, term: Term) -> Term: 
        ''' Implements custom reduction rule for (mul (add ? 1) k) -> (add (mul ? k) k) 
            Goal: reduce number of constants while preserving the term size
            This is reduction at place - no deep transformation
        '''
        def arg_arg_shape(arg_arg: Term) -> bool: 
            if isinstance(arg_arg, Value) and self.mul_identity(arg_arg):
                return True
            return False
        def arg_shape(arg: Term) -> Optional[Term]:
            if isinstance(arg, Op) and (arg.op_id == "add"):
                args = arg.get_args()
                t1 = args[0]
                t2 = args[1]
                if arg_arg_shape(t1):
                    return t2
                if arg_arg_shape(t2):
                    return t1
            return None
        if isinstance(term, Op) and (term.op_id == "mul"):
            args = term.get_args()
            t1 = args[0]
            t2 = args[1]
            arg1 = arg_shape(t1)
            if arg1 is not None:
                new_mul = self.syntax.get_op("mul", arg1, t2)
                new_term = self.syntax.get_op("add", new_mul, t2)
                return new_term
            arg2 = arg_shape(t2)
            if arg2 is not None:
                new_mul = self.syntax.get_op("mul", arg2, t1)
                new_term = self.syntax.get_op("add", new_mul, t1)
                return new_term
        return term
        
    # def reduce_all_term_ops_consts(self, term: Term, 
    #                                 ops: dict[str, Callable] = {"add": lambda vs: sum(v for v in vs), "mul": lambda vs: prod(v for v in vs)},
    #                                 identities: dict[str, Callable] = {}) -> Term:
    #     if not isinstance(term, Op):
    #         return term
    #     new_term = term
    #     for op_id, op_reduce in ops.items():
    #         new_term = self.reduce_consts(new_term, ops, identities)
    #         if new_term != term: # cannot reduce further at point 
    #             break 
    #     if isinstance(new_term, Op):
    #         new_args = [self.reduce_all_term_ops_consts(arg, ops=ops) for arg in new_term.get_args()]
    #         final_term = self.syntax.get_op(new_term.op_id, *new_args)
    #         return final_term        
    #     return new_term

    # (add t k), (mul k t)
    def add_fill_hole(self, term: Term, hole_root: Term, hole_pos: TermPos, l2: float) -> Optional[HoleFilling]:

        # fit_subterm0 = self.strip_add_mul_consts(term)
        # if isinstance(fit_subterm0, Value):
        #     fit_term = fit_subterm0
        # else:
        if isinstance(term, Value):
            fit_term = term
        else:
            k_value = self.syntax.one_value
            b_value = self.syntax.zero_value
            fit_subterm1 = self.syntax.get_op("mul", k_value, term)
            fit_term = self.syntax.get_op("add", fit_subterm1, b_value)
        # NOTE: change hole pos to remove (mul ? c) and (add ? c).
        while hole_pos.parent is not None:   
            parent = hole_pos.parent             
            if isinstance(parent.term, Op) and \
                ((parent.term.op_id == "mul") or (parent.term.op_id == "add")) and \
                all(isinstance(a, Value) for arg_pos, a in enumerate(parent.term.get_args()) if arg_pos != hole_pos.pos):
                hole_pos = parent
                continue
            break                       

        # TODO: NOTE: here we do not handle const overflow 
        #             more precise handling woul try the term itself instead of linear combination
        #             when we above the const limit - for n ow it is intended. Just increase const limit in config (+2)
        new_term = self.syntax.replace_position(hole_root, hole_pos, fit_term)
        if new_term is None:
            # if self.debug:
            #     print(f"\tconstr violation: {fit_term} --> {hole_root}@({hole_pos.term}, {hole_pos.occur})")
            return None
        
        # reducing consstants before optimization 
        new_term = self.reduce_consts(new_term)

        optimized = self.const_optimizer.optimize(new_term, with_loss=True)
        # assert optimized.term is not None, "Const optimizer must return valid term"

        # optimized_reduced_term = self.reduce_consts(optimized.term, 
        #                                 identities={"add": self.add_identity, "mul": self.mul_identity}
        #                                 )
        
        new_filling_id = self.filling_id
        self.filling_id += 1
        hole_filling = HoleFilling(optimized.loss, l2, 
                                    new_filling_id,
                                    optimized.term,
                                #    optimized_reduced_term,
                            term, hole_root, hole_pos,
                            term_semantics=self.term_index.get_semantics_for_term(term, denormalize=True),
                            hole_semantics=self.hole_index.get_semantics_for_term((hole_root, hole_pos), denormalize=True)
                            ) # no skeletons here as we optimized constants
        
        # if self.debug and len(self.hole_fillings) > 0 and \
        #     ((self.hole_fillings[0].priority < hole_filling.priority and self.hole_fillings[0].l2 > hole_filling.l2) or \
        #         (self.hole_fillings[0].priority > hole_filling.priority and self.hole_fillings[0].l2 < hole_filling.l2)):
        #     print(f"loss/l2: new {hole_filling.priority:.4f}/{hole_filling.l2:.4f} vs best {self.hole_fillings[0].priority:.4f}/{self.hole_fillings[0].l2:.4f}")
        #     print(f">> {hole_filling.term}")
        #     pass

        if hole_filling.term in self.present_fillings:
            # if self.debug:
            #     print(f"\tduplicate filling: {hole_filling.term}")
            return 
        # if self.debug:
        #     print(f"\tadded {hole_filling.term}")        
        heappush(self.hole_fillings, hole_filling)      
        self.present_fillings.add(hole_filling.term)

    def has_fillings(self) -> bool:
        return len(self.hole_fillings) > 0   
    
    def get_best_hole_filling(self, force_pick: bool = False) -> HoleFilling | None:
        loss_threshold = self.good_filling_loss if self.fitness.best_term_fitness is None else min(self.good_filling_loss, self.fitness.best_term_fitness.item())
        while (len(self.hole_fillings) > 0) and \
              (force_pick or \
               (self.hole_fillings[0].priority < loss_threshold) or \
               (len(self.hole_fillings) > self.max_filling_queue_size)):

            filling = heappop(self.hole_fillings)
            self.present_fillings.remove(filling.term)
            return filling
        return None