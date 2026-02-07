from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from math import prod
from typing import Callable, Literal, Optional

import torch

from t_search.base import ServiceBase
from t_search.evaluators.const_optimizer import ConstOptimizer
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.semantics import Semantics
from t_search.evaluators.term_spatial import HoleVectorStorage, Normalizer, TermVectorStorage
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
    start_loss: float
    l2: float
    id: int # sequential id in a sequence of created fillings
    term: Term
    found_term: Term 
    hole_root: Term 
    hole_pos: TermPos
    # term_semantics: torch.Tensor
    # hole_semantics: torch.Tensor
    # skeletons: list[Term]

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return f"HF[{self.id:04d}](loss={self.priority:.3f}←{self.start_loss:4.3f} l2={self.l2:.2f}\n\t\t{self.term}\n\t\t{self.hole_root}@({self.hole_pos.term}, {self.hole_pos.occur}) with {self.found_term})"

class TermHolePairs(EvalListener, ServiceBase):
    ''' For new terms search for sketches, for new sketches search for terms.  '''

    def __init__(self, *, 
                    target: torch.Tensor,
                    syntax: Syntax,
                    # evaluator: Evaluator,
                    normalizer: Normalizer,
                    fitness: Fitness,
                    semantics: Semantics,
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
                    term_filter_strategy: str = "term_filter_accept_all",
                    hole_filter_strategy: str = "hole_filter_accept_all",
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
        self.delayed_terms: list[tuple[Term, torch.Tensor]] = [] # new terms on eval 
        self.present_fillings: set[Term] = set()

        self.good_filling_loss = good_filling_loss
        self.max_filling_queue_size = max_filling_queue_size 
        self.debug = debug
        self.const_optimizer = const_optimizer
        self.fitness = fitness
        self.semantics = semantics
        self.normalizer = normalizer
        self.term_filter_strategy = getattr(self, term_filter_strategy)
        self.hole_filter_strategy = getattr(self, hole_filter_strategy)

        pass 

    # def init(self):
    #     # NOTE: we have to add to term index at least one constant to discover constant terms for holes
    #     if self.syntax.max_consts > 0:
    #         zero_term = self.syntax.get_const(value=0.0)
    #         # WARNINING: next fake eval works outsied evaluator object - hack that would avoid circular dependencies 
    #         # Instead, Value(0) term may have different semantics for different evaluators 
    #         fake_eval = torch.zeros_like(self.target) # NOTE: for now we assume it is ok 
    #         self.register_terms([zero_term], fake_eval.unsqueeze(0))
    #         del fake_eval
    #         pass 
    #     pass        

    def on_eval(self, terms: list[Term], semantics: torch.Tensor):
        ''' New terms appear, queue themfor later optimization '''
        # filtered_terms = [t for t in terms if self.syntax.get_num_consts(t) == 0]
        self.insert_new_terms(terms, semantics)
        # self.register_terms(terms, semantics)
        pass

    def deduplicate(self, terms: list[Term], vectors: torch.Tensor) -> tuple[list[Term], torch.Tensor]:
        ''' Remove duplicates from terms based on their vectors '''
        unique_set = set()
        unique_ids = []
        for i, t in enumerate(terms):
            if t in unique_set or not self.semantics.is_valid(t):
                continue
            unique_set.add(t)
            unique_ids.append(i)
        if len(unique_ids) == len(terms):
            return terms, vectors
        unique_terms = [terms[i] for i in unique_ids]
        unique_vectors = vectors[unique_ids]
        return unique_terms, unique_vectors
    
    def split_const_nonconst(self, terms: list[Term], vectors: torch.Tensor, return_consts: bool = False) -> tuple:
        term_consts = self.semantics.is_const(vectors)

        const_terms = [] 
        nonconst_term_ids = []

        for hid, c in enumerate(term_consts):
            if c is not None and return_consts:
                const_terms.append((terms[hid], self.syntax.get_const(value=c)))
            else:
                nonconst_term_ids.append(hid)  

        if len(nonconst_term_ids) < len(terms):
            terms = [terms[i] for i in nonconst_term_ids]
            vectors = vectors[nonconst_term_ids]

        return const_terms, terms, vectors
    
    def term_filter_accept_all(self, terms: list[Term], vectors: torch.Tensor) -> tuple:
        return terms, vectors
    
    def hole_filter_accept_all(self, holes: list[tuple[Term, TermPos]], vectors: torch.Tensor) -> tuple:
        return holes, vectors
    
    def term_filter_no_consts(self, terms: list[Term], vectors: torch.Tensor) -> tuple:
        term_ids = []
        for hid, t in enumerate(terms):
            if self.syntax.get_num_consts(t) == 0:
                term_ids.append(hid)
        if len(term_ids) < len(terms):
            terms = [terms[i] for i in term_ids]
            vectors = vectors[term_ids]        
        return terms, vectors
    
    def insert_new_terms(self, terms: list[Term], term_params: torch.Tensor) -> None:

        terms, term_params = self.deduplicate(terms, term_params)
        
        _, terms, term_params = self.split_const_nonconst(terms, term_params, return_consts=False)
        if len(terms) == 0:
            return
        
        new_terms, new_term_params = self.term_index.filter_new(terms, term_params)  
        if len(new_terms) == 0: # already queried earlier as querying happens just after insert
            return
        
        new_normalized, new_terms = self.normalizer.normalize(new_term_params, new_terms)

        if len(new_terms) == 0:
            return
        
        #deduplication by normalized semantics
        new_terms, new_normalized = self.term_index.find_unique(new_terms, new_normalized, return_type='entries')
        if len(new_terms) == 0:
            return
        
        term_to_insert, vectors_to_insert = self.term_filter_strategy(new_terms, new_normalized)

        self.term_index.insert(term_to_insert, vectors_to_insert)

        # adding mew_terms, new_normalized to delayed list
        # for t, v in zip(new_terms, new_normalized):
        self.delayed_terms.append((new_terms, new_normalized))
        pass
       
    def register_delayed_terms(self) -> list[HoleFilling]:

        result_fillings = []
        if len(self.delayed_terms) == 0:
            return result_fillings
        
        for new_terms, new_normalized in self.delayed_terms:

            # searching for nearby holes 
            found_holes = self.hole_index.query_closest(new_normalized, 
                                        start_delta=self.start_delta, 
                                        multiplier=self.multiplier, 
                                        num_steps=self.num_steps,
                                        num_closest=self.num_closest)
            del new_normalized

            for term, holes in zip(new_terms, found_holes):
                for (l2, hole) in holes:
                    new_filling = self.fill_holes(term, hole[0], hole[1], l2, use_global_threshold=True)        
                    if new_filling is not None:
                        result_fillings.append(new_filling)          

        self.delayed_terms.clear() 

        return result_fillings

    def register_holes(self, holes: list[tuple[Term, TermPos]], 
                        hole_params: torch.Tensor) -> list[HoleFilling]:
        ''' Adds new sketches and produces good terms '''
        result_fillings = []
        if len(holes) == 0:
            return result_fillings
        
        holes, hole_params = self.deduplicate(holes, hole_params)
        
        const_holes, holes, hole_params = self.split_const_nonconst(holes, hole_params, return_consts=True)

        for hole, term in const_holes:
            new_filling = self.fill_holes(term, hole[0], hole[1], 0.0)
            if new_filling is not None:
                result_fillings.append(new_filling)

        if len(holes) == 0:
            return result_fillings
        
        new_holes, new_hole_params = self.hole_index.filter_new(holes, hole_params)
        if len(new_holes) == 0:          
            return result_fillings
        
        new_normalized, new_holes = self.normalizer.normalize(new_hole_params, new_holes)

        if len(new_holes) == 0:
            return result_fillings
        
        # NOTE: should we deduplicate holes by normalized semantics? 
                #deduplication by normalized semantics
        new_holes, new_normalized = self.hole_index.find_unique(new_holes, new_normalized, return_type='entries')
        if len(new_holes) == 0:
            return result_fillings
        
        holes_to_insert, vectors_to_insert = self.hole_filter_strategy(new_holes, new_normalized)
        
        self.hole_index.insert(holes_to_insert, vectors_to_insert)        

        # normalized_holes, normalized_semantics = self.hole_index.get_new_normalized(new_holes, allow_nonrepr=True)

        found_terms = self.term_index.query_closest(new_normalized, 
                                     start_delta=self.start_delta, 
                                     multiplier=self.multiplier, 
                                     num_steps=self.num_steps,
                                     num_closest=self.num_closest)
        del new_normalized
        
        for hole, terms in zip(new_holes, found_terms):
            for (l2, term) in terms: 
                new_filling = self.fill_holes(term, hole[0], hole[1], l2)
                if new_filling is not None:
                    result_fillings.append(new_filling)
        return result_fillings

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

    def fill_holes(self, term: Term, hole_root: Term, hole_pos: TermPos, l2: float,
                        use_global_threshold: bool = False) -> Optional[HoleFilling]:

        # if self.debug:
        #     print(f"Proposed term: {term} for l2={l2:.2f}, {hole_root}@({hole_pos.term}, {hole_pos.occur})")

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
            #     print(f"\tDiscarded: {fit_term} --> {hole_root}@({hole_pos.term}, {hole_pos.occur})")
            return None
        
        # reducing consstants before optimization 
        new_term = self.reduce_consts(new_term)

        optimized = self.const_optimizer.optimize(new_term, with_loss=True)

        if use_global_threshold and self.fitness.best_term_fitness is not None:
            original_loss = self.fitness.best_term_fitness.item()
        else:
            original_loss = self.fitness.get_fitness(hole_root).item()

        if optimized.loss >= original_loss:
            return None
        
        # assert optimized.term is not None, "Const optimizer must return valid term"

        # optimized_reduced_term = self.reduce_consts(optimized.term, 
        #                                 identities={"add": self.add_identity, "mul": self.mul_identity}
        #                                 )
        
        new_filling_id = self.filling_id
        self.filling_id += 1
        hole_filling = HoleFilling(optimized.loss, 
                                    original_loss,
                                    l2, 
                                    new_filling_id,
                                    optimized.term,
                                #    optimized_reduced_term,
                            term, hole_root, hole_pos,
                            ) # no skeletons here as we optimized constants

        # if self.debug: 
        #     # 0. debug - make sure everything is cocmputed correctly
        #     #    0.1 Consts should not be in term_index/hole_index --> register_term/register_hole should filter out constants
        #     #    0.2 On uneval we check obtained trace for constant (query_closest)
        #     # asserting correctness of computed semantics and l2                                     
        #     term_semantics=self.term_index.get_semantics_for_term(term, denormalize=False)
        #     hole_semantics=self.hole_index.get_semantics_for_term((hole_root, hole_pos), denormalize=False)
        #     _l2 = dl2(term_semantics, hole_semantics)
        #     assert abs(_l2 - l2) < 1e-5, f"Computed l2={l2} differs from actual l2={_l2}"
        #     pass

        # if self.debug and len(self.hole_fillings) > 0 and \
        #     ((self.hole_fillings[0].priority < hole_filling.priority and self.hole_fillings[0].l2 > hole_filling.l2) or \
        #         (self.hole_fillings[0].priority > hole_filling.priority and self.hole_fillings[0].l2 < hole_filling.l2)):
        #     print(f"loss/l2: new {hole_filling.priority:.4f}/{hole_filling.l2:.4f} vs best {self.hole_fillings[0].priority:.4f}/{self.hole_fillings[0].l2:.4f}")
        #     print(f">> {hole_filling.term}")
        #     pass

        # if self.debug:
        #     print(f"\tNew loss={hole_filling.priority:.4f}, {hole_filling.term}")  
        #     print("\tVectors: ")      
        #     term_semantics = self.term_index.get_semantics_for_term(hole_filling.found_term, denormalize=True)
        #     term_semantics_normalized = self.term_index.get_semantics_for_term(hole_filling.found_term, denormalize=False)
        #     hole_semantics = self.hole_index.get_semantics_for_term((hole_filling.hole_root, hole_filling.hole_pos), denormalize=True)
        #     hole_semantics_normalized = self.hole_index.get_semantics_for_term((hole_filling.hole_root, hole_filling.hole_pos), denormalize=False)
        #     print(f"\t\tx=torch.tensor([{', '.join([f'{f:+5.4f}' for f in self.semantics.var_bindings['x0'].tolist()])}])")
        #     print(f"\t\ty=torch.tensor([{', '.join([f'{f:+5.4f}' for f in self.target.tolist()])}])")
        #     print(f"\t\tts=torch.tensor([{', '.join([f'{f:+5.4f}' for f in term_semantics.tolist()])}])")
        #     print(f"\t\ths=torch.tensor([{', '.join([f'{f:+5.4f}' for f in hole_semantics.tolist()])}])")
        #     print(f"\t\ttsN=torch.tensor([{', '.join([f'{f:+5.4f}' for f in term_semantics_normalized.tolist()])}])")
        #     print(f"\t\thsN=torch.tensor([{', '.join([f'{f:+5.4f}' for f in hole_semantics_normalized.tolist()])}])")

        # TODO: FINISH THIS
        # if hole_filling.term in self.present_fillings:
        #     # if self.debug:
        #     #     print(f"\tDuplicate dicarded")
        #     return 
        # heappush(self.hole_fillings, hole_filling)      
        # self.present_fillings.add(hole_filling.term)
        return hole_filling