from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Optional

import torch

from t_search.base import ServiceBase
from t_search.evaluators.term_spatial import HoleVectorStorage, TermVectorStorage
from t_search.operators.listeners import EvalListener
from t_search.operators.mutation import TermMutation
from t_search.syntax import Term, TermPos
from t_search.syntax.syntax import Syntax

@dataclass(order=True)
class PriorityPair:
    priority: float
    term: Term = field(compare=False)
    hole: tuple[Term, TermPos] = field(compare=False)

class TermHolePairs(EvalListener, ServiceBase):
    ''' For new terms search for sketches, for new sketches search for terms.  '''

    def __init__(self, *, 
                    target: torch.Tensor,
                    syntax: Syntax,
                    # evaluator: Evaluator,
                    # fitness: Fitness,
                    term_index: TermVectorStorage,
                    hole_index: HoleVectorStorage,
                    syn_simplifier: TermMutation | None = None,
                    start_delta: float = 1e-5,
                    multiplier: float = 10,
                    num_steps: int = 3,
                    num_closest: int = 3,
                    small_value: float = 1e-5):

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
        self.small_value = small_value

        self.term_hole_pairs: list[PriorityPair] = [] # priority queue

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
        return self.register_terms(terms, semantics)
       
    def register_terms(self, terms: list[Term], term_params: torch.Tensor) -> list[Term]: 
        if len(terms) == 0:
            return []

        self.term_index.insert(terms, term_params)

        normalized_semantics = self.term_index.get_semantics_for_terms(terms, denormalize=False)
        # searching for nearby holes 
        found_holes = self.hole_index.query_closest(normalized_semantics, 
                                     start_delta=self.start_delta, 
                                     multiplier=self.multiplier, 
                                     num_steps=self.num_steps,
                                     num_closest=self.num_closest)
        del normalized_semantics

        for term, holes in zip(terms, found_holes):
            for (l2, hole) in holes:
                heappush(self.term_hole_pairs, PriorityPair(l2, term, hole))
        
        pass

    def register_holes(self, holes: list[tuple[Term, TermPos]], hole_params: torch.Tensor) -> list[Term]:
        ''' Adds hole and its semantics to index and outputs currently present fillings '''
        if len(holes) == 0:
            return []
        
        self.hole_index.insert(holes, hole_params)

        normalized_semantics = self.hole_index.get_semantics_for_terms(holes, denormalize=False)
        found_terms = self.term_index.query_closest(normalized_semantics, 
                                     start_delta=self.start_delta, 
                                     multiplier=self.multiplier, 
                                     num_steps=self.num_steps,
                                     num_closest=self.num_closest)
        del normalized_semantics

        for hole, terms in zip(holes, found_terms):
            for (l2, term) in terms:
                heappush(self.term_hole_pairs, PriorityPair(l2, term, hole))

        pass 

    def fill_hole(self, term: Term, hole_root: Term, hole_pos: TermPos) -> Optional[Term]:
        # NOTE: we commented the followign as rescaling would create different term
        # if hole_pos.term == term:
        #     return None 

        term_semantics = self.term_index.get_semantics_for_term(term)
        hole_semantics = self.hole_index.get_semantics_for_term((hole_root, hole_pos))

        # we compute k * ts + b that is closest to hs
        # (k * ts + b - hs)^2 --> min
        # Sx = sum(ts), Sy = sum(hs), Sxx = sum(ts^2), Sxy = sum(ts * hs)

        # (k * x + b - y)^2 --> min
        # 2 (k * x + b - y) * x = 0    

        Sx = term_semantics.sum()
        Sy = hole_semantics.sum()
        Sxx = (term_semantics * term_semantics).sum()
        Sxy = (term_semantics * hole_semantics).sum()
        n = term_semantics.shape[0]
        n_Covar_xy = (n * Sxy - Sx * Sy)
        n_Var_x = (n * Sxx - Sx * Sx)
        if n_Var_x / n < self.small_value: # no variation of x - x == c - searching for best b:
            b = Sy / n # approximate with constant 
            hole_term = self.syntax.get_const(value=b)
        else:
            k = n_Covar_xy / n_Var_x
            b = (Sy - k * Sx) / n
            
            if torch.abs(k) < self.small_value:
                # approximate with constant 
                hole_term = self.syntax.get_const(value=b)
            elif torch.abs(b) < self.small_value:
                # approximate with scaling only
                hole_term = self.syntax.get_op("mul", self.syntax.get_const(value=k), term)
            else: # general case        
                hole_term = self.syntax.get_op("add", 
                                self.syntax.get_op("mul", self.syntax.get_const(value=k), term),
                                self.syntax.get_const(value=b))
        
        if hole_term is None:
            return None
        
        new_term = self.syntax.replace_position(hole_root, hole_pos, hole_term)
        if new_term is None:
            return None

        if self.syn_simplifier is None:
            return new_term
        
        new_simplified = self.syn_simplifier.mutate_term(new_term)
        
        return new_simplified  
    

    def get_best_term_hole_pairs(self, max_pairs: int) -> list[tuple[Term, tuple[Term, TermPos]]]:
        res: list[tuple[Term, tuple[Term, TermPos]]] = []
        while len(res) < max_pairs and len(self.term_hole_pairs) > 0:
            pp = heappop(self.term_hole_pairs)
            res.append((pp.term, pp.hole))
        return res        
    
    def get_best_hole_fillings(self, max_fillings: int) -> list[tuple[Term, PriorityPair]]:
        res: list[tuple[Term, PriorityPair]] = []
        while len(res) < max_fillings and len(self.term_hole_pairs) > 0:
            pp = heappop(self.term_hole_pairs)
            filled = self.fill_hole(pp.term, pp.hole[0], pp.hole[1])
            if filled is not None:

                # asserts that filled holes produce better outcomes than original 
                # self.evaluator.eval(filled)
                # new_fitness = self.fitness.get_fitness(filled)
                # old_fitness = self.fitness.get_fitness(pp.hole[0])
                # assert new_fitness < old_fitness, "Filling must improve fitness"

                res.append((filled, pp))
        return res