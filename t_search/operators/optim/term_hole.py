from heapq import heappop, heappush
from typing import Optional

import torch

from t_search.evaluators.term_spatial import HoleVectorStorage, TermVectorStorage
from t_search.operators.listeners import EvalListener
from t_search.operators.mutation import TermMutation
from t_search.syntax import Term, TermPos
from t_search.syntax.syntax import Syntax

class TermHolePairs(EvalListener):
    ''' For new terms search for sketches, for new sketches search for terms.  '''

    def __init__(self, *, 
                    target: torch.Tensor,
                    syntax: Syntax,
                    term_index: TermVectorStorage,
                    hole_index: HoleVectorStorage,
                    syn_simplifier: TermMutation | None = None,
                    start_delta: float = 1e-5,
                    multiplier: float = 10,
                    num_steps: int = 3,
                    num_closest: int = 3):

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

        self.term_hole_pairs: list[tuple[float, Term, tuple[Term, TermPos]]] = [] # priority queue

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
                heappush(self.term_hole_pairs, (l2, term, hole))
        
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
                heappush(self.term_hole_pairs, (l2, term, hole))

        pass 

    def fill_hole(self, term: Term, hole_root: Term, hole_pos: TermPos) -> Optional[Term]:
        if hole_pos.term == term:
            return None 

        term_semantics = self.term_index.get_semantics_for_term(term)
        hole_semantics = self.hole_index.get_semantics_for_term((hole_root, hole_pos))

        # we compute k * ts + b that is closest to hs
        # (k * ts + b - hs)^2 --> min
        # Sx = sum(ts), Sy = sum(hs), Sxx = sum(ts^2), Sxy = sum(ts * hs)

        Sx = term_semantics.sum()
        Sy = hole_semantics.sum()
        Sxx = (term_semantics * term_semantics).sum()
        Sxy = (term_semantics * hole_semantics).sum()
        n = term_semantics.shape[0]
        k = (n * Sxy - Sx * Sy) / (n * Sxx - Sx * Sx)
        b = (Sy - k * Sx) / n
        if torch.isfinite(k) == False or torch.isfinite(b) == False:
            return None
        
        hole_term = self.syntax.get_op("add", 
                        self.syntax.get_op("mul", self.syntax.get_const(value=k), term),
                        self.syntax.get_const(value=b))
        
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
            _, term, hole = heappop(self.term_hole_pairs)
            res.append((term, hole))
        return res        
    
    def get_best_hole_fillings(self, max_fillings: int) -> list[Term]:
        res: list[Term] = []
        while len(res) < max_fillings and len(self.term_hole_pairs) > 0:
            _, term, hole = heappop(self.term_hole_pairs)
            filled = self.fill_hole(term, hole[0], hole[1])
            if filled is not None:
                res.append(filled)
        return res