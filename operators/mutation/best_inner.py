from typing import TYPE_CHECKING

import torch
from .base import TermMutation
from term import Term, get_inner_terms

if TYPE_CHECKING:
    from gp import GPSolver

class BestInner(TermMutation):
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
    
        # NOTE: next is for taking K best 
        # sort_ids = torch.argsort(inner_fitness) 
        # best_ids = sort_ids[:self.inner_cnt]
        # best_inners = [present_terms[i] for i in best_ids.tolist()]
        # if len(present_terms) == len(inner_terms):
        #     self.term_best_inner_term_cache[term] = best_inners
        # del inner_fitness        
