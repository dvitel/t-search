import torch

from t_search.evaluators.evaluator import Evaluator
from .base import TermMutation
from t_search.syntax import Term
from t_search.syntax.stats import get_inner_terms

class BestInner(TermMutation):
    ''' Replaces each term with its inner term with best fitness '''

    def __init__(self, 
                 *,
                 evaluator: Evaluator,
                 **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.term_best_inner_term_cache: dict[Term, Term] = {}

    def mutate_term(self, term: Term) -> Term | None:
        if term in self.term_best_inner_term_cache:
            child = self.term_best_inner_term_cache[term]
            return child 
        inner_terms = get_inner_terms(term)
        # self.term_inner_terms_cache[term] = inner_terms
        inner_fitness = self.evaluator.eval(inner_terms, return_fitness="tensor").fitness
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
