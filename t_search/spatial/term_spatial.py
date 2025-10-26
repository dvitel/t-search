''' Adds syntax to plain vector storage '''

from bisect import insort
from pyparsing import Callable
import torch
from .base import VectorStorage
from syntax import Term


class TermVectorStorage:

    def __init__(self, index: VectorStorage):
        self.index = index
        self.sid_to_terms: dict[int, list[Term]] = {}
        self.term_to_sid: dict[Term, int] = {}

    def insert(self, terms: list[Term], vectors: torch.Tensor, eq_group_order_key: Callable) -> None:
        ''' Returns mapping of term to its id in the equivalence group '''
        ids = self.index.insert(vectors)
        for term, sid in zip(terms, ids):
            if sid not in self.sid_to_terms:
                self.sid_to_terms[sid] = []
            insort(self.sid_to_terms[sid], term, key=eq_group_order_key)
            self.term_to_sid[term] = sid
        return
    
    def get_repr_terms(self) -> list[Term]:
        repr_terms = [terms[0] for terms in self.sid_to_terms.values()]
        return repr_terms
    
    def get_repr_term(self, sem_id: int) -> Term:
        return self.sid_to_terms[sem_id][0]
    
    def len_sem(self) -> int:
        return len(self.sid_to_terms)
    
    def len_terms(self) -> int:
        return len(self.term_to_sid)
    
    def get_semantics(self) -> torch.Tensor:
        all_semantics = self.index.vectors[:self.index.cur_id]
        return all_semantics

    
    # def get_num_terms(self) -> int:
    #     return len(self.term_to_sid)
    
    # def get_num_sids(self) -> int: 
    #     return len(self.sid_to_terms)

    # def get_term_by_index(self, index: int) -> Term:
    #     return self.terms[index]