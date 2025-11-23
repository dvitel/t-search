''' Adds syntax to plain vector storage '''

from bisect import insort
from typing import Optional
from pyparsing import Callable
import torch
from ..spatial.base import VectorStorage
from t_search.syntax import Term


class TermVectorStorage:

    def __init__(self, index: VectorStorage):
        self.index = index
        self.sid_to_terms: dict[int, list[Term]] = {}
        self.term_to_sid: dict[Term, int] = {}

    def reset(self):
        self.sid_to_terms = {}
        self.term_to_sid = {}
        self.index.reset()

    def get_semantics_for_term(self, term: Term) -> Optional[torch.Tensor]:
        if term not in self.term_to_sid:
            return None
        sid = self.term_to_sid[term]
        vector = self.index.get_vectors(sid)
        return vector
    
    def get_term_for_semantics(self, vector: torch.Tensor) -> Optional[Term]:
        sid, *_ = self.index.query_points(vector.unsqueeze(0))
        if sid == -1:
            return None
        terms = self.sid_to_terms.get(sid, [])
        if not terms:
            return None
        return terms[0] # repr

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
        
    def get_semantics(self) -> torch.Tensor:
        all_semantics = self.index.get_vectors()
        return all_semantics

    def num_sem(self) -> int:
        return len(self.sid_to_terms)
    
    def num_terms(self) -> int:
        return len(self.term_to_sid)
    