''' Adds syntax to plain vector storage '''

from bisect import insort
from typing import Any, Callable, Optional
import torch

from t_search.base import ServiceBase
from ..spatial.base import VectorStorage
from t_search.syntax import Term

class InvalidTerms(ServiceBase):

    def __init__(self):
        self.terms: dict[Term, torch.Tensor] = {}

    def add_invalid(self, term: Term, outputs: torch.Tensor) -> None:
        self.terms[term] = outputs

    def is_invalid(self, term: Term) -> bool:
        return term in self.terms
    
    def get_outputs(self, term: Term) -> Optional[torch.Tensor]:
        return self.terms.get(term, None)
    
    def __len__(self) -> int:
        return len(self.terms)
    
    def get_finalizer(self):
        def finalizer():
            for outputs in self.terms.values():
                del outputs
        return finalizer


class TermVectorStorage:

    def __init__(self, *, 
                 term_order: Callable[[Term], Any],
                 index: VectorStorage):
        self.term_order = term_order
        self.index = index
        self.sid_to_terms: dict[int, list[Term]] = {}
        self.term_to_sid: dict[Term, int] = {}

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

    def insert(self, terms: list[Term], vectors: torch.Tensor) -> None:
        ''' Returns mapping of term to its id in the equivalence group '''
        ids = self.index.insert(vectors)
        for term, sid in zip(terms, ids):
            if sid not in self.sid_to_terms:
                self.sid_to_terms[sid] = []
            insort(self.sid_to_terms[sid], term, key=self.term_order)
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
    