''' Adds syntax to plain vector storage '''

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Protocol, Sequence
from pyparsing import TypeVar
import torch

from t_search.base import ServiceBase
from t_search.spatial.base import VectorStorage
from t_search.syntax.term import Term, TermPos

TTermPos = TypeVar('TTermPos')

class Normalizer(Protocol, Generic[TTermPos]):
    def normalize(self, vectors: torch.Tensor, terms: list[TTermPos]) -> torch.Tensor:
        pass 

    def denormalize(self, normalized_vectors: torch.Tensor, terms: list[TTermPos]) -> torch.Tensor:
        pass 

    def get_params(self, term: TTermPos) -> Any:
        return None

class IdentityNormalizer(Normalizer):
    def normalize(self, vectors: torch.Tensor, terms: list[Any]) -> torch.Tensor:
        return vectors

    def denormalize(self, normalized_vectors: torch.Tensor, terms: list[Any]) -> torch.Tensor:
        return normalized_vectors
    
@dataclass 
class ZScoreParams:
    mean: torch.Tensor
    std: torch.Tensor
    
class ZScoreNormalizer(Normalizer, ServiceBase, Generic[TTermPos]):
    """Z-score normalization (mean=0, std=1)."""
    
    def __init__(self, zero, one, small_std: float = 1e-5):
        self.zero = zero
        self.one = one
        self.small_std = small_std
        self.params: dict[TTermPos, tuple[torch.Tensor, torch.Tensor]] = {}
    
    def normalize(self, vectors: torch.Tensor, keys: list[TTermPos]) -> torch.Tensor:
        means = torch.mean(vectors, dim=1, keepdim=True)
        stds = torch.std(vectors, dim=1, keepdim=True)
        zero_mask = torch.all(torch.isclose(stds, self.zero, atol=self.small_std, rtol=0), dim=1)
        normalized = (vectors - means) / stds
        normalized[zero_mask] = self.zero
        stds[zero_mask] = self.zero
        
        # Store normalization params per key
        for key, mean, std in zip(keys, means.squeeze(-1), stds.squeeze(-1)):
            self.params[key] = (mean, std)
        
        return normalized
    
    def denormalize(self, vectors: torch.Tensor, keys: list[TTermPos]) -> torch.Tensor:
        """Denormalize using stored params."""
        result = torch.empty_like(vectors)
        means, stds = zip(*[self.params[key] if key in self.params else (self.zero, self.one) for key in keys])
        mean_tensor = torch.stack(means)
        std_tensor = torch.stack(stds)
        result = torch.where(std_tensor > 0, vectors * std_tensor + mean_tensor, mean_tensor)
        return result
    
    def get_params(self, term: TTermPos) -> ZScoreParams:
        mean, std = self.params.get(term, (None, None))
        if mean is None: 
            return {}
        return ZScoreParams(mean, std)
    
    def get_finalizer(self) -> Callable[[], None]:
        def cleanup():
            for mean, std in self.params.values():
                del mean, std
            self.params.clear()
        return cleanup

class BaseVectorStorage(ServiceBase, Generic[TTermPos]):

    def __init__(self, *, 
                 term_order: Callable[[TTermPos], Any],
                 normalizer: Normalizer[TTermPos],
                 add_metrics: Callable,
                 index: VectorStorage):
        self.term_order = term_order
        self.index = index
        self.normalizer: Normalizer[TTermPos] = normalizer
        self.sid_to_term: dict[int, TTermPos] = {} # stores only one representative term per semantics id
        self.term_to_sid: dict[TTermPos, int] = {}
        self.invalid_terms: dict[TTermPos, torch.Tensor] = {}
        self.add_metrics = add_metrics

    def get_finalizer(self):
        def finalizer():
            for v in self.invalid_terms.values():
                del v
            self.invalid_terms.clear()
        self.add_metrics(invalid_terms=len(self.invalid_terms))
        return finalizer
    
    def is_invalid(self, term: TTermPos) -> bool:
        return term in self.invalid_terms
    
    def get_invalid_semantics(self, term: TTermPos, denormalize: bool = True) -> Optional[torch.Tensor]:
        vector = self.invalid_terms.get(term, None)
        if vector is not None and denormalize:
            vector = self.normalizer.denormalize(vector.unsqueeze(0), [term]).squeeze(0)
        return vector

    def get_semantics_for_term(self, term: TTermPos, denormalize: bool = True) -> Optional[torch.Tensor]:
        if term not in self.term_to_sid:
            return self.get_invalid_semantics(term, denormalize=denormalize)
        sid = self.term_to_sid[term]
        vector = self.index.get_vectors(sid)
        if denormalize:
            vector = self.normalizer.denormalize(vector.unsqueeze(0), [term]).squeeze(0)
        return vector
    
    def get_semantics_for_terms(self, terms: list[TTermPos], denormalize: bool = True) -> torch.Tensor:
        res = torch.empty((len(terms), self.index.dims), dtype=self.index.dtype, device=self.index.device)
        sids: list[int] = []
        tids: list[int] = []
        for tid, term in enumerate(terms):
            if term in self.term_to_sid:
                sids.append(self.term_to_sid[term])
                tids.append(tid)
            else:
                invalid_sem = self.get_invalid_semantics(term, denormalize=False)
                if invalid_sem is not None:
                    res[tid] = invalid_sem
        if len(sids) > 0:
            vectors = self.index.get_vectors(sids)
            res[tids] = vectors
        if denormalize:
            res = self.normalizer.denormalize(res, terms)
        return res
    
    def get_term_for_semantics(self, vector: torch.Tensor) -> Optional[TTermPos]:
        mapped_vectors = self.normalizer.normalize(vector.unsqueeze(0), terms=[])
        sid, *_ = self.index.query_points(mapped_vectors)
        if sid == -1:
            return None
        term = self.sid_to_term.get(sid, None)
        return term

    def insert(self, terms: list[TTermPos], vectors: torch.Tensor) -> None:
        ''' Returns mapping of term to its id in the equivalence group '''
        mapped_vectors = self.normalizer.normalize(vectors, terms)
        finite_all_mask = torch.isfinite(mapped_vectors)
        finite_mask = torch.all(finite_all_mask, dim=1)
        invalid_ids, = torch.where(~finite_mask)
        for iid in invalid_ids.tolist():
            self.invalid_terms[terms[iid]] = vectors[iid].clone()
        finite_ids, = torch.where(finite_mask)
        finite_terms = [terms[i] for i in finite_ids.tolist()]
        finite_mapped_vectors = mapped_vectors[finite_ids]
        del mapped_vectors, finite_all_mask, finite_mask, finite_ids, invalid_ids
        ids = self.index.insert(finite_mapped_vectors)
        del finite_mapped_vectors
        for term, sid in zip(finite_terms, ids):
            if sid not in self.sid_to_term:
                self.sid_to_term[sid] = term 
            elif self.term_order(term) < self.term_order(self.sid_to_term[sid]):
                self.sid_to_term[sid] = term
            self.term_to_sid[term] = sid
        return
    
    def get_repr_terms(self) -> list[TTermPos]:
        repr_terms = [term for term in self.sid_to_term.values()]
        return repr_terms
        
    def get_semantics(self) -> torch.Tensor:
        all_semantics = self.index.get_vectors()
        return all_semantics

    def num_sem(self) -> int:
        return len(self.sid_to_term)
    
    def num_terms(self) -> int:
        return len(self.term_to_sid)

    def query_closest(self, queries: Sequence[torch.Tensor],
                            start_delta: float = 1e-5,
                            multiplier: float = 10,
                            num_steps: int = 3,
                            num_closest: int = 3) -> list[list[tuple[float, TTermPos]]]:
        ''' Returns map: query id to found ids in index (list) '''
        
        res = []
        for q in queries:
            delta = start_delta
            q_res = None
            left_num_steps = num_steps
            while left_num_steps > 0:
                range = torch.stack([q - delta, q + delta], dim=0)
                found_ids = self.index.query_range(range)
                if len(found_ids) > 0:
                    vectors = self.index.get_vectors(found_ids)
                    l2 = torch.sum((vectors - q.unsqueeze(0)) ** 2, dim=1)
                    l2_sort_order = torch.argsort(l2)
                    min_l2_ids = l2_sort_order[:num_closest]
                    q_res = [(l2[fid].item(), self.sid_to_term[found_ids[fid]]) for fid in min_l2_ids.tolist()]
                    del l2, vectors, found_ids
                    break
                delta *= multiplier    
                left_num_steps -= 1      
            if q_res is not None:                  
                res.append(q_res)
        return res
    
    def closest_or_self(self, queries: torch.Tensor,
                            start_delta: float = 1e-5,
                            multiplier: float = 10,
                            num_steps: int = 3) -> torch.Tensor:
        ''' Returns closest vector or self '''
        
        closest = torch.clone(queries)
        for qid, q in enumerate(queries):
            delta = start_delta
            while num_steps > 0:
                range = torch.stack([q - delta, q + delta], dim=0)
                found_ids = self.index.query_range(range)
                if len(found_ids) > 0:
                    vectors = self.index.get_vectors(found_ids)
                    l2 = torch.sum((vectors - q.unsqueeze(0)) ** 2, dim=1)
                    l2_id = torch.argmin(l2)
                    closest[qid] = vectors[l2_id]
                    break
                delta *= multiplier    
                num_steps -= 1   
        
        return closest
        
class TermVectorStorage(BaseVectorStorage[Term]):
    pass 

class HoleVectorStorage(BaseVectorStorage[tuple[Term, TermPos]]):
    pass