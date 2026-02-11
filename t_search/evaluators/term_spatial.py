''' Adds syntax to plain vector storage '''

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, Optional, Protocol, Sequence
from pyparsing import TypeVar
import torch

from t_search.base import ServiceBase
from t_search.spatial.base import VectorStorage
from t_search.syntax.term import Term, TermPos
from t_search.utils import rank

@dataclass(frozen=False)
class QueryClosestParams:
    delta: float = 1e-5
    multiplier: float = 10
    num_steps: int = 3
    num_closest: int = 3
    exclude_ids: set[int] = field(default_factory=set)

TTermPos = TypeVar('TTermPos')

@dataclass(frozen=True)
class QueryClosestResult:
    l2: list[float] = field(default_factory=list)
    found_ids: list[int] = field(default_factory=list)
    terms: list[TTermPos] = field(default_factory=list)
    final_delta: float = 0.0

class Normalizer(Protocol, Generic[TTermPos]):
    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Normalizes term vectors and removes bad terms'''
        pass 
    def get_normalized_target(self) -> torch.Tensor:
        pass

class IdentityNormalizer(Normalizer):
    def __init__(self, target: torch.Tensor, **_):
        self.target = target

    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Normalizes term vectors and removes bad terms'''
        return vectors.clone()
    def get_normalized_target(self) -> torch.Tensor:
        return self.target
    
class ZScoreNormalizer(Normalizer):
    """Z-score normalization (mean=0, std=1)."""
    
    def __init__(self, target: torch.Tensor,
                 epsilon: float = 1e-7):
        self.epsilon = epsilon
        self.target = target
        self.target_normalized = None 
        self.target_normalized = self.normalize(target.unsqueeze(0))
        pass

    def get_normalized_target(self) -> torch.Tensor:
        return self.target_normalized.squeeze(0)
    
    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Normalizes term vectors and removes constant terms'''
        means = torch.mean(vectors, dim=1, keepdim=True)
        stds = torch.std(vectors, dim=1, keepdim=True)
        stds[stds < self.epsilon] = self.epsilon
        normalized = (vectors - means) / stds
        if self.target_normalized is None:
            return normalized
        correlations = torch.sum(normalized * self.target_normalized, dim=1)
        neg_corr_mask = correlations < 0
        normalized[neg_corr_mask] *= -1.0
        return normalized
    
class ZRankNormalizer(ZScoreNormalizer):
    """Z-score on ranks. See ZScoreNormalizer. ZScore produces l2 ordering w.r.t. some y that correspond to Pearson correlation.
       Z-Rank considers ordering based by monotonicity (close by l2 are close by monotonicity, high Spearman correlation)
    """
    
    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Normalizes term vectors and removes constant terms'''
        ranks = rank(vectors)  # get ranks
        normalized = super().normalize(ranks) 
        return normalized

class GaussRankNormalizer(ZRankNormalizer):
    """ A.k.a Van der Waerden transformation. Applies inv(Φ) (Probit) on top of ranks / (n+1).
        Even better than ZRan w.r.t. outliers in traces.
    """

    sqrt2 = 2.0 ** 0.5
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Normalizes term vectors and removes constant terms'''
        x_ranks = rank(vectors)
        x_scaled_ranks = (x_ranks + 1.0) / (vectors.shape[1] + 1.0)
        normalized = torch.erfinv(2.0 * x_scaled_ranks - 1.0) * self.sqrt2

        if self.target_normalized is None:
            return normalized
        correlations = torch.sum(normalized * self.target_normalized, dim=1)
        neg_corr_mask = correlations < 0
        normalized[neg_corr_mask] *= -1.0
        return normalized

class BaseVectorStorage(ServiceBase, Generic[TTermPos]):

    def __init__(self, *, 
                 term_order: Callable[[TTermPos], Any],
                #  normalizer: Normalizer[TTermPos],
                 add_metrics: Callable,
                 index: VectorStorage):
        self.term_order = term_order
        self.index = index
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
    
    def get_invalid_semantics(self, term: TTermPos) -> Optional[torch.Tensor]:
        vector = self.invalid_terms.get(term, None)
        return vector

    def get_semantics_for_term(self, term: TTermPos) -> Optional[torch.Tensor]:
        if term not in self.term_to_sid:
            return self.get_invalid_semantics(term)
        sid = self.term_to_sid[term]
        vector = self.index.get_vectors(sid)
        return vector
    
    # def get_new_normalized(self, terms: list[TTermPos], allow_nonrepr: bool = False) -> tuple[list[Term], torch.Tensor]:
    #     sids: list[int] = []
    #     normalized_terms: list[TTermPos] = []
    #     present_reprs = set()
    #     for term in terms:
    #         if term in self.term_to_sid:
    #             term_sid = self.term_to_sid[term]
    #             if allow_nonrepr:
    #                 repr_term = term
    #             else:
    #                 repr_term = self.sid_to_term[term_sid]
    #             # NOTE: (add ?0 1) has same semantics as (neg ?0) and as ?0. Holes are subjects to constraints 
    #             #        therefore, holes should not be filtered by repr (neg ?0) can be blocked
    #             if (repr_term == term) and (repr_term not in present_reprs):
    #                 # NOTE: only NEW and BEST representative is returned
    #                 present_reprs.add(repr_term)
    #                 normalized_terms.append(repr_term)
    #                 sids.append(term_sid)
    #     if len(sids) > 0:
    #         vectors = self.index.get_vectors(sids)
    #         return normalized_terms, vectors
    #     return [], None
    
    def get_terms_for_semantics(self, vectors: torch.Tensor) -> list[TTermPos | None]:
        sids = self.index.query_points(vectors)
        terms = [self.sid_to_term.get(sid, None) for sid in sids]
        return terms

    def filter_new(self, terms: list[TTermPos], vectors: torch.Tensor | None = None) -> list[TTermPos] | tuple[list[TTermPos], torch.Tensor | None]: 
        ''' Filters terms and semantics to leave only nonpresent yet terms '''
        new_term_ids = [i for i, term in enumerate(terms) if term not in self.term_to_sid and term not in self.invalid_terms]
        if len(new_term_ids) == 0:
            if vectors is not None:
                return [], None
            return []        
        if len(new_term_ids) < len(terms):
            terms = [terms[i] for i in new_term_ids]
            if vectors is not None:
                vectors = vectors[new_term_ids]
        if vectors is not None:
            return terms, vectors
        return terms
    
    def find_unique(self, terms: list[Term], vectors: torch.Tensor, return_type: Literal["ids", "entries"] = "ids") -> list[int] | tuple[list[Term], torch.Tensor]:
        unique_vectors, unique_ids = self.index.find_unique(vectors)
        groups = {}
        for i in range(len(terms)):
            unique_id = unique_ids[i] 
            prev_i = groups.get(unique_id, None)
            if (prev_i is None) or (self.term_order(terms[i]) < self.term_order(terms[prev_i])):
                groups[unique_id] = i
        found_sids = self.index.query_points(unique_vectors)
        found_reprs = [self.sid_to_term.get(sid, None) for sid in found_sids]
        filtered_ids = []
        for unique_id, i in groups.items():
            found_repr = found_reprs[unique_id]
            if (found_repr is None) or (self.term_order(terms[i]) < self.term_order(found_repr)):
                filtered_ids.append(i)
        if return_type == "ids":
            return filtered_ids
        if len(filtered_ids) == 0:
            return [], None
        if len(filtered_ids) < len(terms):
            terms = [terms[i] for i in filtered_ids]
            new_vectors = vectors[filtered_ids]
            del vectors
            vectors = new_vectors
        return terms, vectors

    def insert(self, terms: list[TTermPos], vectors: torch.Tensor) -> None:
        ''' Insert terms and their vectors into storage, outputs inserted terms '''
        # new_term_ids = [i for i, term in enumerate(terms) if term not in self.term_to_sid and term not in self.invalid_terms]
        # if len(new_term_ids) == 0:
        #     return []
        # require_cleanup = False
        # if len(new_term_ids) < len(terms):
        #     terms = [terms[i] for i in new_term_ids]
        #     vectors = vectors[new_term_ids]
        #     require_cleanup = True
        # mapped_vectors, terms = self.normalizer.normalize(vectors, terms)
        if len(terms) == 0: # all filtered
            return
        finite_all_mask = torch.isfinite(vectors)
        finite_mask = torch.all(finite_all_mask, dim=1)
        invalid_ids, = torch.where(~finite_mask)
        for iid in invalid_ids.tolist():
            self.invalid_terms[terms[iid]] = vectors[iid].clone()
        finite_ids, = torch.where(finite_mask)
        finite_terms = [terms[i] for i in finite_ids.tolist()]
        finite_mapped_vectors = vectors[finite_ids]
        del finite_all_mask, finite_mask, finite_ids, invalid_ids
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
    
    def get_repr_for_term(self, term: TTermPos) -> Optional[TTermPos]:
        if term not in self.term_to_sid:
            return None
        sid = self.term_to_sid[term]
        repr_term = self.sid_to_term.get(sid, None)
        return repr_term
        
    def get_semantics(self) -> torch.Tensor:
        all_semantics = self.index.get_vectors()
        return all_semantics

    def num_sem(self) -> int:
        return len(self.sid_to_term)
    
    def num_terms(self) -> int:
        return len(self.term_to_sid)

    def query_closest(self, query: torch.Tensor, params: QueryClosestParams) -> QueryClosestResult:
        ''' Returns map: query id to found ids in index (list) '''
        
        delta = params.delta
        found_ids = []
        while params.num_steps > 0: # gradually increase the query range to capture specified number of points
            query_range = torch.stack([query - delta, query + delta], dim=0)
            existing_ids = self.index.query_range(query_range)
            found_ids = [fid for fid in existing_ids if fid not in params.exclude_ids]
            if len(found_ids) >= params.num_closest:
                break
            delta *= params.multiplier
            params.num_steps -= 1 
        if len(found_ids) == 0:
            return QueryClosestResult([], [], [], delta)
        vectors = self.index.get_vectors(found_ids)
        l2 = torch.sum((vectors - query.unsqueeze(0)) ** 2, dim=1)
        if params.num_closest == 1:
            min_l2_id = torch.argmin(l2)
            min_l2 = l2[min_l2_id].item()
            min_found_id = found_ids[min_l2_id.item()]
            # min_vector = vectors[min_l2_id].clone()
            min_term = self.sid_to_term[min_found_id]
            # query_res = [(l2[min_l2_id].item(), self.sid_to_term[found_ids[min_l2_id.item()]])]
            res =  QueryClosestResult([min_l2], [min_found_id], [min_term], delta)
            del l2, vectors
            return res 
        l2_sort_order = torch.argsort(l2)
        min_l2_ids = l2_sort_order[:params.num_closest]
        min_l2 = l2[min_l2_ids].tolist()
        min_found_ids = [found_ids[fid] for fid in min_l2_ids.tolist()]
        min_found_terms = [self.sid_to_term[fid] for fid in min_found_ids]
        res = QueryClosestResult(min_l2, min_found_ids, min_found_terms, delta)
        del l2, vectors, found_ids
        return res
    
    def closest_or_self(self, queries: torch.Tensor,
                            start_delta: float = 1e-2,
                            multiplier: float = 10,
                            max_num_steps: int = 3) -> torch.Tensor:
        ''' Returns closest vector or self '''
        
        closest = torch.clone(queries)
        for qid, q in enumerate(queries):
            delta = start_delta
            num_steps = max_num_steps
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

def hole_order(term_order: Callable):
    def fn(hole): 
        return (hole[1].at_depth, *term_order(hole[0]), hole[1].occur)
    return fn

class HoleVectorStorage(BaseVectorStorage[tuple[Term, TermPos]]):    
    def __init__(self, *, term_order, **kwargs):
        super().__init__(term_order=hole_order(term_order), **kwargs)