from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch

from t_search.syntax import Term, TermPos, Value
from t_search.spatial import VectorStorage

from .base import Listener

if TYPE_CHECKING:    
    from ..mutation import Reduce
    from t_search.solver import GPSolver

@dataclass 
class TermSemantics:
    term: Term # term or sketch (w OptimPoint)
    sid: int # id in spatial index 
    # normalization: semantics(Term) = std * index(sid) + mean 
    std: torch.Tensor # scaling coefficient 
    mean: torch.Tensor # shift coefficient

@dataclass 
class HoleSemantics:    
    root_term: Term
    pos: TermPos
    sid: int 
    std: torch.Tensor
    mean: torch.Tensor

def _normalize_filter(terms: list[Term], semantics: torch.Tensor, zero: torch.Tensor):
    ''' Remove constant terms, returns for each term (normalized, mean, std) '''
    means = torch.mean(semantics, dim=1, keepdim=False)
    stds = torch.std(semantics, dim=1, keepdim=False)
    const_mask = torch.isclose(stds, zero, rtol=0, atol=1e-2)
    nonconst_mask = ~const_mask
    nonconst_ids, = torch.where(nonconst_mask)
    if nonconst_ids.numel() == 0:
        return []
    final_means = means[nonconst_ids]
    final_stds = stds[nonconst_ids]
    normalized_semantics = (semantics[nonconst_ids] - final_means.unsqueeze(-1)) / final_stds.unsqueeze(-1)
    nonconst_terms: list[Term] = [terms[i] for i in nonconst_ids.tolist()]
    return (nonconst_terms, normalized_semantics, final_means, final_stds)

class TermSketchSearch(Listener):
    ''' For new terms search for sketches, for new sketches search for terms.  '''

    def __init__(self, name="TermSketchSearch", *, 
                    index_type = None, # type: ignore
                    syn_simplify: 'Reduce | None' = None,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.zero = torch.tensor(0.0)
        self.one = torch.tensor(1.0)  
        self.index_type = index_type

        self.term_index: VectorStorage | None = None         
        self.term_semantics: dict[Term, TermSemantics] = {}
        self.semantic_terms: dict[int, TermSemantics] = {}
        
        self.hole_index: VectorStorage | None = None
        self.semantic_holes: dict[int, dict[tuple[Term, Term, int, int], HoleSemantics]] = {} 
        self.syn_simplify = syn_simplify

    def on_eval(self, solver, terms, semantics, fitness):
        return self.register_terms(solver, terms, semantics)    
    
    def on_start(self, solver):
        super().on_start(solver)
        self.zero = torch.zeros((1,), dtype = solver.dtype, device = solver.device)
        self.one = torch.ones((1,), dtype = solver.dtype, device = solver.device)    

        # TODO: cleanup 
        
        if self.term_index is None: 
            self.term_index: VectorStorage = self.index_type(
                capacity = solver.max_evals // 2, 
                dims = solver.target.shape[0], 
                dtype = solver.dtype, device = solver.device
            )
        else:
            self.term_index.reset()
        self.term_semantics.clear()
        self.semantic_terms.clear()        
        
        if self.hole_index is None:
            self.hole_index: VectorStorage = self.index_type(
                capacity = solver.max_evals // 2, 
                dims = solver.target.shape[0], 
                dtype = solver.dtype, device = solver.device
            )
        else:
            self.hole_index.reset()
        self.semantic_holes.clear()

        self.one = torch.ones((1,), dtype = solver.dtype, device = solver.device)

        # if self.normalize_semantics:
        #     if "add" not in solver.ops or "mul" not in solver.ops or solver.max_consts == 0:
        #         print(f"Warning: normalization was disabled as there are no operations (add, mul) or consts to revert it")
        #         self.normalize_semantics = False # normalization requires add, mul in solver.ops
        
        # if solver.max_consts > 0 and self.normalize_semantics:
        # if self.normalize_semantics:
        self.zero = torch.zeros((1,), dtype = solver.dtype, device = solver.device)
        zero_id, *_ = self.term_index.insert(torch.zeros_like(solver.target).unsqueeze(0))
        zero_const = solver.const_builder.fn(value = self.zero)
        zero_semantics = TermSemantics(term=zero_const, sid=zero_id, std=self.zero, mean=self.zero)
        self.term_semantics[zero_const] = zero_semantics
        self.semantic_terms[zero_id] = zero_semantics


    def register_terms(self, solver: 'GPSolver', terms: list[Term], semantics: torch.Tensor) -> list[Term]: 
        if len(terms) == 0:
            return []
        
        # if self.normalize_semantics:
        #     means = torch.mean(semantics, dim=1, keepdim=False)
        #     stds = torch.std(semantics, dim=1, keepdim=False)
        #     const_mask = torch.isclose(stds, self.zero, rtol=0, atol=1e-2)
        #     nonconst_mask = ~const_mask
        #     nonconst_ids, = torch.where(nonconst_mask)
        #     if nonconst_ids.numel() == 0:
        #         return []
        #     final_means = means[nonconst_ids]
        #     final_stds = stds[nonconst_ids]
        #     normalized_semantics = (semantics[nonconst_ids] - final_means.unsqueeze(-1)) / final_stds.unsqueeze(-1)
        #     nonconst_terms: list[Term] = [terms[i] for i in nonconst_ids.tolist()]
        # else:
        #     normalized_semantics = semantics
        #     nonconst_terms = terms
        #     final_means = [self.zero] * len(terms)
        #     final_stds = [self.one] * len(terms)

        nonconst_terms, normalized_semantics, final_means, final_stds = _normalize_filter(terms, semantics, self.zero) 
        semantic_ids = self.term_index.insert(normalized_semantics)
        for term, semantic_id, mean, std in zip(nonconst_terms, semantic_ids, final_means, final_stds):
            term_semantics = TermSemantics(term=term, sid=semantic_id, std=std, mean=mean)
            self.term_semantics[term] = term_semantics
            if semantic_id in self.semantic_terms: # pick smallest term as representative
                cur_t = self.semantic_terms[semantic_id].term
                cur_t_depth = solver.get_depth(cur_t)
                t_depth = solver.get_depth(term)
                if t_depth < cur_t_depth:
                    self.semantic_terms[semantic_id] = term_semantics
            else:
                self.semantic_terms[semantic_id] = term_semantics

        # searching for nearby holes 
        found_hole_ids = self._query_index(self.hole_index, normalized_semantics)
        closest_pairs = [(hole_sem, self.semantic_terms[semantic_ids[qid]]) 
                        for qid, hids in found_hole_ids.items()
                        for hid in hids 
                        for hole_sem in self.semantic_holes.get(hid, {}).values()]
        new_terms = []
        present_terms = set()
        for hole_sem, term_sem in closest_pairs:
            new_term = self.fill_hole(solver, hole_sem, term_sem)
            if new_term is not None and new_term not in present_terms:
                present_terms.add(new_term)
                new_terms.append(new_term)
        return new_terms

    def register_holes(self, solver: 'GPSolver', holes: list[tuple[Term, TermPos, torch.Tensor]]) -> list[Term]:
        ''' Adds hole and its semantics to index and outputs currently present fillings '''
        if len(holes) == 0:
            return []
        semantics = solver.stack_rows([s for _, _, s in holes])

        if self.normalize_semantics:
            means = torch.mean(semantics, dim=1, keepdim=True)
            stds = torch.std(semantics, dim=1, keepdim=True)
            const_mask = torch.all(torch.isclose(semantics, means, rtol=1e-2, atol=1e-2), dim=-1)
            # nonconst_mask = ~const_mask
            # nonconst_ids, = torch.where(nonconst_mask)
            normalized_semantics = (semantics - means) / stds
            normalized_semantics[const_mask] = self.zero
            stds[const_mask] = self.zero

            # nonconst_terms: list[Term] = [terms[i] for i in nonconst_ids.tolist()]
            # semantic_ids = self.term_index.insert(normalized_semantics)
        else:
            normalized_semantics = semantics
            means = [self.zero] * semantics.shape[0]
            stds = [self.one] * semantics.shape[0]

        all_hole_ids = self.hole_index.insert(normalized_semantics)

        cur_start = 0
        hole_semantics_map = {}
        for (root_term, hole_pos, hs) in holes:
            hole_query_ids = list(range(cur_start, cur_start + hs.shape[0]))
            cur_start += hs.shape[0]
            for qid in hole_query_ids:
                hole_sem_id = all_hole_ids[qid]
                mean = means[qid]
                std = stds[qid]
                hole_semantics = HoleSemantics(root_term=root_term, pos=hole_pos, sid=hole_sem_id, std=std, mean=mean)
                hole_semantics_map[qid] = hole_semantics
                sem_sketches = self.semantic_holes.setdefault(hole_sem_id, {})
                sem_sketches[(root_term, hole_pos.term, hole_pos.occur, hole_sem_id)] = hole_semantics

        query_ids = self._query_index(self.term_index, normalized_semantics)

        cur_start = 0
        new_terms = []
        present_terms = set()
        for (root_term, hole_pos, hole_semantics) in holes:
            hole_query_ids = list(range(cur_start, cur_start + hole_semantics.shape[0]))
            cur_start += hole_semantics.shape[0]
            present_tuples = set()
            for qid in hole_query_ids:
                hole_semantics = hole_semantics_map[qid]
                for term_sid in query_ids.get(qid, []):
                    term_semantics = self.semantic_terms[term_sid]
                    tuple_key = (hole_semantics.root_term, hole_semantics.pos.term, hole_semantics.pos.occur, term_semantics.term)
                    if tuple_key in present_tuples:
                        continue
                    present_tuples.add(tuple_key)
                    new_term = self.fill_hole(solver, hole_semantics, term_semantics)
                    if new_term is not None and new_term not in present_terms:
                        present_terms.add(new_term)
                        new_terms.append(new_term)
        del normalized_semantics
        return new_terms            
    
    def fill_hole(self, solver: 'GPSolver', 
                    hole_semantics: HoleSemantics, term_semantics: TermSemantics) -> Optional[Term]:
        if hole_semantics.pos.term == term_semantics.term:
            return None 
        

        if self.normalize_semantics:

            # denormalize --> though we have much by normalized semantics, mean and std is different 
            #  we create a term that would match hole_semantics 
            # t* = hs * (t - tm) / ts + hm = (hs / ts) * t + hm - (hs / ts) * tm
            # new_term = k * term_semantics.term + b 
            #       where k = hole_semantics.std / term_semantics.std 
            #             b = hole_semantics.mean - hole_semantics.std / term_semantics.std * term_semantics.mean

            term_std_zero = torch.isclose(term_semantics.std, self.zero, rtol=0, atol=1e-2)
            hole_std_zero = torch.isclose(hole_semantics.std, self.zero, rtol=0, atol=1e-2)
            if term_std_zero and hole_std_zero: # const adjustment hs / ts = 0 / 0 = 1
                k = self.one #torch.ones_like(hole_semantics.std)
                b = hole_semantics.mean - term_semantics.mean
            elif term_std_zero:
                return None # cannot adjust const to hole 
            else:
                k = hole_semantics.std / term_semantics.std
                b = hole_semantics.mean - (hole_semantics.std / term_semantics.std) * term_semantics.mean
            k_is_one = torch.isclose(k, self.one, rtol=0, atol=1e-2)
            b_is_zero = torch.isclose(b, self.zero, rtol=0, atol=1e-2)
            if k_is_one and b_is_zero:
                hole_term = term_semantics.term
            elif k_is_one:
                # NOTE: only one constant is allowed in term_index - Value(0)
                if isinstance(term_semantics.term, Value):
                    hole_term = Value(b)
                else:
                    hole_term = solver.op_builders["add"].fn(term_semantics.term, solver.const_builder.fn(value = b))
            elif b_is_zero:
                hole_term = solver.op_builders["mul"].fn(solver.const_builder.fn(value = k), term_semantics.term)
            else:
                if isinstance(term_semantics.term, Value):
                    hole_term = Value(b)
                else:
                    hole_term = solver.op_builders["add"].fn(
                                    solver.op_builders["mul"].fn(solver.const_builder.fn(value = k), term_semantics.term),
                                    solver.const_builder.fn(value = b))
        else:
            hole_term = term_semantics.term
        
        new_term = solver.replace_position(hole_semantics.root_term, hole_semantics.pos, hole_term)
        
        if new_term is not None and self.syn_simplify is not None:
            new_term = self.syn_simplify.mutate_term(solver, new_term)
        return new_term
    
    def _query_index(self, idx: VectorStorage, 
                            query: torch.Tensor,
                            qtype: Literal["point", "range"] = "point",
                            deltas = [0.001, 0.01, 0.1],) -> dict[int, list[int]]:
        ''' Either point query or more complelx iterative range query 
            Returns map: query id to found ids in index (list)
        '''
        
        if qtype == "point":
            found_ids = idx.query_points(query, rtol=0, atol=1e-1) #atol or self.atol, rtol=rtol or self.rtol)
            res = {qid:[v] for qid, v in enumerate(found_ids) if v >= 0}
            return res 
        
        # qtype == "range":

        # const_val = self.find_any_const(query)
        # if const_val is not None:
        #     return [Value(const_val)]
    
        res = {}
        for delta in deltas:
            for qid, q in enumerate(query):
                range = torch.stack([q - delta, q + delta], dim=0)
                found_ids = idx.query_range(range)
                if len(found_ids) > 0:
                    res[qid] = found_ids
            if len(res) > 0:
                break
            
        return res    