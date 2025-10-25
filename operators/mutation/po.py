

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Literal, Optional

import torch

from ..base import TermsListener
from .base import PositionMutation
from .reduce import Reduce
from spatial import VectorStorage
from term import Term, TermPos, Value
from torch_alg import OptimPoint, OptimState, get_pos_optim_state, optimize_positions
from util import stack_rows, stack_rows_2d

if TYPE_CHECKING:
    from gp import GPSolver


@dataclass 
class OptimizedPos:
    term_pos: list[TermPos]
    cur_id: int

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
            
class PO(PositionMutation, TermsListener):
    ''' Position Optimization, adjust selected position with optimizer ''' 
    
    def __init__(self, name = "point_opt", *, num_vals: int = 1, max_tries: int = 1,
                 num_evals: int = 10, lr = 1.0, delta: float = 0.1,
                 num_best: int = 5,
                 loss_threshold: Optional[float] = None,
                 sem_atol: float = 1e-5,
                 collect_inner_binding: bool = True,
                 index_type = VectorStorage,
                 normalize_semantics: bool = True,
                 syn_simplify: Optional[Reduce] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.num_vals = num_vals
        self.max_tries = max_tries
        self.num_evals = num_evals
        self.lr = lr
        self.delta = delta
        self.sem_atol = sem_atol
        self.num_best = num_best
        self.tries_pos: dict[Term, OptimizedPos] = {}
        self.optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term | None] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}
        self.optim_point_pos_cache: dict[Term, TermPos] = {}
        self.loss_threshold = loss_threshold
        self.collect_inner_binding = collect_inner_binding

        self.index_type = index_type
        self.term_index: VectorStorage | None = None         
        self.hole_index: VectorStorage | None = None
        self.term_semantics: dict[Term, TermSemantics] = {}
        self.semantic_terms: dict[int, TermSemantics] = {}
        self.semantic_holes: dict[int, dict[tuple[Term, Term, int, int], HoleSemantics]] = {} 
        self.zero: torch.Tensor | None = None
        self.one: torch.Tensor | None = None
        self.normalize_semantics = normalize_semantics
        self.syn_simplify = syn_simplify

    def op_init(self, solver: 'GPSolver'):
        if self.term_index is not None:
            del self.term_index
        self.term_index: VectorStorage = \
            self.index_type(capacity = solver.max_evals // 2, dims = solver.target.shape[0], 
                dtype = solver.dtype, device = solver.device,
                rtol = 0, atol = self.sem_atol)
        if self.hole_index is not None:
            del self.hole_index
        self.hole_index: VectorStorage = \
            self.index_type(capacity = solver.max_evals // 2, dims = solver.target.shape[0], 
                dtype = solver.dtype, device = solver.device,
                rtol = 0, atol = self.sem_atol)
        
        self.term_semantics: dict[Term, TermSemantics] = {}
        self.semantic_terms: dict[int, TermSemantics] = {}
        self.semantic_holes: dict[int, dict[tuple[Term, Term, int, int], HoleSemantics]] = {} 
        self.zero = torch.zeros((1,), dtype = solver.dtype, device = solver.device)
        self.one = torch.ones((1,), dtype = solver.dtype, device = solver.device)

        if self.normalize_semantics:
            if "add" not in solver.ops or "mul" not in solver.ops or solver.max_consts == 0:
                print(f"Warning: normalization was disabled as there are no operations (add, mul) or consts to revert it")
                self.normalize_semantics = False # normalization requires add, mul in solver.ops
        
        # if solver.max_consts > 0 and self.normalize_semantics:
        if self.normalize_semantics:
            zero_ids = self.term_index.insert(torch.zeros_like(solver.target).unsqueeze(0))
            zero_id = zero_ids[0]
            zero_const = Value(self.zero)
            zero_semantics = TermSemantics(term=zero_const, sid=zero_id, std=self.zero, mean=self.zero)
            self.term_semantics[zero_const] = zero_semantics
            self.semantic_terms[zero_id] = zero_semantics

    def select_positions(self, solver: 'GPSolver', term: Term) -> Generator[TermPos]:
        if term not in self.tries_pos:
            positions = solver.get_positions(term)
            positions = [pos for pos in positions if pos not in solver.invalid_term_outputs]
            # DECISION 1: how to pick term pos?? shuffle, sorted by depth?
            positions.sort(key=lambda pos: pos.at_depth) # start with shallowest positions
            # positions = solver.rnd.permutation(positions)
            self.tries_pos[term] = OptimizedPos(term_pos=positions, cur_id=0)

        term_pos = self.tries_pos[term]
        end_id = term_pos.cur_id - 1 
        if end_id < 0:
            end_id = len(term_pos.term_pos) - 1
        cur_pos = None
        optim_state = None
        # pos_to_remove = set()
        while term_pos.cur_id <= end_id:
            cur_pos = term_pos.term_pos[term_pos.cur_id % len(term_pos.term_pos)]
            term_pos.cur_id += 1
            optim_state = get_pos_optim_state(term, (cur_pos,), 
                                optim_term_cache = self.optim_term_cache, 
                                optim_state_cache = self.optim_state_cache,
                                builders = solver.builders,
                                num_vals = self.num_vals,
                                output_size = solver.target.shape[0],
                                max_tries = self.max_tries,
                                dtype = solver.dtype, device = solver.device)
            # DECISION 2: how many optim attempts? what starting point to take?
            if optim_state is None:
                # pos_to_remove.add((cur_pos.term, cur_pos.occur))
                continue
            if optim_state.max_tries <= 0:
                optim_state = None
                # point was already optimized and is in the hole index
                continue # try next pos // or should we try next term??? return None
                # return None ???
            break 

        # if len(pos_to_remove) > 0:
        #     term_pos.term_pos = [pos for pos in term_pos.term_pos if (pos.term, pos.occur) not in pos_to_remove]
        #     if len(term_pos.term_pos) == 0:
        #         del self.tries_pos[term]
        #         return None 
        #     term_pos.cur_id = term_pos.cur_id % len(term_pos.term_pos)
        return cur_pos, optim_state

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        
        optim_term = self.optim_term_cache.get((term, position))
        optim_state = self.optim_state_cache.get(optim_term)
        if optim_state is None:
            return None
        
        pos_output, *_ = solver.eval(position.term, return_outputs="list").outputs
        output_range = stack_rows([pos_output, solver.target], target_size=solver.target.shape[0])
        range_mins = torch.minimum(output_range[0], output_range[1])
        range_maxs = torch.maximum(output_range[0], output_range[1])
        output_range[0] = range_mins - self.delta
        output_range[1] = range_maxs + self.delta

        num_evals, num_root_evals = \
            optimize_positions(optim_state, solver.fitness_fn,
                solver.ops, solver._get_binding,
                output_range, 
                solver.eval_fn,
                pos_outputs=(pos_output,),
                num_vals = self.num_vals,
                max_evals=self.num_evals,
                num_best = self.num_best,
                lr = self.lr, loss_threshold = (solver.best_fitness if self.loss_threshold is None else self.loss_threshold),
                collect_inner_binding = self.collect_inner_binding,
                torch_gen=solver.torch_gen)
        
        solver.report_evals(num_evals, num_root_evals)
        if optim_state.best_loss is None: 
            return None    
        # good semantics to add to the hole index

        holes_w_semantics: list[tuple[Term, TermPos, torch.Tensor]] = []

        if self.collect_inner_binding:

            if optim_state.optim_term not in self.optim_point_pos_cache:
                optim_term_poss = solver.get_positions(optim_state.optim_term)
                optim_point_pos = next(pos for pos in optim_term_poss if isinstance(pos.term, OptimPoint))
                self.optim_point_pos_cache[optim_state.optim_term] = optim_point_pos

            optim_point_pos = self.optim_point_pos_cache[optim_state.optim_term]
            
            # now we have pos (in term) and optim_point_pos (in optim_term)
            # we can build chains in both terms to the root 

            cur_pos = position
            cur_optim_pos = optim_point_pos
            while cur_pos.term != term:
                cur_binding = optim_state.best_binding[cur_optim_pos.term]
                holes_w_semantics.append((term, cur_pos, cur_binding))
                cur_pos = cur_pos.parent
                cur_optim_pos = cur_optim_pos.parent
        else: # we collected only point binding 
            cur_binding = optim_state.best_binding[optim_state.optim_points[0]]
            holes_w_semantics.append((term, position, cur_binding))

        new_terms = self.register_holes(solver, holes_w_semantics)
        
        return new_terms 

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
            new_term = self.syn_simplify.mutate_term(solver, new_term, [], [])
        return new_term
        
    def register_terms(self, solver: 'GPSolver', terms: list[Term], semantics: torch.Tensor) -> list[Term]: 
        if len(terms) == 0:
            return []
        if self.normalize_semantics:
            means = torch.mean(semantics, dim=1, keepdim=False)
            stds = torch.std(semantics, dim=1, keepdim=False)
            const_mask = torch.isclose(stds, self.zero, rtol=0, atol=1e-2)
            nonconst_mask = ~const_mask
            nonconst_ids, = torch.where(nonconst_mask)
            if nonconst_ids.numel() == 0:
                return
            final_means = means[nonconst_ids]
            final_stds = stds[nonconst_ids]
            normalized_semantics = (semantics[nonconst_ids] - final_means.unsqueeze(-1)) / final_stds.unsqueeze(-1)
            nonconst_terms: list[Term] = [terms[i] for i in nonconst_ids.tolist()]
        else:
            normalized_semantics = semantics
            nonconst_terms = terms
            final_means = [self.zero] * len(terms)
            final_stds = [self.one] * len(terms)
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
        semantics = stack_rows_2d([s for _, _, s in holes], target_size=solver.target.shape[0])

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