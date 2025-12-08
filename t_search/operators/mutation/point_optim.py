

from collections import deque
from typing import Generator, Optional

import torch

from t_search.listeners.term_sketch import TermSketchSearch

from .base import PositionMutation
from t_search.syntax import Term, TermPos
from ...evaluators.optimization import OptimPoint, OptimState, get_pos_optim_state, optimize_positions

# TODO: simplification: optimization of term only once - same in const optimization 
# TODO: loss_threshold - may we move it outside core optimization loop? 
# TODO: tabu list     
            
class PointOptim(PositionMutation):
    ''' Position Optimization, adjust selected position with optimizer ''' 
    
    def __init__(self, *, 
                 search: TermSketchSearch,
                 num_vals: int = 1,
                 num_evals: int = 10, lr = 1.0, delta: float = 0.1,
                 num_best: int = 5,
                 loss_threshold: Optional[float] = None,
                #  sem_atol: float = 1e-5,
                 collect_inner_binding: bool = True,
                #  index_type = VectorStorage,
                #  normalize_semantics: bool = True,
                #  syn_simplify: Optional[Reduce] = None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.search = search
        self.num_vals = num_vals
        self.num_evals = num_evals
        self.lr = lr
        self.delta = delta
        # self.sem_atol = sem_atol
        self.num_best = num_best
        self.term_positions: dict[Term, deque] = {}
        self.optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term | None] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}
        self.optim_point_pos_cache: dict[Term, TermPos] = {}
        self.loss_threshold = loss_threshold
        self.collect_inner_binding = collect_inner_binding

        # self.index_type = index_type
        # self.term_index: VectorStorage | None = None         
        # self.hole_index: VectorStorage | None = None
        # self.term_semantics: dict[Term, TermSemantics] = {}
        # self.semantic_terms: dict[int, TermSemantics] = {}
        # self.semantic_holes: dict[int, dict[tuple[Term, Term, int, int], HoleSemantics]] = {} 
        # self.zero: torch.Tensor | None = None
        # self.one: torch.Tensor | None = None
        # self.normalize_semantics = normalize_semantics
        # self.syn_simplify = syn_simplify

    # def op_init(self, solver: 'GPSolver'):
    #     if self.term_index is not None:
    #         del self.term_index
    #     self.term_index: VectorStorage = \
    #         self.index_type(capacity = solver.max_evals // 2, dims = solver.target.shape[0], 
    #             dtype = solver.dtype, device = solver.device,
    #             rtol = 0, atol = self.sem_atol)
    #     if self.hole_index is not None:
    #         del self.hole_index
    #     self.hole_index: VectorStorage = \
    #         self.index_type(capacity = solver.max_evals // 2, dims = solver.target.shape[0], 
    #             dtype = solver.dtype, device = solver.device,
    #             rtol = 0, atol = self.sem_atol)
        
    #     self.term_semantics: dict[Term, TermSemantics] = {}
    #     self.semantic_terms: dict[int, TermSemantics] = {}
    #     self.semantic_holes: dict[int, dict[tuple[Term, Term, int, int], HoleSemantics]] = {} 
    #     self.zero = torch.zeros((1,), dtype = solver.dtype, device = solver.device)
    #     self.one = torch.ones((1,), dtype = solver.dtype, device = solver.device)

    #     if self.normalize_semantics:
    #         if "add" not in solver.ops or "mul" not in solver.ops or solver.max_consts == 0:
    #             print(f"Warning: normalization was disabled as there are no operations (add, mul) or consts to revert it")
    #             self.normalize_semantics = False # normalization requires add, mul in solver.ops
        
    #     # if solver.max_consts > 0 and self.normalize_semantics:
    #     if self.normalize_semantics:
    #         zero_ids = self.term_index.insert(torch.zeros_like(solver.target).unsqueeze(0))
    #         zero_id = zero_ids[0]
    #         zero_const = Value(self.zero)
    #         zero_semantics = TermSemantics(term=zero_const, sid=zero_id, std=self.zero, mean=self.zero)
    #         self.term_semantics[zero_const] = zero_semantics
    #         self.semantic_terms[zero_id] = zero_semantics

    def select_positions(self, solver: 'GPSolver', term: Term) -> Generator[TermPos, None, None]:

        if term not in self.term_positions:
            positions = solver.get_positions(term)
            positions_at_first_depth = [pos for pos in positions if pos.at_depth == 1]
            # positions = [pos for pos in positions if pos not in solver.invalid_term_outputs]
            # # NOTE: positions are visited in depth order
            # positions.sort(key=lambda pos: pos.at_depth) # start with shallowest positions
            # # positions = solver.rnd.permutation(positions)
            self.term_positions[term] = deque(positions_at_first_depth)

        positions = self.term_positions[term]

        while len(positions) > 0:
            position: TermPos = positions.popleft()
            positions.extend(position.children)
            optim_state = get_pos_optim_state(term, (position,), 
                                optim_term_cache = self.optim_term_cache, 
                                optim_state_cache = self.optim_state_cache,
                                builders = solver.builders,
                                num_vals = self.num_vals,
                                output_size = solver.target.shape[0],
                                dtype = solver.dtype, device = solver.device)
            if optim_state is None:
                continue
            yield position

        pass

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        
        optim_term = self.optim_term_cache.get((term, position))
        optim_state = self.optim_state_cache.get(optim_term)
        if optim_state is None:
            return None
        
        pos_output, *_ = solver.eval(position.term, return_outputs="list").outputs
        output_range = solver.stack_rows([pos_output, solver.target])
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