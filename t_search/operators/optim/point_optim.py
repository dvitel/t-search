from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Literal

import numpy as np
import torch

from t_search.base import ServiceBase
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.optimization import OptimResult, clean_optim_result, get_all_grads, set_local_minimas_, get_slowest_funs, optimize, threshold_optim_result_, total_threshold_optim_result_
from t_search.evaluators.semantics import Semantics
from t_search.operators.mutation import PositionMutation
from t_search.operators.optim.term_hole import HoleFilling, TermHolePairs

from t_search.syntax import Term, TermPos
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import Op, OptimPoint, Value

@dataclass(order=True)
class PrioritizedTermPos:
    priority: Any 
    pos: TermPos = field(compare=False)

@dataclass(frozen=False)
class OptimState:
    optim_term: Term # term with OptimPoint
    # optim_position: TermPos
    term: Term 
    position: TermPos
    # path: list[PathNode] # maps new pos to old pos, these points are also collected in optimization
    # tabu_markers: set[Term] # set of skeletons that represent the optimization path
    binding: dict[OptimPoint, torch.Tensor]
    # ranges: torch.Tensor
    # const_binding: dict[OptimPoint, torch.Tensor] = field(default_factory=dict)

@dataclass(frozen=True)
class Hole:    
    start_loss: float # of parent term
    term: Term # parent 
    position: TermPos # where is the hole
    bindings: list[torch.Tensor] # possible bindings for the hole
            
class PointOptim(PositionMutation, ServiceBase):
    ''' Position Optimization, adjust selected position with optimizer ''' 
    
    def __init__(self, *, 
                 var_bindings: dict[str, torch.Tensor],
                 term_hole_pairs: TermHolePairs,
                 target: torch.Tensor,
                 fitness: Fitness,
                 semantics: Semantics,
                 evaluator: Evaluator,
                 torch_gen: torch.Generator,
                 get_cur_gen: Callable,
                 position_strategy: Literal["rand_position_order", "shallow_to_deep_position_order", "best_grad_position_order"] = "rand_position_order",
                 num_starts: int = 10,
                 range_delta: float = 0.1,
                 max_evals: int = 20,
                 lr:float = 0.1,
                 tolerance_change: float = 1e-6,
                 tolerance_grad: float = 1e-3,
                 min_loss_rtol: float = 1e-1,
                 with_tabu: bool = True,
                 max_hole_bindings: int = 1, # one hole can have multiple good bindings
                 num_children: int = 1000,
                 **kwargs):
        super().__init__(**kwargs, rate=None)
        self.term_hole_pairs = term_hole_pairs
        self.target = target
        self.evaluator = evaluator
        self.fitness = fitness
        self.semantics = semantics
        self.var_bindings = var_bindings
        self.with_tabu = with_tabu
        self.num_children = num_children
        self.position_strategy = getattr(self, position_strategy)  
        # self.term_position_orders: dict[Term, deque] = {}
        # self.tabu_positions: dict[Term, set[TermPos]] = {} # any position below the tabu position should be ignored
        self.tabu_set: set[Term] = set()

        self.torch_gen = torch_gen
        self.num_starts = num_starts
        self.range_delta = range_delta
        self.tried_optim_terms: set[Term] = set()
        self.tried_optim_terms_hit: int = 0
        self.lr = lr
        self.max_evals = max_evals
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad
        self.min_loss_rtol = min_loss_rtol
        self.default_loss_fn = evaluator.get_loss_fn()
        self.max_hole_bindings = max_hole_bindings

        # self.pos_queue: list[HolePos] = [] # hole priority queue
        # self.added_terms = set() # terms with added positions
        self.get_cur_gen = get_cur_gen
        min_y = torch.min(self.target) - self.range_delta
        max_y = torch.max(self.target) + self.range_delta
        self.ranges = torch.zeros((self.target.shape[0], 2), dtype=self.target.dtype, device=self.target.device)
        self.ranges[:, 0] = min_y
        self.ranges[:, 1] = max_y
        self.optim_point = OptimPoint(0)
        self.lineage: dict[Term, HoleFilling] = {} # child to parent map for backtracking
        self.term_positions: dict[Term, list[PrioritizedTermPos]] = {} # cached position priorities for terms

        #TODO
        # 1. Metrics - DONE
        # 2. num_holes for one term - not just 1 - DONE
        # 2.2 num_bindings - DONE
        # 3. Const check for optimized hole. - DONE
        # 4. testing on simple term

        #metrics 
        self.num_better_fills = 0 
        self.num_total_fills = 0
        self.num_holes_created = 0
        self.num_terms_optimized = 0

    def rand_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        ''' Returns list of priorities for positions - (rand,) '''
        priorities = [(self.rnd.random(),) for _ in positions]
        return priorities

    def shallow_to_deep_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        priorities = [(self.get_cur_gen(), p.at_depth, self.rnd.random(), ) for  p in positions]
        return priorities

    def best_grad_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        grads = get_all_grads(term, var_bindings=self.var_bindings, 
                      get_loss_fn=self.evaluator.get_loss_fn,
                      dtype=self.target.dtype, device=self.target.device)
        priorities = [ (-grads[(pos.term, pos.occur)].item(), age, ) for age, pos in enumerate(positions)]
        return priorities
    
    def optim_term_for_consts(self, term: Term) -> tuple[Term, dict[OptimPoint, torch.Tensor]]:
        const_optim_points: list[OptimPoint] = []
        const_binding = {}     

        # NOTE: taken from const_optimizer - should it be a separate routine?
        def const_to_optim_point(term, *_):
            if isinstance(term, Value):
                point_id = len(const_optim_points)
                point = OptimPoint(1 + point_id)
                const_optim_points.append(point)

                const_binding[point] = term.value
                return point

        optim_term = self.syntax.replace_fn(term, const_to_optim_point)      
        return (optim_term, const_binding)
    
    def bind_consts(self, term: Term, binding: dict[OptimPoint, torch.Tensor]) -> Term:
        def optim_point_to_const(term, *_):
            if term in binding:
                new_value = self.syntax.get_const(value=binding[term])
                return new_value

        optim_term = self.syntax.replace_fn(term, optim_point_to_const)      
        return optim_term

    def is_lincomb(self, pos: TermPos) -> bool:
        if pos.parent is None:
            return False
        term = pos.parent.term
        if isinstance(term, Op) and term.op_id in ["add", "mul"]:
            other_arg = term.get_args()[1 - pos.pos]
            if isinstance(other_arg, Value):
                return True 
        return False

    # TODO X1: optim_term --> optim_term_skeleton ?? Or optimize consts and pos at same time?? -- MANY (add OptimPoint <some_const>) - more complex function of consts could have different vectros for constant 
    #          DONE with skipping lincomb. Attempted optimization of consts with point became complicated and unstable - reverted back to point optim then const optim.
    # TODO X2: order of positions to optimize - should we try best term first??? experiment with position grad and random.
    # TODO X3: when increasing num_starts for pos optimizer it seems that results degrade -> maybe should take mean of loss?? - not sure why. 
    #          DONE - switched to mean
    # TODO X4: random exploration operator of term shapes. 
    # TODO X5: gathering correlation data for a good predictor (l2 to loss) of good pair and of good position
    #           DONE 1: fixed zscore standardization 
    # TODO X6: fetching terms for filling - should we fetch further terms when search is exhausted? Collect data on l2 with different normalization 
    # TODO X7; should we register complex terms in term index? Should we resort only to unique terms without constants. 
    # TODO X8: theoretical guarantees (visiting all search space positions/completeness/soundness?/l2 vs loss correlation)

    # TODO X9: backtracking through lineage (add term to parent map and provide parent position order)
    #          DONE 
    # TODO X10: loss_threshold - consider radius vector of CM CTS.
    #           DONE
    # TODO X11: const_optimizer one start with identities (1 and 0)
    #           DONE
    # TODO X12: bringing delayed fillings in the overrided __call__ in addtion to super().__call__
    # TODO X13; __call__ pick only n best HoleFillings out of produced.
    def _get_optim_state(self, term: Term, position: TermPos) -> OptimState | None:
        ''' None is returned if term,position is already optimized '''

        while self.is_lincomb(position):
            position = position.parent

        if position.parent is None:
            return None # cannot optimize root
        
        optim_term = self.syntax.replace_position(term, position, self.optim_point, with_validation=False)

        # optim_term, const_binding = self.optim_term_for_consts(orig_optim_term)

        if optim_term in self.tried_optim_terms:
            self.tried_optim_terms_hit += 1
            # if self.debug:
            #     print(f"Skipped tried: {optim_term} for {term}@({position.term},{position.occur})")            
            return None
        self.tried_optim_terms.add(optim_term)

        pos_outputs = self.semantics.get_outputs(position.term)

        binding = { self.optim_point: pos_outputs }

        # range_mins = torch.minimum(pos_outputs, self.target)
        # range_maxs = torch.maximum(pos_outputs, self.target)
        # range_mins -= self.range_delta
        # range_maxs += self.range_delta        
        # ranges = torch.stack([range_mins, range_maxs], dim=0).t()

        optim_term_positions = self.syntax.get_positions(optim_term)
        optim_position = next(p for p in optim_term_positions if p.term == self.optim_point)
        # path: list[PathNode] = [] # excludes optim point and root
        # cur_pos_path = get_path(optim_position, with_current=False, with_root=False)
        # path = [PathNode(p) for p in cur_pos_path]

        parent_optim_terms = self.syntax.replace_path_unvalidated(optim_term, optim_position.parent, lambda *_: self.optim_point)
        if len(parent_optim_terms) > 0:
            parent_optim_terms.pop() # remove root OptimPoint

        if self.is_in_tabu(optim_term, parent_optim_terms):
            if self.debug:
                print(f"Skipped tabu: {optim_term} for {term}@({position.term},{position.occur})")
            return None

        # tabu_markers = set()
        # if self.with_tabu: # creating tabu markers from path
        #     tabu_markers.add(optim_term)
        #     parent_optim_points = [self.optim_point for _ in path]
        #     if len(parent_optim_points) > 0:
        #         parent_optim_terms = self.syntax.replace_path_unvalidated(optim_term, optim_position.parent, parent_optim_points)
        #         for path_node, tabu_marker in zip(path, parent_optim_terms):
        #             path_node.tabu_marker = tabu_marker
        #         tabu_markers.update(parent_optim_terms)

        optim_state = OptimState(optim_term, term, position, binding)

        return optim_state
    
    def is_in_tabu(self, optim_term: Term, parent_optim_terms) -> bool:
        if optim_term in self.tabu_set or any(t in self.tabu_set for t in parent_optim_terms):
            return True
        return False
    
    def add_to_tabu(self, optim_state: OptimState) -> None:
        if self.with_tabu:
            if self.debug:
                print(f"Tabu: {optim_state.optim_term}")
            self.tabu_set.add(optim_state.optim_term)

    def optimize_state(self, optim_state: OptimState) -> Hole | None:
        ''' Takes one hole at a time, None if no holes left '''

        # if optim_state is None: # already optimized 
        #     return []
        
        # if self.is_in_tabu(optim_state):
        #     if self.debug:
        #         print(f"Skipped tabu: {optim_state.optim_term} for {optim_state.term}@({optim_state.position.term},{optim_state.position.occur})")
        #     return []
        
        # pos_to_collect = [(p.optim_term_pos.term, p.optim_term_pos.occur) 
        #                   for p in optim_state.path 
        #                   if p.tabu_marker not in self.tried_optim_terms and not self.is_lincomb(p.optim_term_pos)]
        
        if self.debug: 
            print(f"Optim: {optim_state.optim_term}")
        
        optim_result: OptimResult = optimize(optim_state.optim_term, 
                                self.ranges, 
                                optim_state.binding,
                                loss_fn_builder=self.evaluator.get_loss_fn,
                                # pos_to_collect=pos_to_collect,
                                num_starts=self.num_starts,
                                lr=self.lr,
                                max_evals=self.max_evals,
                                tolerance_change=self.tolerance_change,
                                tolerance_grad=self.tolerance_grad,
                                torch_gen=self.torch_gen)
        
        # if self.debug:
        #     best_optim_result = get_best_optim_result(optim_result)
        #     min_loss = ' '.join([f'{f:.0e}' for f in best_optim_result.loss[0].tolist()])
        #     print(f"Loss: {min_loss}")
        #     point = next(iter(best_optim_result.binding.values()))[0]
        #     point_trace = ' '.join([f'{f:+5.2f}' for f in point.tolist()])
        #     print(f"Trac: {point_trace}")
        #     # NOTE: next is for manual checking of optimization correctness
        #     # for x, v in self.var_bindings.items():
        #     #     x_trace = ' '.join([f'{f:.1e}' for f in v.tolist()])
        #     #     print(f"{x:4}: {x_trace}")
        #     # target_trace = ' '.join([f'{f:.1e}' for f in self.target.tolist()])
        #     # print(f"Trgt: {target_trace}")

        self.num_terms_optimized += 1

        start_loss = self.fitness.get_fitness(optim_state.term).item()

        # if optim_result is None:
        #     self.add_to_tabu(optim_state)
        #     return None
                    

        total_threshold_optim_result_(optim_result, start_loss)

        if torch.any(torch.all(torch.isinf(optim_result.loss), dim=0)): # no minimas found
            self.add_to_tabu(optim_state)
            return None

        # holes: list[Hole] = []
            
        set_local_minimas_(optim_result)

        slowest_traces = get_slowest_funs(optim_result, max_num_funs=self.max_hole_bindings)

        clean_optim_result(optim_result)

        # if slowest_traces is None:
        #     self.add_to_tabu(optim_state)
        #     continue
        
        slowest_traces_binding = [t.clone() for traces in slowest_traces.binding.values() for t in traces] 

        hole = Hole(start_loss, optim_state.term, optim_state.position, slowest_traces_binding)
        # holes.append(hole)
        # if len(slowest_traces.additional_binding) > 0:
        #     term_path = get_path(optim_state.position, with_current=False, with_root=False)
        #     assert len(optim_state.path) == len(term_path)
        #     for path_node, path_pos in zip(optim_state.path, term_path):
        #         key = (path_node.optim_term_pos.term, path_node.optim_term_pos.occur)
        #         if key not in slowest_traces.additional_binding:
        #             continue
        #         traces = [t.clone() for t in slowest_traces.additional_binding[key]]
        #         new_hole = Hole(optim_state.term, path_pos, traces)
        #         holes.append(new_hole)

        clean_optim_result(slowest_traces)

        # if self.debug:
        #     for i, hole in enumerate(holes):
        #         print(f"Hole {i}: {hole.term} at ({hole.position.occur}, {hole.position.term})")
        #         for j, hb in enumerate(hole.bindings):
        #             hb_trace = ' '.join([f'{f:.1e}' for f in hb.tolist()])
        #             print(f"----> {hb_trace}")
        #     pass

        return hole
       
    def select_positions(self, term: Term) -> Generator[OptimState, None, None]:   

        if not self.semantics.is_valid(term): # do not optimize invalid terms
            return

        if term not in self.term_positions:
            positions = self.syntax.get_positions(term)
            priorities = self.position_strategy(term, positions) # should be cached if necessary
            ppositions = [PrioritizedTermPos(p, pos) for p, pos in zip(priorities, positions) if p is not None]
            ppositions.sort(key=lambda x: x.priority, reverse=True)
            self.term_positions[term] = ppositions
            
            # sorted_positions = [p for _, p in sorted(zip(priorities, positions), key=lambda x: x[0])]
            
        ppositions = self.term_positions[term]

        while len(ppositions) > 0:
            pos = ppositions.pop().pos
            optim_state = self._get_optim_state(term, pos)
            if optim_state is None: # already optimized 
                continue

            # hole_pos = HolePos(priority, term, pos)
            yield optim_state
            # heappush(self.pos_queue, hole_pos)

        if term in self.lineage: # we finished with term positions, backtrack to parent 
            parent_filling = self.lineage[term]
            yield from self.select_positions(parent_filling.hole_root)

        # self.added_terms.add(term)
        return

    def mutate_position(self, term: Term, optim_state: OptimState) -> list[HoleFilling]:
        
        bound_hole = self.optimize_state(optim_state)

        if bound_hole is None:
            return []

        holes = []
        hole_bindings = []
        for hole_binding in bound_hole.bindings:
            holes.append((bound_hole.term, bound_hole.position))
            hole_bindings.append(hole_binding)
        
            self.num_holes_created += len(holes)
        
        hole_position_tensor = torch.stack(hole_bindings)
        fillings = self.term_hole_pairs.register_holes(holes, hole_position_tensor)
        del hole_position_tensor
        for hb in hole_bindings:
            del hb 

        if len(fillings) == 0: # try next position
            return None
        
        return fillings 
    
    def __call__(self, population):
        delayed_fillings = self.term_hole_pairs.register_delayed_terms()
        new_fillings = super().__call__(population)
        new_fillings = [f for f in new_fillings if isinstance(f, HoleFilling)]
        children_fillings = new_fillings + delayed_fillings
        children_fillings.sort(key=lambda f: f.priority)
        if len(children_fillings) > self.num_children:
            # pick best num_children fillings
            children_fillings = children_fillings[:self.num_children]
            pass            
        children = []
        for f in children_fillings:
            if f.term in self.lineage:
                prev_filling = self.lineage[f.term]
                if prev_filling.priority > f.priority:
                    self.lineage[f.term] = f
            else:
                self.lineage[f.term] = f # hole_root is parent term
            if self.debug:
                print(f)
            children.append(f.term)        
        return children
    
    def get_finalizer(self):
        self.add_metrics(
            num_better_fills=self.num_better_fills,
            num_total_fills=self.num_total_fills,
            num_holes_created=self.num_holes_created,
            num_terms_optimized=self.num_terms_optimized,
            tried_optim_terms=len(self.tried_optim_terms),
            tried_optim_terms_hit=self.tried_optim_terms_hit,
            # "num_terms_created": self.num_terms_created,
            # tabu_positions=sum(len(v) for v in self.tabu_positions.values())
            tabu_positions=len(self.tabu_set)
            )
        