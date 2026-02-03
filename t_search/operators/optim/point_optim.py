from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Callable, Generator, Literal, Sequence

import numpy as np
import torch

from t_search.base import ServiceBase
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.optimization import OptimResult, clean_optim_result, get_all_grads, get_best_optim_result, set_local_minimas_, get_slowest_funs, optimize, threshold_optim_result_
from t_search.evaluators.semantics import Semantics
from t_search.operators.operator import Operator
from t_search.operators.optim.term_hole import TermHolePairs

from t_search.syntax import Term, TermPos
from t_search.syntax.flow import shuffled_position_flow
from t_search.syntax.stats import get_path
from t_search.syntax.syntax import Syntax
from t_search.syntax.term import Op, OptimPoint, Value, Variable

@dataclass(frozen=False)    
class PathNode: 
    optim_term_pos: TermPos 
    tabu_marker: Term | None = None

@dataclass(frozen=False)
class OptimState:
    optim_term: Term # term with OptimPoint
    optim_position: TermPos
    term: Term 
    position: TermPos
    path: list[PathNode] # maps new pos to old pos, these points are also collected in optimization
    tabu_markers: set[Term] # set of skeletons that represent the optimization path
    binding: dict[OptimPoint, torch.Tensor]
    # ranges: torch.Tensor
    # const_binding: dict[OptimPoint, torch.Tensor] = field(default_factory=dict)

@dataclass(order=True)
class HolePos:
    priority: Any
    term: Term = field(compare=False)
    pos: TermPos = field(compare=False)

@dataclass(frozen=True)
class Hole:    
    term: Term # parent 
    position: TermPos # where is the hole
    bindings: list[torch.Tensor] # possible bindings for the hole
            
class PointOptim(Operator, ServiceBase):
    ''' Position Optimization, adjust selected position with optimizer ''' 
    
    def __init__(self, *, 
                 var_bindings: dict[str, torch.Tensor],
                 term_hole_pairs: TermHolePairs,
                 target: torch.Tensor,
                 syntax: Syntax,
                 fitness: Fitness,
                 semantics: Semantics,
                 evaluator: Evaluator,
                 rnd: np.random.Generator,
                 torch_gen: torch.Generator,
                 add_metrics: Callable,
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
                 debug: bool = False,
                 loss_threshold: float = 1e-3,
                 target_variance: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.term_hole_pairs = term_hole_pairs
        self.target = target
        self.syntax = syntax
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

        self.rnd = rnd
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
        self.add_metrics = add_metrics
        self.max_hole_bindings = max_hole_bindings
        self.loss_threshold = loss_threshold / target_variance

        self.pos_queue: list[HolePos] = [] # hole priority queue
        self.added_terms = set() # terms with added positions
        self.get_cur_gen = get_cur_gen
        min_y = torch.min(self.target) - self.range_delta
        max_y = torch.max(self.target) + self.range_delta
        self.ranges = torch.zeros((self.target.shape[0], 2), dtype=self.target.dtype, device=self.target.device)
        self.ranges[:, 0] = min_y
        self.ranges[:, 1] = max_y
        self.optim_point = OptimPoint(0)

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
        self.debug = debug        

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
    #          DONE with ssikiping lincomb. Attempted optimization of consts with point became complicated and unstable - reverted back to point optim then const optim.
    # TODO X2: order of positions to optimize - should we try best term first??? experiment with position grad and random.
    # TODO X3: when increasing num_starts for pos optimizer it seems that results degrade -> maybe should take mean of loss?? - not sure why. 
    #          DONE - switched to mean
    # TODO X4: random exploration operator of term shapes. 
    # TODO X5: gathering correlation data for a good predictor (l2 to loss) of good pair and of good position
    # TODO X6: fetching terms for filling - should we fetch further terms when search is exhausted? Collect data on l2 with different normalization 
    # TODO X7; should we register complex terms in term index? Should we resort only to unique terms without constants. 
    # TODO X8: theoretical guarantees (visiting all search space positions/completeness/soundness?/l2 vs loss correlation)
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
            if self.debug:
                print(f"Skipped tried: {optim_term} for {term}@({position.term},{position.occur})")            
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
        path: list[PathNode] = [] # excludes optim point and root
        cur_pos_path = get_path(optim_position, with_current=False, with_root=False)
        path = [PathNode(p) for p in cur_pos_path]

        tabu_markers = set()
        if self.with_tabu: # creating tabu markers from path
            tabu_markers.add(optim_term)
            parent_optim_points = [self.optim_point for _ in path]
            if len(parent_optim_points) > 0:
                parent_optim_terms = self.syntax.replace_path_unvalidated(optim_term, optim_position.parent, parent_optim_points)
                for path_node, tabu_marker in zip(path, parent_optim_terms):
                    path_node.tabu_marker = tabu_marker
                tabu_markers.update(parent_optim_terms)

        optim_state = OptimState(optim_term, optim_position, term, position, path, tabu_markers, binding)

        return optim_state
    
    def is_in_tabu(self, optim_state: OptimState) -> bool:
        no_blocked = set.isdisjoint(optim_state.tabu_markers, self.tabu_set)
        return not no_blocked
    
    def add_to_tabu(self, optim_state: OptimState) -> None:
        if self.with_tabu:
            if self.debug:
                print(f"Tabu: {optim_state.optim_term}")
            self.tabu_set.add(optim_state.optim_term)

    def create_holes(self, term: Term, pos: TermPos) -> list[Hole]:
        ''' Takes one hole at a time, None if no holes left '''

        optim_state = self._get_optim_state(term, pos)

        if optim_state is None: # already optimized 
            return []
        
        if self.is_in_tabu(optim_state):
            if self.debug:
                print(f"Skipped tabu: {optim_state.optim_term} for {optim_state.term}@({optim_state.position.term},{optim_state.position.occur})")
            return []
        
        pos_to_collect = [(p.optim_term_pos.term, p.optim_term_pos.occur) 
                          for p in optim_state.path 
                          if p.tabu_marker not in self.tried_optim_terms and not self.is_lincomb(p.optim_term_pos)]
        
        if self.debug: 
            print(f"Optim: {optim_state.optim_term}")
        
        optim_result: OptimResult = optimize(optim_state.optim_term, 
                                self.ranges, 
                                optim_state.binding,
                                loss_fn_builder=self.evaluator.get_loss_fn,
                                pos_to_collect=pos_to_collect,
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

        threshold_optim_result_(optim_result, self.loss_threshold)

        if torch.any(torch.all(torch.isinf(optim_result.loss), dim=0)): # no minimas found
            self.add_to_tabu(optim_state)
            return []

        holes: list[Hole] = []
            
        set_local_minimas_(optim_result)

        slowest_traces = get_slowest_funs(optim_result, max_num_funs=self.max_hole_bindings)

        clean_optim_result(optim_result)

        # if slowest_traces is None:
        #     self.add_to_tabu(optim_state)
        #     continue
        
        slowest_traces_binding = [t.clone() for traces in slowest_traces.binding.values() for t in traces] 

        hole = Hole(optim_state.term, optim_state.position, slowest_traces_binding)
        holes.append(hole)
        if len(slowest_traces.additional_binding) > 0:
            term_path = get_path(optim_state.position, with_current=False, with_root=False)
            assert len(optim_state.path) == len(term_path)
            for path_node, path_pos in zip(optim_state.path, term_path):
                key = (path_node.optim_term_pos.term, path_node.optim_term_pos.occur)
                if key not in slowest_traces.additional_binding:
                    continue
                traces = [t.clone() for t in slowest_traces.additional_binding[key]]
                new_hole = Hole(optim_state.term, path_pos, traces)
                holes.append(new_hole)

        clean_optim_result(slowest_traces)

        # if self.debug:
        #     for i, hole in enumerate(holes):
        #         print(f"Hole {i}: {hole.term} at ({hole.position.occur}, {hole.position.term})")
        #         for j, hb in enumerate(hole.bindings):
        #             hb_trace = ' '.join([f'{f:.1e}' for f in hb.tolist()])
        #             print(f"----> {hb_trace}")
        #     pass

        return holes
    
    def add_hole_pos(self, term: Term) -> None:
        ''' Adds term positions into priority of holes to optimize '''

        if term in self.added_terms or not self.semantics.is_valid(term): # do not optimize invalid terms
            return

        positions = self.syntax.get_positions(term)
        
        priorities = self.position_strategy(term, positions)

        for priority, pos in zip(priorities, positions):
            hole_pos = HolePos(priority, term, pos)
            heappush(self.pos_queue, hole_pos)

        self.added_terms.add(term)
        return
    
    def has_pos_to_optimize(self) -> bool:
        return len(self.pos_queue) > 0
    
    # TODO: test concrete terms to see the trajectories - collect them for writing

    # TODO 0: debug strange case of (add cos(x) (neg x)) --> (add cos(x) (mul 1 (neg x))) - why it had good fit?? - DONE (not reappearing)
    # TODO 1: tabu list as set of skeletons (optim_terms) - DONE
    # TODO 2: redo the loop by adding instant jump to children gen when good pair appears,  - DONE
    # TODO 3: do not use batch for holes, but control queues sizes !!! - DONE
    
    # TODO 4: trace step by step execution of the optimizer when loss is inf - for small set of test - DONE
    # TODO 5: how optimizer works in constraints of number of constants? --> pick only optim point that would not ruin constant constraints?? - Not important - solved on hole filling - validation happens then
    # TODO 6: term normalization through semantics mapping??? - DONE (note: it does not always work andd simple axioms are not applied, but removes present introns)
    def __call__(self, population: Sequence[Term]) -> Sequence[Term]: 
        ''' 
            1. Optimize holes from population
            2. Pick best terms form term_hole_pairs
        '''

        self.cur_parents = population

        # population = population[0:2]        

        for parent in population:
            # parent_skeleton = self.syntax.get_skeleton(parent)
            self.add_hole_pos(parent)

        children = []

        if self.debug:
            print(f"- Par: {len(population)}, Pos: {len(self.pos_queue)}, Fil: {len(self.term_hole_pairs.hole_fillings)}, Tabu: {len(self.tabu_set)}, Tried: {len(self.tried_optim_terms)}, Hit: {self.tried_optim_terms_hit} -")
        pass

        while (len(children) < self.num_children) \
                and (self.term_hole_pairs.has_fillings() or self.has_pos_to_optimize()):

            child = self.term_hole_pairs.get_best_hole_filling(force_pick=not self.has_pos_to_optimize())
            if child is not None:
                self.num_total_fills += 1
                children.append(child)
                if child.priority < self.fitness.fitness_atol: # found solution - break 
                    if self.debug:
                        print(f"Filling at {child.id}")
                    break
                # self.tried_optim_terms.update(child.skeletons)
                continue

            # while len(holes) < self.hole_batch_size and len(self.pos_queue) > 0:
            cur_holes = []
            while self.has_pos_to_optimize() and (len(cur_holes) == 0):
                hole_pos = heappop(self.pos_queue)
                if self.debug:
                    print('---------------------------------')
                    print(f">>> [{hole_pos.priority}] {hole_pos.term} at ({hole_pos.pos.term}, {hole_pos.pos.occur})")
                cur_holes = self.create_holes(hole_pos.term, hole_pos.pos)

            if len(cur_holes) == 0: # all pos attempted 
                continue

            holes = []
            hole_bindings = []
            for hole in cur_holes:
                for hole_binding in hole.bindings:
                    holes.append((hole.term, hole.position))
                    hole_bindings.append(hole_binding)

            self.num_holes_created += len(holes)
        
            hole_position_tensor = torch.stack(hole_bindings)
            self.term_hole_pairs.register_holes(holes, hole_position_tensor)
            del hole_position_tensor
            for hb in hole_bindings:
                del hb
            
            pass 

        if self.debug: 
            print(f"= Ch: {len(children)}, Pos: {len(self.pos_queue)}, Fil: {len(self.term_hole_pairs.hole_fillings)}, Tabu: {len(self.tabu_set)}, Tried: {len(self.tried_optim_terms)}, Hit: {self.tried_optim_terms_hit} =")

        new_terms = [ch.term for ch in children]    
        return new_terms
    
    def get_finalizer(self):
        self.add_metrics(
            num_better_fills=self.num_better_fills,
            num_total_fills=self.num_total_fills,
            num_holes_created=self.num_holes_created,
            num_terms_optimized=self.num_terms_optimized,
            # "num_terms_created": self.num_terms_created,
            # tabu_positions=sum(len(v) for v in self.tabu_positions.values())
            tabu_positions=len(self.tabu_set)
            )
        