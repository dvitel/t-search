from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Callable, Generator, Literal, Sequence

import numpy as np
import torch

from t_search.base import ServiceBase
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.optimization import OptimPoint, OptimResult, clean_optim_result, get_all_grads, set_local_minimas_, get_slowest_funs, optimize, threshold_optim_result_
from t_search.evaluators.semantics import Semantics
from t_search.operators.operator import Operator
from t_search.operators.optim.term_hole import TermHolePairs

from t_search.syntax import Term, TermPos
from t_search.syntax.flow import shuffled_position_flow
from t_search.syntax.syntax import Syntax

@dataclass(frozen=False)
class OptimState:
    optim_term: Term # term with OptimPoint
    path: dict[tuple[Term, int], TermPos] # maps new pos to old pos, these points are also collected in optimization
    binding: dict[OptimPoint, torch.Tensor]
    ranges: torch.Tensor

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
                 position_strategy: Literal["rand_position_order", "shallow_to_deep_position_order", "best_grad_position_order"] = "rand_position_order",
                 num_starts: int = 10,
                 range_delta: float = 0.1,
                 max_evals: int = 20,
                 lr:float = 0.1,
                 tolerance_change: float = 1e-6,
                 tolerance_grad: float = 1e-3,
                 min_loss_rtol: float = 1e-1,
                 with_tabu: bool = True,
                 closer_to_points: bool = False,
                 closer_to_points_lambda: float = 1e-2,
                 max_hole_bindings: int = 1, # one hole can have multiple good bindings
                 num_children: int = 1000,
                 hole_batch_size: int = 16,
                 debug: bool = False,
                 loss_threshold: float = 1e-3,
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
        self.tabu_positions: dict[Term, set[TermPos]] = {} # any position below the tabu position should be ignored
        self.rnd = rnd
        self.torch_gen = torch_gen
        self.num_starts = num_starts
        self.range_delta = range_delta
        self.optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term] = {}
        self.tried_optim_terms: set[Term] = set()
        self.lr = lr
        self.max_evals = max_evals
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad
        self.min_loss_rtol = min_loss_rtol
        self.default_loss_fn = evaluator.get_loss_fn()
        self.closer_to_points = closer_to_points
        self.closer_to_points_lambda = closer_to_points_lambda
        self.add_metrics = add_metrics
        self.max_hole_bindings = max_hole_bindings
        self.hole_batch_size = hole_batch_size
        self.loss_threshold = loss_threshold

        self.pos_queue: list[HolePos] = [] # hole priority queue
        self.added_terms = set() # terms with added positions

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
        priorities = [(p.at_depth, age, self.rnd.random(), ) for age, p in enumerate(positions)]
        return priorities

    def best_grad_position_order(self, term: Term, positions: list[TermPos]) -> list[Any]:
        grads = get_all_grads(term, var_bindings=self.var_bindings, 
                      get_loss_fn=self.evaluator.get_loss_fn,
                      dtype=self.target.dtype, device=self.target.device)
        priorities = [ (-grads[(pos.term, pos.occur)].item(), age, ) for age, pos in enumerate(positions)]
        return priorities

    def _get_optim_state(self, term: Term, position: TermPos) -> OptimState | None:
        ''' None is returned if term,position is already optimized '''
        optim_term = self.optim_term_cache.get((term, (position.term, position.occur)))
        if optim_term is not None:  # position is already optimized
            return None
        
        optim_point = OptimPoint(0)
        def pos_to_optim_point(term: Term, occur: int):
            if term == position.term and occur == position.occur:
                return optim_point
            return None
        optim_term = self.syntax.replace_fn(term, pos_to_optim_point)

        self.optim_term_cache[(term, (position.term, position.occur))] = optim_term
        if optim_term in self.tried_optim_terms:
            return None
        self.tried_optim_terms.add(optim_term)        

        pos_outputs = self.semantics.get_outputs(position.term)

        binding = { optim_point: pos_outputs }

        range_mins = torch.minimum(pos_outputs, self.target)
        range_maxs = torch.maximum(pos_outputs, self.target)
        range_mins -= self.range_delta
        range_maxs += self.range_delta        
        ranges = torch.stack([range_mins, range_maxs], dim=0).t()

        optim_term_positions = self.syntax.get_positions(optim_term)
        optim_term_position = next(p for p in optim_term_positions if p.term == optim_point)
        path = {} # excludes optim point and root
        cur_pos = optim_term_position.parent
        cur_real_pos = position.parent
        while cur_pos.parent is not None:
            path[(cur_pos.term, cur_pos.occur)] = cur_real_pos
            cur_pos = cur_pos.parent
            cur_real_pos = cur_real_pos.parent

        optim_state = OptimState(optim_term, path, binding, ranges)

        return optim_state
    
    def local_improve(self, orig_term: Term, local_minimas: list[float]) -> list[int]:
        cur_loss = self.default_loss_fn(orig_term).item()
        where_improved = []
        for i, local_minima in enumerate(local_minimas):
            if (cur_loss - local_minima) > (self.min_loss_rtol * cur_loss):
                where_improved.append(i)
        return where_improved
    
    def global_improve(self, orig_term: Term, local_minimas: list[float]) -> list[int]:
        best_term = self.fitness.best_term
        if best_term is None:
            return True
        cur_loss = self.default_loss_fn(best_term).item()
        where_improved = []
        for i, local_minima in enumerate(local_minimas):
            if (cur_loss - local_minima) > (self.min_loss_rtol * cur_loss):
                where_improved.append(i)
        return where_improved
    
    def get_optim_loss_fn(self, **kwargs) -> callable:
        if self.closer_to_points: # add term that attracts the search to existing points
            base_loss_fn = self.evaluator.get_loss_fn(**kwargs)
            def optim_loss_fn(term: Term, *, binding) -> torch.Tensor:
                one_binding = next(iter(binding.values()))
                closest = self.term_hole_pairs.term_index.closest_or_self(one_binding)
                base_loss = base_loss_fn(term, binding=binding)
                point_loss = torch.sum((one_binding - closest) ** 2, dim=1)
                total_loss = base_loss + self.closer_to_points_lambda * point_loss
                return total_loss
            return optim_loss_fn
        return self.evaluator.get_loss_fn(**kwargs)
    
    def is_in_tabu(self, term: Term, position: TermPos) -> bool:
        if term not in self.tabu_positions:
            return False 
        
        term_tabu = self.tabu_positions[term]
        blocked = False
        cur_pos = position 
        while cur_pos is not None:
            if cur_pos in term_tabu:
                blocked = True
                break
            cur_pos = cur_pos.parent
        return blocked
    
    def add_to_tabu(self, term: Term, position: TermPos) -> None:
        if self.with_tabu:
            if term not in self.tabu_positions:
                self.tabu_positions[term] = set()
            self.tabu_positions[term].add(position)
        pass

    def create_holes(self, hole_pos: HolePos) -> list[Hole]:
        ''' Takes one hole at a time, None if no holes left '''

        # tabu list check
        if self.is_in_tabu(hole_pos.term, hole_pos.pos):
            return []
            
        optim_state = self._get_optim_state(hole_pos.term, hole_pos.pos)

        if optim_state is None: # already optimized 
            return []
        
        pos_to_collect = set(optim_state.path.keys())
        
        optim_result: OptimResult = optimize(optim_state.optim_term, 
                                optim_state.ranges, 
                                optim_state.binding,
                                loss_fn_builder=self.get_optim_loss_fn,
                                pos_to_collect=pos_to_collect,
                                num_starts=self.num_starts,
                                lr=self.lr,
                                max_evals=self.max_evals,
                                tolerance_change=self.tolerance_change,
                                tolerance_grad=self.tolerance_grad,
                                torch_gen=self.torch_gen,
                                num_local_minimas=self.max_hole_bindings,
                                debug=self.debug)
                    
        self.num_terms_optimized += 1

        threshold_optim_result_(optim_result, self.loss_threshold)

        if torch.any(torch.all(torch.isinf(optim_result.loss), dim=0)): # no minimas found
            self.add_to_tabu(hole_pos.term, hole_pos.pos)
            return []

        set_local_minimas_(optim_result)

        slowest_traces = get_slowest_funs(optim_result, max_num_funs=self.max_hole_bindings)

        clean_optim_result(optim_result)

        if slowest_traces is None:
            self.add_to_tabu(hole_pos.term, hole_pos.pos)
            return []
            
        slowest_traces_binding = [t.clone() for traces in slowest_traces.binding.values() for t in traces] 

        hole = Hole(hole_pos.term, hole_pos.pos, slowest_traces_binding)
        holes = [hole]
        for k, v in optim_state.path.items():
            traces = [t.clone() for t in slowest_traces.additional_binding[k]]
            new_hole = Hole(hole_pos.term, v, traces)
            holes.append(new_hole)

        clean_optim_result(slowest_traces)
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

    # TODO -1: bug with extracting local minimas - DONE
    # TODO 0: debug strange case of (add cos(x) (neg x)) --> (add cos(x) (mul 1 (neg x))) - why it had good fit?? - DONE (not reappearing)
    # TODO 1: tabu list as set of skeletons (optim_terms)
    # TODO 2: redo the loop by adding instant jump to children gen when good pair appears, 
    # TODO 3: do not use batch for holes, but control queues sizes !!!
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

        while len(children) < self.num_children:

            # NOTE: first we try to drain the queue of hole-term pairs and then create new holes
            #       use hole_batch_size to control how many holes to create in advance 
            child, pair = self.term_hole_pairs.get_best_hole_filling()
            if child is not None:
                # self.evaluator.eval(child)
                # new_fitness = self.fitness.get_fitness(child)
                # old_fitness = self.fitness.get_fitness(pair.hole[0])
                # # assert new_fitness < old_fitness, "Filling must improve fitness"  
                # # self.num_total_fills += 1
                # if new_fitness < old_fitness:
                #     self.num_better_fills += 1

                self.num_total_fills += 1
                children.append(child)
                continue

            holes = []
            hole_bindings = []
            while len(holes) < self.hole_batch_size and len(self.pos_queue) > 0:
                hole_pos = heappop(self.pos_queue)
                cur_holes = self.create_holes(hole_pos)
                for hole in cur_holes:
                    for hole_binding in hole.bindings:
                        holes.append((hole.term, hole.position))
                        hole_bindings.append(hole_binding)

            if len(holes) == 0:
                break

            self.num_holes_created += len(holes)
        
            hole_position_tensor = torch.stack(hole_bindings)
            self.term_hole_pairs.register_holes(holes, hole_position_tensor)
            del hole_position_tensor
            for hb in hole_bindings:
                del hb
            
            pass 

        return children    
    
    def get_finalizer(self):
        self.add_metrics(
            num_better_fills=self.num_better_fills,
            num_total_fills=self.num_total_fills,
            num_holes_created=self.num_holes_created,
            num_terms_optimized=self.num_terms_optimized,
            # "num_terms_created": self.num_terms_created,
            tabu_positions=sum(len(v) for v in self.tabu_positions.values()))
        