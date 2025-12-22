

from typing import Generator, Literal, Optional, Sequence

import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.evaluators.optimization import OptimPoint, get_all_grads, optimize
from t_search.evaluators.semantics import Semantics
from t_search.operators.operator import Operator
from t_search.operators.optim.term_hole import TermHolePairs

from t_search.syntax import Term, TermPos
from t_search.syntax.flow import shuffled_position_flow
from t_search.syntax.syntax import Syntax

# TODO: simplification: optimization of term only once - same in const optimization 
# TODO: loss_threshold - may we move it outside core optimization loop? 
# TODO: tabu list     
            
class PointOptim(Operator):
    ''' Position Optimization, adjust selected position with optimizer ''' 
    
    def __init__(self, *, 
                 var_bindings: dict[str, torch.Tensor],
                 term_hole_pairs: TermHolePairs,
                 target: torch.Tensor,
                 syntax: Syntax,
                 fitness: Fitness,
                 semantics: Semantics,
                 evaluator: Evaluator,
                 torch_gen: torch.Generator,
                 position_strategy: Literal["rand_position_order", "shallow_to_deep_position_order", "best_grad_position_order"] = "rand_position_order",
                 term_strategy: Literal["rand_term_order", "best_term_order", "age_term_order"] = "rand_term_order",
                 improve_strategy: Literal["local_improve", "global_improve"] = "local_improve",
                 num_starts: int = 10,
                 range_delta: float = 0.1,
                 max_evals: int = 20,
                 lr:float = 0.1,
                 tolerance_change: float = 1e-6,
                 tolerance_grad: float = 1e-3,
                 min_loss_rtol: float = 1e-1,
                 with_tabu: bool = True,
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
        self.position_strategy = getattr(self, position_strategy)  
        self.term_strategy = getattr(self, term_strategy)
        self.term_position_orders: dict[Term, Generator[TermPos, None, None]] = {}
        self.term_age: dict[Term, int] = {} # num of attempted optimizations    
        self.tabu_positions: dict[Term, set[TermPos]] = {} # any position below the tabu position should be ignored
        self.improve_strategy = getattr(self, improve_strategy)
        self.torch_gen = torch_gen
        self.num_starts = num_starts
        self.range_delta = range_delta
        self.optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term] = {}
        self.lr = lr
        self.max_evals = max_evals
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad
        self.min_loss_rtol = min_loss_rtol
        self.default_loss_fn = evaluator.get_loss_fn()

    def rand_position_order(self, term: Term) -> Optional[TermPos]:
        positions = self.syntax.get_positions(term)
        flow = shuffled_position_flow(positions, self.solver.torch_gen)
        return flow

    def shallow_to_deep_position_order(self, term: Term) -> Generator[TermPos, None, None]:
        term_positions = self.syntax.get_positions(term)
        ordered_positions = sorted(term_positions, key=lambda pos: pos.at_depth)
        for position in ordered_positions:
            yield position

    def best_grad_position_order(self, term: Term) -> Generator[TermPos, None, None]:
        term_positions = self.syntax.get_positions(term)
        # requires forward pass
        grads = get_all_grads(term, var_bindings=self.var_bindings, 
                      get_loss_fn=self.evaluator.get_loss_fn,
                      dtype=self.target.dtype, device=self.target.device)
        pos_grads = {pos:grads[(pos.term, pos.occur)].item() for pos in term_positions}
        ordered_positions = sorted(term_positions, key=lambda pos: pos_grads[pos], reverse=True) # highest grad first
        for position in ordered_positions:
            yield position

    def rand_term_order(self, population: list[Term]) -> Generator[Term, None, None]:
        size = len(population)
        permuted_term_ids = self.rnd.permutation(size) 
        for term_id in permuted_term_ids:
            yield population[term_id]

    def best_term_order(self, population: list[Term]) -> Generator[Term, None, None]:
        fitness = self.fitness.get_fitness(population, return_type="list")
        term_fitness = {term: fit.item() for term, fit in zip(population, fitness)}
        ordered_terms = sorted(population, key=lambda term: term_fitness[term])
        for term in ordered_terms:
            yield term

    def age_term_order(self, population: list[Term]) -> Generator[Term, None, None]:
        term_counts = {}
        for term in population:
            term_counts[term] = self.term_counts.get(term, 0) + 1
        ordered_terms = sorted(population, key=lambda term: self.term_ages.get(term, 0) + term_counts[term])
        for term in ordered_terms:
            yield term

    def select_terms(self, population):
        return self.term_strategy(population)

    def select_positions(self, term: Term) -> Generator[TermPos, None, None]:
        if term not in self.term_position_orders:
            self.term_position_orders[term] = self.position_strategy(term)
        order = self.term_position_orders[term]        
        if self.with_tabu:
            if term not in self.tabu_positions:
                return order
            term_tabu = self.tabu_positions[term]
            for pos in order:
                is_in_tabu = False
                cur_pos = pos 
                while cur_pos is not None:
                    if cur_pos in term_tabu:
                        is_in_tabu = True
                        break
                    cur_pos = cur_pos.parent
                if not is_in_tabu:
                    yield pos
        else:
            return order
        
    def get_next_position(self, term: Term) -> Optional[TermPos]:
        positions = self.select_positions(term)
        return next(positions, default=None)

    def _get_optim_state(self, term: Term, position: TermPos) -> tuple[Term, dict[OptimPoint, torch.Tensor], torch.Tensor] | None:
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

        pos_outputs = self.semantics.get_outputs(position.term)

        binding = { optim_point: pos_outputs }

        range_mins = torch.minimum(pos_outputs, self.target)
        range_maxs = torch.maximum(pos_outputs, self.target)
        range_mins -= self.range_delta
        range_maxs += self.range_delta        
        ranges = torch.stack([range_mins, range_maxs], dim=0).t()
        return optim_term, binding, ranges
    
    def local_improve(self, orig_term: Term, best_loss: torch.Tensor) -> bool:        
        cur_loss = self.default_loss_fn(orig_term)
        return (cur_loss - best_loss) > (self.min_loss_rtol * cur_loss)
    
    def global_improve(self, orig_term: Term, best_loss: torch.Tensor) -> bool:
        best_term = self.fitness.best_term
        if best_term is None:
            return True
        best_term_loss = self.default_loss_fn(best_term)
        return (best_term_loss - best_loss) > (self.min_loss_rtol * best_term_loss)

    def create_hole(self, term: Term) -> tuple[Term, TermPos, torch.Tensor] | None:
        ''' we optimize the term at position only once '''

        position = self.get_next_position(term)
        
        optim_state = self._get_optim_state(term, position)
        if optim_state is None: # already optimized 
            return None, None, None
        
        optim_term, start_binding, start_range = optim_state

        best_loss, best_binding = optimize(optim_term, start_range, start_binding,
                 loss_fn_builder=self.evaluator.get_loss_fn,
                 num_starts=self.num_starts,
                 lr=self.lr,
                 max_evals=self.max_evals,
                 tolerance_change=self.tolerance_change,
                 tolerance_grad=self.tolerance_grad,
                 torch_gen=self.torch_gen
                 )

        if best_loss is None or not self.improve_strategy(term, best_loss): # cannot optimize
            if self.with_tabu:
                if term not in self.tabu_positions:
                    self.tabu_positions[term] = set()
                self.tabu_positions[term].add(position)
            return None, None, None 
        
        point_best_binding = next(best_binding.values())
        
        # self.term_hole_pairs.register_holes([(term, position)], point_best_binding.unsqueeze(0))

        return (term, position, point_best_binding)
    

    def __call__(self, population: Sequence[Term]) -> Sequence[Term]: 
        ''' 
            1. Optimize holes from population
            2. Pick best terms form term_hole_pairs
        '''

        self.cur_parents = population

        holes = []
        hole_bindings = []
        for parent in population:
            term, position, binding = self.create_hole(parent)
            if term is not None:
                holes.append((term, position))
                hole_bindings.append(binding)

        if len(holes) > 0:
        
            hole_position_tensor = torch.stack(hole_bindings)
            self.term_hole_pairs.register_holes(holes, hole_position_tensor)

        children = self.term_hole_pairs.get_best_hole_fillings(max_fillings=len(population))

        return children    