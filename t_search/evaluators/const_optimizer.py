''' Implementation of optimizer '''

from typing import TYPE_CHECKING

import torch

from t_search.datasets.sampling import get_interval_grid
from .optimizer import Optimizer
from t_search.syntax.replacement import replace_fn

from .optimization import OptimPoint, OptimState, optimize
from t_search.syntax.term import Term, Value

if TYPE_CHECKING:
    from t_search.solver import GPSolver
    
class ConstOptimizer(Optimizer):

    def __init__(self, 
                    num_vals: int = 10,
                    max_evals: int = 20,
                    lr:float = 0.1,
                    tolerance_change: float = 1e-6,
                    tolerance_grad: float = 1e-3,
                 ):
        self.num_vals = num_vals
        self.max_evals = max_evals
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad
        self.lr = lr
        self.term_values_cache: dict[Term, list[Value]] = {}
        self.optim_term_cache: dict[Term, Term] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}

    def reset(self):
        self.term_values_cache = {}
        self.optim_term_cache = {}
        self.optim_state_cache = {}

    def get_optim_state(self, solver: 'GPSolver', term: Term, initial_term_loss: torch.Tensor | None = None) -> OptimState:
        if term not in self.optim_term_cache:  # need to build optim term with optim points

            optim_points: list[OptimPoint] = []
            binding = {}
            values = []

            def const_to_optim_point(term, *_):
                if isinstance(term, Value):
                    point_id = len(optim_points)
                    point = OptimPoint(point_id)
                    optim_points.append(point)
                    value = torch.zeros(
                        (self.num_vals, 1 if len(term.value.shape) == 0 else term.value.shape[0]),
                        dtype=term.value.dtype,
                        device=term.value.device,
                    )
                    binding[point] = value
                    values.append(term)
                    return point

            optim_term = replace_fn(term, const_to_optim_point, solver.builders)

            self.optim_term_cache[term] = optim_term
            self.term_values_cache[term] = values
            if optim_term not in self.optim_state_cache:
                if initial_term_loss is not None:
                    best_binding = dict(binding)
                    best_loss = initial_term_loss,
                    best_term = term
                optim_state = OptimState(optim_term, optim_points, binding, best_binding, best_loss, best_term,
                                            is_optimized=len(optim_points) == 0)
                self.optim_state_cache[optim_term] = optim_state
            else:
                optim_state = self.optim_state_cache[optim_term]
        else:
            optim_term = self.optim_term_cache[term]
            optim_state = self.optim_state_cache[optim_term]

        return optim_state

    def optimize(self,
        solver: 'GPSolver',
        term: Term,
        initial_term_loss: torch.Tensor | None = None,
    ) -> Term:
        """Searches for the term const values that would bring it closer to the target outputs.
        Restarts will reinitialize the constants.
        """

        optim_state = self.get_optim_state(solver, term, initial_term_loss = initial_term_loss)
        if optim_state.is_optimized:
            return optim_state.best_term

        starts_to_attempt = []

        rand_points_to_attempt = self.num_vals
        if optim_state.best_binding is not None:  # at first try we also optimize current values
            starts_to_attempt.append([optim_state.best_binding[op] for op in optim_state.optim_points])
            rand_points_to_attempt -= 1

        if rand_points_to_attempt > 0:  # we use grid sampling with rand shifts
            should_del_ranges = False
            if len(start_range.shape) == 1:  # 1d range
                should_del_ranges = True
                start_range = torch.tile(start_range, (len(optim_state.optim_points), 1))
            steps = (start_range[:, 1] - start_range[:, 0]) / (rand_points_to_attempt + 1)
            rand_points = get_interval_grid(steps, start_range, rand_deltas=True, generator=solver.torch_gen)
            if rand_points.shape[0] > rand_points_to_attempt:
                selected_ids = torch.randperm(rand_points.shape[0], device=rand_points.device, generator=solver.torch_gen)[
                    :rand_points_to_attempt
                ]
                new_rand_points = rand_points[selected_ids, :]
                del rand_points
                rand_points = new_rand_points
            starts_to_attempt.extend([[v for v in p] for p in rand_points])
            if should_del_ranges:
                del start_range

        const_vectors = []
        for point in optim_state.optim_points:
            const_values = torch.tensor(
                [[p[point.point_id]] for p in starts_to_attempt], device=solver.device, dtype=solver.dtype
            )
            const_vectors.append(const_values)

        for p, cv in zip(optim_state.optim_points, const_vectors):
            binding = optim_state.binding[p]
            binding.requires_grad = False
            binding.copy_(cv)  # copy new value to optim point
            binding.requires_grad = True

        num_evals, num_root_evals = optimize(
            optim_state,
            solver.evaluator.fitness_fn,
            solver.ops,
            solver.evaluator._get_binding,
            eval_fn=solver.evaluator.eval_fn,
            lr=self.lr,
            max_evals=self.max_evals,
            tolerance_change=self.tolerance_change,
            tolerance_grad=self.tolerance_grad
        )

        solver.evaluator.report_evals(num_evals, num_root_evals)

        if optim_state.best_loss is not None:

            def bind_optim_points(term, occur, **_):
                if isinstance(term, OptimPoint):
                    return solver.const_builder.fn(value=optim_state.best_binding[term])

            optim_state.best_term = replace_fn(optim_state.optim_term, bind_optim_points, solver.builders)

        return optim_state.best_term or term