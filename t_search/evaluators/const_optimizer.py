''' Implementation of optimizer '''

import torch

from t_search.datasets.sampling import get_interval_grid
from t_search.evaluators.evaluator import Evaluator
from t_search.syntax.syntax import Syntax
from .optimizer import Optimizer

from .optimization import OptimPoint, OptimState, optimize
from t_search.syntax.term import Term, Value
    
class ConstOptimizer(Optimizer):

    def __init__(self, *,
                    
                    # from solver context
                    syntax: Syntax,
                    evaluator: Evaluator,

                    device: torch.device,
                    dtype: torch.dtype,
                    torch_gen: torch.Generator,
                    
                    # parameters from config
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
        # self.evaluator = evaluator
        self.syntax = syntax
        self.evaluator = evaluator
        self.torch_gen = torch_gen
        self.device = device
        self.dtype = dtype
        self.term_values_cache: dict[Term, list[Value]] = {}
        self.optim_term_cache: dict[Term, Term] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}

    def get_finalizer(self):
        self.term_values_cache = {}
        self.optim_term_cache = {}
        self.optim_state_cache = {}

    def _get_optim_state(self, term: Term, initial_term_loss: torch.Tensor | None = None) -> OptimState:
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
                        dtype=self.dtype,
                        device=self.device,
                    )
                    binding[point] = value
                    values.append(term)
                    return point

            optim_term = self.syntax.replace_fn(term, const_to_optim_point)

            self.optim_term_cache[term] = optim_term
            self.term_values_cache[term] = values
            if optim_term not in self.optim_state_cache:
                if initial_term_loss is not None:
                    best_binding = dict(binding)
                    best_loss = initial_term_loss,
                    best_term = term
                
                optim_state = OptimState(optim_term, optim_points, binding, best_binding, best_loss, best_term,
                                            is_optimized=len(optim_points) == 0)
                if not optim_state.is_optimized:
                    optim_state.loss_fn = self.evaluator.get_loss_fn(
                                get_binding = optim_state.get_binding)
                self.optim_state_cache[optim_term] = optim_state
            else:
                optim_state = self.optim_state_cache[optim_term]
        else:
            optim_term = self.optim_term_cache[term]
            optim_state = self.optim_state_cache[optim_term]

        return optim_state

    def optimize(self,
        term: Term,
    ) -> Term:
        """Searches for the term const values that would bring it closer to the target outputs.
        Restarts will reinitialize the constants.
        """

        initial_term_loss, *_ = self.evaluator.eval(term, return_fitness="list").fitness

        optim_state = self._get_optim_state(term, initial_term_loss = initial_term_loss)
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
            rand_points = get_interval_grid(steps, start_range, rand_deltas=True, generator=self.trnd)
            if rand_points.shape[0] > rand_points_to_attempt:
                selected_ids = torch.randperm(rand_points.shape[0], device=rand_points.device, generator=self.trnd)[
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
                [[p[point.point_id]] for p in starts_to_attempt], device=self.device, dtype=self.dtype
            )
            const_vectors.append(const_values)

        for p, cv in zip(optim_state.optim_points, const_vectors):
            binding = optim_state.binding[p]
            binding.requires_grad = False
            binding.copy_(cv)  # copy new value to optim point
            binding.requires_grad = True

        optimize(
            optim_state,
            lr=self.lr,
            max_evals=self.max_evals,
            tolerance_change=self.tolerance_change,
            tolerance_grad=self.tolerance_grad
        )

        if optim_state.best_loss is not None:

            def bind_optim_points(term, occur, **_):
                if isinstance(term, OptimPoint):
                    return self.syntax.get_const(value=optim_state.best_binding[term])

            optim_state.best_term = self.syntax.replace_fn(optim_state.optim_term, bind_optim_points)

        return optim_state.best_term or term