''' Implementation of optimizer '''

from dataclasses import dataclass
from functools import partial
import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.syntax.syntax import Syntax
from .optimizer import Optimizer

from .optimization import clean_optim_result, optimize_consts
from t_search.syntax.term import Op, OptimPoint, Term, Value

@dataclass(frozen=False)
class Optimized:
    term: Term
    loss: float | None
    optim_term: Term | None = None
    # start_loss: float | None = None
    
class ConstOptimizer(Optimizer):

    def __init__(self, *,
                    
                    # from solver context
                    syntax: Syntax,
                    evaluator: Evaluator,

                    device: torch.device,
                    dtype: torch.dtype,
                    torch_gen: torch.Generator,
                    const_range: torch.Tensor,
                    
                    # parameters from config
                    num_starts: int = 10,
                    max_evals: int = 20,
                    lr:float = 0.1,
                    tolerance_change: float = 1e-6,
                    tolerance_grad: float = 1e-3,
                    debug: bool = False,
                    loss_threshold: float = 0.01,
                 ):
        self.num_starts = num_starts
        self.max_evals = max_evals
        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad
        self.loss_threshold = loss_threshold
        self.lr = lr
        # self.evaluator = evaluator
        self.syntax = syntax
        self.evaluator = evaluator
        self.torch_gen = torch_gen
        self.device = device
        self.dtype = dtype
        self.optim_term_cache: dict[Term, Term] = {}
        self.best_terms_cache: dict[Term, Optimized] = {} # optim term to term
        # self.default_loss_fn = evaluator.get_loss_fn()
        self.const_range = const_range.unsqueeze(0)
        self.debug = debug

    def _get_optim_state(self, term: Term) -> Optimized | tuple[Term, dict[OptimPoint, torch.Tensor]]:
        ''' Either returns already optimized term or optimization state'''
        if term in self.optim_term_cache:
            # print(f"ConstOptimizer: cache hit for term {term}")
            return self.best_terms_cache[self.optim_term_cache[term]]
        optim_points: list[OptimPoint] = []
        binding = {}
        # values = []        

        def const_to_optim_point(term, _, parent):
            if isinstance(term, Value):
                point_id = len(optim_points)
                point = OptimPoint(point_id)
                optim_points.append(point)

                binding_values = [term.value]
                # adding identities 
                if isinstance(parent, Op):
                    if parent.op_id == "mul": 
                        binding_values.append(self.syntax.one_value.value)
                    elif parent.op_id == "add":
                        binding_values.append(self.syntax.zero_value.value)
                binding[point] = binding_values
                # values.append(term)
                return point

        optim_term = self.syntax.replace_fn(term, const_to_optim_point)

        # default_loss = self.default_loss_fn(term).item()

        self.optim_term_cache[term] = optim_term

        if optim_term in self.best_terms_cache:
            return self.best_terms_cache[optim_term]

        # self.best_terms_cache[optim_term] = (term, None)
        if len(binding) == 0: # nothing to optimize
            return Optimized(term, None, None)
        
        return (optim_term, binding)
    
    def optimize(self,
        term: Term,
        with_loss: bool = False,
        num_starts: int | None = None,
    ) -> Term | Optimized:
        """Searches for the term const values that would bring it closer to the target outputs.
        Restarts will reinitialize the constants.
        """

        optim_state = self._get_optim_state(term)
        if isinstance(optim_state, Optimized): # already optimized
            if with_loss:
                return optim_state
            return optim_state.term # return best known term
        
        optim_term, start_binding = optim_state

        optim_result = optimize_consts(
            optim_term,
            const_range=self.const_range,
            start_binding=start_binding,
            loss_fn_builder=partial(self.evaluator.get_loss_fn, with_mean_loss_logging=True),
            num_starts=num_starts or self.num_starts,
            lr=self.lr,
            max_evals=self.max_evals,
            tolerance_change=self.tolerance_change,
            tolerance_grad=self.tolerance_grad,
            torch_gen=self.torch_gen,
            loss_threshold=self.loss_threshold
        )

        # best_loss_id = torch.argmin(optim_result.loss)

        const_vals = []

        def bind_optim_points(term, *_):
            if isinstance(term, OptimPoint):
                const_val = self.syntax.get_const(value=optim_result.binding[term].item())
                const_vals.append(const_val)
                # if const_val is None:
                #     print(f"Cannot create const term for value {best_binding[term]}")
                #     raise ValueError(f"Cannot create const term for value {best_binding[term]}")
                return const_val

        # try:
        best_term = self.syntax.replace_fn(optim_term, bind_optim_points)
        # except ValueError as e:                
        #     return term

        best_loss = optim_result.loss.item()
        # start_loss = optim_result.start_loss.item()
        optimized = Optimized(best_term, best_loss, optim_term)

        self.best_terms_cache[optim_term] = optimized

        clean_optim_result(optim_result)

        if with_loss:
            return optimized

        # if return_consts:
        #     return (term, const_vals)
        return best_term