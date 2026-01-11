''' Implementation of optimizer '''

import torch

from t_search.evaluators.evaluator import Evaluator
from t_search.syntax.syntax import Syntax
from .optimizer import Optimizer

from .optimization import OptimPoint, optimize
from t_search.syntax.term import Term, Value
    
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
                 ):
        self.num_starts = num_starts
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
        self.optim_term_cache: dict[Term, Term] = {}
        self.best_terms_cache: dict[Term, tuple[Term, float]] = {} # optim term to term
        self.default_loss_fn = evaluator.get_loss_fn()
        self.const_range = const_range

    def _get_optim_state(self, term: Term) -> Term | tuple[Term, dict[OptimPoint, torch.Tensor]]:
        ''' Either returns already optimized term or optimization state'''
        if term in self.optim_term_cache:
            return self.best_terms_cache[self.optim_term_cache[term]][0]
        optim_points: list[OptimPoint] = []
        binding = {}
        values = []        

        def const_to_optim_point(term, *_):
            if isinstance(term, Value):
                point_id = len(optim_points)
                point = OptimPoint(point_id)
                optim_points.append(point)

                del steps, rand_points
                binding[point] = term.value
                values.append(term)
                return point

        optim_term = self.syntax.replace_fn(term, const_to_optim_point)

        default_loss = self.default_loss_fn(term).item()

        self.optim_term_cache[term] = optim_term

        if len(binding) == 0:
            self.best_terms_cache[optim_term] = (term, default_loss) 
            return term    

        if optim_term in self.best_terms_cache: # already optimized
            cur_best_term, cur_best_loss = self.best_terms_cache[optim_term]
            if default_loss < cur_best_loss:
                self.best_terms_cache[optim_term] = (term, default_loss)
                return term
            else:
                return cur_best_term  

        self.best_terms_cache[optim_term] = (term, default_loss)          
        
        return (optim_term, binding)
    
    def optimize(self,
        term: Term,
    ) -> Term:
        """Searches for the term const values that would bring it closer to the target outputs.
        Restarts will reinitialize the constants.
        """

        optim_state = self._get_optim_state(term)
        if isinstance(optim_state, Term): # already optimized
            return optim_state # return best known term
        
        optim_term, start_binding = optim_state

        best_loss, best_binding = optimize(
            optim_term,
            self.const_range,
            start_binding,
            loss_fn_builder=self.evaluator.get_loss_fn,
            num_starts=self.num_starts,
            lr=self.lr,
            max_evals=self.max_evals,
            tolerance_change=self.tolerance_change,
            tolerance_grad=self.tolerance_grad,
            torch_gen=self.torch_gen,
        )

        if best_loss is not None:

            def bind_optim_points(term, **_):
                if isinstance(term, OptimPoint):
                    const_val = self.syntax.get_const(value=best_binding[term])
                    # if const_val is None:
                    #     print(f"Cannot create const term for value {best_binding[term]}")
                    #     raise ValueError(f"Cannot create const term for value {best_binding[term]}")
                    return const_val

            try:
                best_term = self.syntax.replace_fn(optim_state.optim_term, bind_optim_points)
            except ValueError as e:                
                return term
            self.best_terms_cache[optim_term] = (best_term, best_loss.item())
            del best_loss, best_binding
            return best_term            

        return term