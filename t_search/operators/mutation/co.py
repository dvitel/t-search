
from typing import TYPE_CHECKING, Optional
from .base import TermMutation
from syntax import Term, Value
from .optimization import OptimState, optimize_consts

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class CO(TermMutation):
    ''' Const Optimization, Adjust consts to correspond to the given target. '''

    def __init__(self, name = "const_opt", *, 
                 frac = 0.2, 
                 num_vals: int = 1,
                 max_tries: int = 1,
                 num_evals: int = 10, lr = 1.0,
                 loss_threshold: Optional[float] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.frac = frac
        self.num_vals = num_vals
        self.max_tries = max_tries
        self.num_evals = num_evals
        self.lr = lr
        self.term_values_cache: dict[Term, list[Value]] = {}
        self.optim_term_cache: dict[Term, Term] = {}
        self.optim_state_cache: dict[Term, OptimState] = {}
        self.loss_threshold = loss_threshold

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        ''' Optimizes all constants inside the term '''
        
        term_loss, *_ = solver.eval(term, return_outputs="list").outputs
        
        optim_res = optimize_consts(term, term_loss, solver.fitness_fn, solver.builders,
                                    solver.ops, solver._get_binding,
                                    solver.const_range, 
                                    eval_fn = solver.eval_fn,
                                    num_vals = self.num_vals,
                                    max_tries=self.max_tries,
                                    max_evals=self.num_evals,
                                    lr = self.lr, loss_threshold = (solver.best_fitness if self.loss_threshold is None else self.loss_threshold),
                                    torch_gen=solver.torch_gen,
                                    term_values_cache=self.term_values_cache,
                                    optim_term_cache=self.optim_term_cache,
                                    optim_state_cache=self.optim_state_cache)

        if optim_res is not None:
            optim_state, num_evals, num_root_evals = optim_res
            solver.report_evals(num_evals, num_root_evals)                        

            # print(f"<<< {optim_res.optim_state.final_term} | {term_loss:.2f} --> {optim_res.optim_state.best_loss.item():.2f} >>>")
            if optim_state.best_loss is not None and (term_loss < optim_state.best_loss[0]):
                return None # can happen when we exhaust all attempts of optimization 
            return optim_state.best_term
        return None             
