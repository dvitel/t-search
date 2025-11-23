
from typing import TYPE_CHECKING, Optional

from t_search.evaluators.optimizer import ConstOptimizer
from .base import TermMutation
from t_search.syntax import Term

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class CO(TermMutation):
    ''' Const Optimization, Adjust consts to correspond to the given target. '''

    def __init__(self, name = "const_opt", *, 
                 optimizer: ConstOptimizer,
                 frac = 0.2, **kwargs):
        super().__init__(name, **kwargs)
        self.frac = frac
        self.optimizer = optimizer

    def on_start(self, solver: 'GPSolver'):
        self.optimizer.reset()

    def on_end(self, solver: 'GPSolver'):
        self.optimizer.reset()

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:
        ''' Optimizes all constants inside the term '''
        
        term_loss, *_ = solver.eval(term, return_outputs="list").outputs
        
        new_term = self.optimizer.optimize(solver, term, initial_term_loss=term_loss)

        return new_term
