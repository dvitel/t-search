from t_search.evaluators.optimizer import Optimizer
from t_search.operators.mutation import TermMutation
from t_search.syntax import Term

class OptimMutation(TermMutation):
    ''' Optimization, Adjust consts to correspond to the given target. '''

    def __init__(self, *, 
                 optimizer: Optimizer,
                 **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer

    def mutate_term(self, term: Term) -> Term | None:
        ''' Optimizes all constants inside the term '''
        
        new_term = self.optimizer.optimize(term)

        return new_term
