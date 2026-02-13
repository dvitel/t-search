

from t_search.base import ServiceBase
from t_search.evaluators.const_optimizer import ConstOptimizer
from t_search.operators.mutation import TermMutation
from t_search.syntax.term import Term


class ConstOptimMutation(TermMutation, ServiceBase):

    def __init__(self, 
                 const_optimizer: ConstOptimizer,
                 **kwargs):
        super().__init__(**kwargs)
        self.const_optimizer = const_optimizer

    def mutate_term(self, term: Term) -> Term | None:

        term = self.const_optimizer.optimize(term)        

        return term