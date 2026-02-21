

from t_search.base import ServiceBase
from t_search.evaluators.const_optimizer import ConstOptimizer
from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.fitness import Fitness
from t_search.operators.mutation import TermMutation
from t_search.syntax.term import Term


class ConstOptimMutation(TermMutation, ServiceBase):

    def __init__(self, 
                 evaluator: Evaluator,
                 fitness: Fitness,
                 const_optimizer: ConstOptimizer,
                 loss_threshold: float | None = None, # start optimizing only when bellow this level
                 only_best_term: bool = False, # optimize only new best if it is given
                 **kwargs):
        super().__init__(**kwargs)
        self.fitness = fitness
        self.evaluator = evaluator
        self.const_optimizer = const_optimizer
        self.loss_threshold = loss_threshold
        self.only_best_term = only_best_term

    def mutate_term(self, term: Term) -> Term | None:

        if self.only_best_term and self.fitness.best_term != term:
            return term 

        if self.loss_threshold is None:
            new_term = self.const_optimizer.optimize(term)        
            return new_term
        self.evaluator.eval(term)
        fitness = self.fitness.get_fitness(term)
        if fitness < self.loss_threshold:
            new_term = self.const_optimizer.optimize(term)        
            return new_term
        return term