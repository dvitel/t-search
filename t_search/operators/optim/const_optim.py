

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
                 n_best_terms: int | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.fitness = fitness
        self.evaluator = evaluator
        self.const_optimizer = const_optimizer
        self.n_best_terms = n_best_terms    
        self.preselected: set[Term] = set()

    def mutate_term(self, term: Term) -> Term | None:

        if self.n_best_terms is not None and term not in self.preselected:
            return None
        new_term = self.const_optimizer.optimize(term)        
        return new_term
    
    def add_to_lineage(self, parent, child):
        ''' Here child is optimized version of parent --> we actually pick parent of parent and treated it as parent for child'''        
        grandparents = self.term_lineage.get(parent, [])
        if len(grandparents) == 0:
            # if self.has_lineage_loop(child, parent):
            #     return # noop            
            super().add_to_lineage(parent, child)
        else:
            for gp in grandparents:
                # if self.has_lineage_loop(child, gp):
                #     return # noop                
                super().add_to_lineage(gp, child)
    
    def __call__(self, population):
        if self.n_best_terms is not None:
            self.evaluator.eval(population)
            if len(population) <= self.n_best_terms:
                self.preselected = set(population)
            else:
                fitness = self.fitness.get_fitness(population, return_type="tensor")
                best_indices = fitness.topk(self.n_best_terms, largest=False).indices
                self.preselected = set(population[i] for i in best_indices)
                del fitness, best_indices
        children = super().__call__(population)
        return children