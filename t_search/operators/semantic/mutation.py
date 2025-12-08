from t_search.evaluators.evaluator import Evaluator
from t_search.operators.classic.rpm import RPM
from t_search.syntax import Term, TermPos
from t_search.evaluators.fitness import l2

class SemanticallyDrivenMutation(RPM):
    ''' Semantically Driven Mutation Beadle and Johnson (2008, 2009b) '''
    def __init__(self, *,
                    evaluator: Evaluator,
                    min_d = 1e-1, max_d=1e+2, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.min_d = min_d
        self.max_d = max_d

    def mutate_position(self, term: Term, position: TermPos) -> Term | None:
        mutated_term = super().mutate_position(term, position)
        if mutated_term is None: 
            return None
        
        # check semantic difference
        parent_sem, mutated_term_sem, *_ = self.evaluator.eval([term, mutated_term], return_outputs="list").outputs
        dist = l2(parent_sem, mutated_term_sem)
        if dist < self.min_d or dist > self.max_d:
            return None        

        return mutated_term 
