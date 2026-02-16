

from t_search.evaluators.evaluator import Evaluator
from t_search.evaluators.semantics import Semantics
from t_search.operators.classic.rpx import RPX
from t_search.syntax import Term, TermPos
from t_search.evaluators.fitness import l2

class SemanticallyDrivenCrossover(RPX):
    ''' Beadle and Johnson (2008, 2009b) '''
    def __init__(self, *, 
                 semantics: Semantics,
                 evaluator: Evaluator,
                 min_d = 1e-1, 
                 max_d=1e+2, 
                 **kwargs):
        super().__init__(**kwargs)
        self.min_d = min_d
        self.max_d = max_d
        self.semantics = semantics
        self.evaluator = evaluator

    def crossover_positions(self, term: Term, position: TermPos, 
                                                        other_term: Term, other_position: TermPos) -> Term | None:
        mutated_term = super().crossover_positions(term, position, other_term, other_position)
        if mutated_term is None: 
            return None
        
        # check semantic difference
        term1_sem, term2_sem, mutated_term_sem = self.evaluator.eval([term, other_term, mutated_term])
        if not self.semantics.is_valid(term1_sem):
            self.should_break_mutation = True 
            return None 
        elif not self.semantics.is_valid(term2_sem) or not self.semantics.is_valid(mutated_term_sem):
            return None
        dist1 = l2(term1_sem, mutated_term_sem)
        dist2 = l2(term2_sem, mutated_term_sem)
        if dist1 < self.min_d or dist1 > self.max_d or dist2 < self.min_d or dist2 > self.max_d:
            return None

        return mutated_term 
