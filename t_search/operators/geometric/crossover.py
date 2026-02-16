import torch
from t_search.evaluators.evaluator import Evaluator
from t_search.operators.crossover import TermCrossover
from t_search.operators.geometric.base import BaseGeometricMutation
from t_search.syntax import Term, Value
from t_search.evaluators.fitness import l2

class SemanticGeometricCrossover(BaseGeometricMutation, TermCrossover):
    ''' Implementing Semantic Geometric Crossover from Moraglio 2012 
        Linear combination of programs
    '''
    def __init__(self,
                    **kwargs):
        super().__init__(**kwargs)

    def crossover_terms(self, term: Term, other_term: Term) -> Term | None:

        if term == other_term:
            return term        
        
        t1 = term  
        t2 = other_term       

        self.evaluator.eval([t1, t2]) # get vectors
        if self.use_best_epsilon:
            s1, s2 = self.semantics.get_outputs([t1, t2], return_type="list")

            sd = s1 - s2 
            sd2 = (sd * sd).sum()

            best_epsilon = ((self.target - s2) * sd).sum() / (sd2 + 1e-8)
            best_epsilon.clamp_(-self.epsilon, self.epsilon)
        
        else:
            best_epsilon = self.rnd.random()*self.epsilon


        # if (self.syntax.get_depth(t1) > (self.syntax.max_term_depth - 4)):
        #     return term # noop        
        
        # if (self.syntax.get_depth(t2) > (self.syntax.max_term_depth - 3)):
        #     return None # next try        
        # assert torch.isfinite(best_epsilon)

        t1_term = self.syntax.get_op("mul", self.syntax.get_const(value = best_epsilon), t1)
        t2_term = self.syntax.get_op("mul", self.syntax.get_const(value = 1 - best_epsilon), t2)

        final_term = self.syntax.get_op("add", t1_term, t2_term)

        trimmed_term = self.trim_deep_term(final_term)
        if isinstance(trimmed_term, Value):
            return None   

        self.evaluator.eval(trimmed_term)     
        
        # final_term = self.trim_deep_term(mutated_term)
        # if self.check_validity and not self.syntax.is_valid(final_term):
        #     return None 

        # if self.min_d is not None: # check effectiveness of the operator
        #     term1_sem, term2_sem, mutated_term_sem, *_ = self.evaluator.eval([term, other_term, final_term])
        #     dist1 = l2(term1_sem[1], mutated_term_sem[1])
        #     dist2 = l2(term2_sem[1], mutated_term_sem[1])
        #     if dist1 < self.min_d or dist2 < self.min_d:
        #         return None
        #     else:
        #         final_term = mutated_term_sem[0]
        return trimmed_term