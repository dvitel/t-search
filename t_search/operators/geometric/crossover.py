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
        
        other_term = self.trim_term(other_term)        
        
        t1 = term  
        t2 = other_term       

        s1, s2 = self.evaluator.eval([t1, t2]) # get vectors
        new_term = self.trim_term(t1)
        if new_term != t1:
            t1 = new_term
            s1 = self.semantics.get_outputs(t1)

        if self.semantics.is_valid(t1) is False:
            self.should_break_mutation = True
            return None
        elif self.semantics.is_valid(t2) is False:
            return None 
        if self.use_best_epsilon:

            sd = s1 - s2 
            sd2 = (sd * sd).sum()

            best_epsilon = ((self.target - s2) * sd).sum() / (sd2 + 1e-8)
            best_epsilon = (1 - (2 * self.rnd.random() - 1) * self.alpha) * best_epsilon
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

        # trimmed_term = self.trim_deep_term(final_term, should_trim=False)
        # if isinstance(trimmed_term, Value):
        #     return None   
        
        return final_term