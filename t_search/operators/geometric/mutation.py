import torch
from t_search.operators.geometric.base import BaseGeometricMutation
from t_search.syntax import Term, Value

class SemanticGeometricMutation(BaseGeometricMutation):
    ''' Implementing Semantic Geometric Mutation from Moraglio 2012 
        Parent program is lineary combined with random term 

        p' = p + r * (t1 - t2)
        r - random const 
        t1, t2 - random terms 
    '''
    def __init__(self, *, 
                    min_grow_depth = 3,
                    max_grow_depth = 5, 
                    **kwargs):
        super().__init__(**kwargs)
        self.min_grow_depth = min_grow_depth
        self.max_grow_depth = max_grow_depth        

    def mutate_term(self, term: Term) -> Term | None:

        final_term = None

        for _ in range(10):
            t1 = self.syntax.grow(self.max_grow_depth)
            if self.syntax.get_depth(t1) >= self.min_grow_depth:
                break 
        for _ in range(10):
            t2 = self.syntax.grow(self.max_grow_depth)
            if self.syntax.get_depth(t2) >= self.min_grow_depth:
                break 

        f, s1, s2 = self.evaluator.eval([term, t1, t2]) # get vectors 
        if not self.semantics.is_valid(f):
            self.should_break_mutation = True
            return None
        elif not self.semantics.is_valid(t1) or not self.semantics.is_valid(t2):
            return None
        
        if self.use_best_epsilon:

            sd = s1 - s2 
            sd2 = (sd * sd).sum()        

            best_epsilon = ((self.target - f) * sd).sum() / (sd2 + 1e-8)
            best_epsilon.clamp_(-self.epsilon, self.epsilon)

        else:

            best_epsilon = self.rnd.random() * self.epsilon

        # assert torch.isfinite(best_epsilon)

        # ind.weights[func_ids[0]] += best_epsilon
        # ind.weights[func_ids[1]] -= best_epsilon
        # ind.outputs += best_epsilon * sd

        t1_term = self.syntax.get_op("mul", self.syntax.get_const(value = best_epsilon), t1)
        t2_term = self.syntax.get_op("mul", self.syntax.get_const(value = -best_epsilon), t2)

        final_term = self.syntax.get_op("add", term, self.syntax.get_op("add", t1_term, t2_term))

        trimmed_term = self.trim_deep_term(final_term)
        if isinstance(trimmed_term, Value):
            return None

        self.evaluator.eval(trimmed_term)
        
        # final_term = self.trim_deep_term(mutated_term)
        # We do not validate here the depth of the tree
        # if self.check_validity and not self.syntax.is_valid(final_term):
        #     return None
        return trimmed_term 