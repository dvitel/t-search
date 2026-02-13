from t_search.base import ServiceBase
from t_search.evaluators.evaluator import Evaluator
from t_search.operators.mutation import TermMutation
from t_search.operators.syntax.reduce import Reduce
from t_search.syntax.syntax import Syntax
from t_search.syntax import Term

class SemanticGeometricMutation(TermMutation, ServiceBase):
    ''' Implementing Semantic Geometric Mutation from Moraglio 2012 
        Parent program is lineary combined with random term 

        p' = p + r * (t1 - t2)
        r - random const 
        t1, t2 - random terms 
    '''
    def __init__(self, *, 
                    syntax: Syntax,
                    evaluator: Evaluator,
                    min_grow_depth = 3,
                    max_grow_depth = 5, 
                    num_tries = 2, 
                    epsilon = 0.02, 
                    check_validity: bool = True,
                    simplifier: Reduce | None = None,
                    **kwargs):
        super().__init__(**kwargs)
        self.num_tries = num_tries
        self.min_grow_depth = min_grow_depth
        self.max_grow_depth = max_grow_depth
        self.epsilon = epsilon
        self.minus_one: Term | None = None
        self.check_validity = check_validity
        self.simplifier = simplifier
        self.syntax = syntax
        self.evaluator = evaluator

    def init(self, **kwargs):
        assert self.syntax.has_op("add"), "SGX requires 'add' operator in the syntax."
        assert self.syntax.has_op("mul"), "SGX requires 'mul' operator in the syntax."
        self.minus_one = self.syntax.get_const(value = -1.0)   

    def mutate_term(self, term: Term) -> Term | None:

        mutated_term = None
        
        for _ in range(self.num_tries):
            d1 = self.rnd.randint(self.min_grow_depth, self.max_grow_depth+1)
            t1 = self.syntax.grow(d1)
            d2 = self.rnd.randint(self.min_grow_depth, self.max_grow_depth+1)
            t2 = self.syntax.grow(d2)

            mutated_term = self.syntax.get_op("add", 
                                              term, 
                                              self.syntax.get_op("mul", 
                                                self.syntax.get_const(value = self.rnd.random() * self.epsilon),
                                                self.syntax.get_op("add", 
                                                    t1, 
                                                    self.syntax.get_op("mul", 
                                                                        self.minus_one, 
                                                                        t2
                                                                        )
                                                    )
                                                )
                                              )
            
            if self.simplifier is not None:
                mutated_term = self.simplifier.mutate_term(mutated_term)
            if self.check_validity and not self.syntax.is_valid(mutated_term):
                mutated_term = None
            if mutated_term is not None:
                break

        return mutated_term 