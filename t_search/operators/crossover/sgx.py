from t_search.base import ServiceBase
from t_search.evaluators.evaluator import Evaluator
from .base import TermCrossover
from ..mutation import Reduce
from t_search.syntax import Term
from t_search.evaluators.fitness import l2

class SGX(TermCrossover, ServiceBase):
    ''' Implementing Semantic Geometric Crossover from Moraglio 2012 
        Linear combination of programs
    '''
    def __init__(self, *, 
                    evaluator: Evaluator,
                    max_grow_depth = 5, 
                    num_tries = 2, 
                    epsilon = 1.0, 
                    check_validity: bool = True,
                    simplifier: Reduce | None = None,
                    min_d: float | None = 1e-2,
                    **kwargs):
        super().__init__(**kwargs)
        self.num_tries = num_tries
        self.max_grow_depth = max_grow_depth
        self.epsilon = epsilon
        self.minus_one: Term | None = None
        self.check_validity = check_validity
        self.simplifier = simplifier
        self.evaluator = evaluator
        self.min_d = min_d

    def init(self):
        assert self.syntax.has_op("add"), "SGX requires 'add' operator in the syntax."
        assert self.syntax.has_op("mul"), "SGX requires 'mul' operator in the syntax."
        self.minus_one = self.syntax.get_const(value = -1.0)        

    def crossover_terms(self, term: Term, other_term: Term) -> Term | None:

        mutated_term = None
        
        t1 = term  
        t2 = other_term       

        for _ in range(self.num_tries):

            mutated_term = self.syntax.get_op("add",
                                t1,
                                self.syntax.get_op("mul",
                                    self.syntax.get_const(self.rnd.random() * self.epsilon),
                                    self.syntax.get_op("add", 
                                        t2,
                                        self.syntax.get_op("mul",
                                            self.minus_one,
                                            t1
                                        )
                                    )
                                )
                            )
            
            if self.simplifier is not None:
                mutated_term = self.simplifier.mutate_term(mutated_term)
            if self.check_validity and not self.syntax.is_valid(mutated_term):
                mutated_term = None
                continue 

            if self.min_d is not None: # check effectiveness of the operator
                term1_sem, term2_sem, mutated_term_sem, *_ = self.evaluator.eval([term, other_term, mutated_term], return_outputs="list").outputs
                dist1 = l2(term1_sem, mutated_term_sem)
                dist2 = l2(term2_sem, mutated_term_sem)
                if dist1 < self.min_d or dist2 < self.min_d:
                    mutated_term = None
                    continue
            if mutated_term is not None:
                break

        return mutated_term 