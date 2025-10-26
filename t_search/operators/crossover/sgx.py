

from .base import TermCrossover
from ..mutation import Reduce
from syntax import Term
from utils.metrics import l2
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from t_search.solver import GPSolver

class SGX(TermCrossover):
    ''' Implementing Semantic Geometric Crossover from Moraglio 2012 
        Linear combination of programs
    '''
    def __init__(self, name = "SGX", *, max_grow_depth = 5, num_tries = 2, epsilon = 1.0, 
                    check_validity: bool = False,
                    simplifier: Reduce | None = None,
                    min_d: float | None = 1e-2,
                    **kwargs):
        super().__init__(name, **kwargs)
        self.num_tries = num_tries
        self.max_grow_depth = max_grow_depth
        self.epsilon = epsilon
        self.minus_one: Term | None = None
        self.check_validity = check_validity
        self.simplifier = simplifier

    def op_init(self, solver):
        self.minus_one = solver.const_builder.fn(value = -1.0)        

    def crossover_terms(self, solver: 'GPSolver', term: Term, other_term: Term) -> Term | None:

        mutated_term = None
        
        t1 = term  
        t2 = other_term       

        for _ in range(self.num_tries):


            neg_t2 = solver.op_builders["mul"].fn(self.minus_one, t2)
            t1_minus_t2 = solver.op_builders["add"].fn(t1, neg_t2)
            r = solver.const_builder.fn(value = solver.rnd.rand() * self.epsilon)
            delta_term = solver.op_builders["mul"].fn(r, t1_minus_t2)
            mutated_term = solver.op_builders["add"].fn(term, delta_term)
            if self.simplifier is not None:
                mutated_term = self.simplifier.mutate_term(solver, [mutated_term])
            if self.check_validity and not solver.is_valid(mutated_term):
                mutated_term = None
                continue 

            if self.min_d is not None: # check effectiveness of the operator
                term1_sem, term2_sem, mutated_term_sem, *_ = solver.eval([term, other_term, mutated_term], return_outputs="list").outputs
                dist1 = l2(term1_sem, mutated_term_sem)
                dist2 = l2(term2_sem, mutated_term_sem)
                if dist1 < self.min_d or dist2 < self.min_d:
                    mutated_term = None
                    continue
            if mutated_term is not None:
                break

        return mutated_term 