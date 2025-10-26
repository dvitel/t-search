

from .base import TermMutation
from .reduce import Reduce
from syntax import Term
from syntax.generation import grow 
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from t_search.solver import GPSolver

class SGM(TermMutation):
    ''' Implementing Semantic Geometric Mutation from Moraglio 2012 
        Parent program is lineary combined with random term 

        p' = p + r * (t1 - t2)
        r - random const 
        t1, t2 - random terms 
    '''
    def __init__(self, name = "SGM", *, max_grow_depth = 5, num_tries = 2, epsilon = 0.02, 
                    check_validity: bool = False,
                    simplifier: Reduce | None = None,
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

    def mutate_term(self, solver: 'GPSolver', term: Term) -> Term | None:

        mutated_term = None
        
        for _ in range(self.num_tries):
            t1 = grow(grow_depth = self.max_grow_depth,
                        builders = solver.builders,
                        gen_metrics = self.metrics, rnd = solver.rnd) 

            t2 = grow(grow_depth = self.max_grow_depth,
                        builders = solver.builders, 
                        gen_metrics = self.metrics, rnd = solver.rnd)               
                
            neg_t2 = solver.op_builders["mul"].fn(self.minus_one, t2)
            t1_minus_t2 = solver.op_builders["add"].fn(t1, neg_t2)
            r = solver.const_builder.fn(value = solver.rnd.rand() * self.epsilon)
            delta_term = solver.op_builders["mul"].fn(r, t1_minus_t2)
            mutated_term = solver.op_builders["add"].fn(term, delta_term)
            if self.simplifier is not None:
                mutated_term = self.simplifier.mutate_term(solver, [mutated_term])
            if self.check_validity and not solver.is_valid(mutated_term):
                mutated_term = None
            if mutated_term is not None:
                break

        return mutated_term 