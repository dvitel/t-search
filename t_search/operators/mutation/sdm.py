

from .rpm import RPM
from typing import TYPE_CHECKING

from t_search.term import Term, TermPos
from t_search.util import l2_distance
if TYPE_CHECKING:
    from t_search.solver import GPSolver

class SDM(RPM):
    ''' Semantically Driven Mutation Beadle and Johnson (2008, 2009b) '''
    def __init__(self, name = "SDM", *, min_d = 1e-1, max_d=1e+2, **kwargs):
        super().__init__(name, **kwargs)
        self.min_d = min_d
        self.max_d = max_d

    def mutate_position(self, solver: 'GPSolver', term: Term, position: TermPos) -> Term | None:
        mutated_term = super().mutate_position(solver, term, position)
        if mutated_term is None: 
            return None
        
        # check semantic difference
        parent_sem, mutated_term_sem, *_ = solver.eval([term, mutated_term], return_outputs="list").outputs
        dist = l2_distance(parent_sem, mutated_term_sem)
        if dist < self.min_d or dist > self.max_d:
            return None        

        return mutated_term 
