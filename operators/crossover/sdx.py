

from typing import TYPE_CHECKING
from .rpx import RPX
from term import Term, TermPos
from util import l2_distance

if TYPE_CHECKING:
    from gp import GPSolver 

class SDX(RPX):
    ''' Semantically Driven Crossover Beadle and Johnson (2008, 2009b) '''
    def __init__(self, name = "SDX", *, min_d = 1e-1, max_d=1e+2, **kwargs):
        super().__init__(name, **kwargs)
        self.min_d = min_d
        self.max_d = max_d

    def crossover_positions(self, solver: 'GPSolver', term: Term, position: TermPos, 
                                                        other_term: Term, other_position: TermPos) -> Term | None:
        mutated_term = super().crossover_positions(solver, term, position, other_term, other_position)
        if mutated_term is None: 
            return None
        
        # check semantic difference
        term1_sem, term2_sem, mutated_term_sem, *_ = solver.eval([term, other_term, mutated_term], return_outputs="list").outputs
        dist1 = l2_distance(term1_sem, mutated_term_sem)
        dist2 = l2_distance(term2_sem, mutated_term_sem)
        if dist1 < self.min_d or dist1 > self.max_d or dist2 < self.min_d or dist2 > self.max_d:
            return None       

        return mutated_term 
