from typing import TYPE_CHECKING
from .base import PositionCrossover
from syntax import Term, TermPos

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class RPX(PositionCrossover):
    ''' One Random Position Crossover '''

    def __init__(self, name: str = "RPX", **kwargs):
        super().__init__(name, **kwargs)
        self.crossover_cache: dict[tuple[Term, Term, int, Term], Term] = {}    

    def crossover_positions(self, solver: 'GPSolver', term: Term, position: TermPos, 
                                                        other_term: Term, other_position: TermPos) -> Term | None:

        crossover_key = (term, position.term, position.occur, other_position.term)
        if crossover_key in self.crossover_cache:
            child = self.crossover_cache[crossover_key]
            return child 

        child = solver.replace_position(term, position, other_position.term)
        
        return child
