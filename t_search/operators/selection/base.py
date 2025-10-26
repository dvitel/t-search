


from typing import TYPE_CHECKING, Sequence

from syntax import Term
from ..base import Operator

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class Selection(Operator):
    ''' Base class for selection operators '''

    def __init__(self, name: str = "randSel", replace: bool = True):
        super().__init__(name)
        self.replace = replace
    
    def exec(self, solver: 'GPSolver', population):
        return self.select(solver, population, solver.pop_size)
    
    def select(self, solver: 'GPSolver', population: Sequence[Term], selection_size: int) -> Sequence[Term]:
        children = solver.rnd.choice(population, selection_size, replace=self.replace).tolist()
        return children