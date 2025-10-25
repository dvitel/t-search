
from typing import TYPE_CHECKING
from t_search.term import Term

if TYPE_CHECKING:
    from t_search.solver import GPSolver
    
class Initialization:

    def __init__(self, name: str):
        self.name = name 
        self.metrics = {}

    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        return []

    def __call__(self, solver: 'GPSolver', pop_size: int):
        ''' Use to trigger initialization, pop_init should not be called directly '''
        self.metrics = {}
        population = self.pop_init(solver, pop_size)
        return population