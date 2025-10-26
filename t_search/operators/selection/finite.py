

from typing import TYPE_CHECKING
from .base import Operator
from syntax import Term


if TYPE_CHECKING:
    from t_search.solver import GPSolver
        
class Finite(Operator):
    ''' Selects only children that have finite or unknown outputs 
        Resorts back to full population if all outputs are infinie or nan
    '''
    def __init__(self, name: str = "finite"):
        super().__init__(name)

    def exec(self, solver: 'GPSolver', population: list[Term]):
        children = [ch for ch in population if ch not in solver.invalid_term_outputs]
        if len(children) == 0:
            print("WARN: all population has nans or infs")
            return population
        return children
