
from typing import TYPE_CHECKING, Sequence

import torch

from t_search.syntax import Term

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class Listener: 
    ''' Reactive to new terms and semantics. May produce new terms. '''

    def __init__(self, name: str):
        self.name = name 
        self.metrics = {}    

    def on_start(self, solver: 'GPSolver'):
        ''' Called when the solver starts search '''
        self.metrics = {}

    def on_gen_start(self, solver: 'GPSolver', gen: int, population: Sequence[Term]):
        ''' Called at start of each generation '''
        pass 

    def on_gen_end(self, solver: 'GPSolver', gen: int, population: Sequence[Term]):
        ''' Called at end of each generation '''
        pass     

    def on_end(self, solver: 'GPSolver'):
        ''' Called when the solver finishes the search '''
        pass

    def on_eval(self, solver: 'GPSolver', term: Term, semantics: torch.Tensor, fitness: torch.Tensor | None = None) -> Sequence[Term] | None:
        ''' Called on new evaluation '''
        pass