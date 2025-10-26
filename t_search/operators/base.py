''' Base interfaces for different evolutionary operators '''

from typing import TYPE_CHECKING, Sequence

import torch

from syntax import Term

if TYPE_CHECKING:
    from t_search.solver import GPSolver  # Import only for type checking

class Operator:
    def __init__(self, name: str):
        self.name = name 
        self.metrics = {}

    def reset_metrics(self):
        self.metrics = {}

    def op_init(self, solver: 'GPSolver'):
        pass        
    
    def exec(self, solver: 'GPSolver', population: Sequence[Term]) -> Sequence[Term]:
        ''' Executes only this operator and update existing metrics state '''
        return population
    
    def call_next(solver: 'GPSolver', population: Sequence[Term], next_ops: list['Operator'] = []):
        if len(next_ops) > 0:
            next_op, *rest_ops = next_ops
            children = next_op(solver, children, rest_ops)        
            return children
        return population

    def __call__(self, solver: 'GPSolver', population: Sequence[Term], next_ops: list['Operator'] = []):
        ''' Executes operator in the chain. New metrics are stored in self.metrics '''
        self.reset_metrics()
        children = self.exec(solver, population)
        children = self.call_next(solver, children, next_ops)
        return children

class TermsListener: 
    ''' Interface to listen for new terms appearing during the eval. 
        
    '''
    def register_terms(self, solver: 'GPSolver', terms: list[Term], semantics: torch.Tensor) -> list[Term]: 
        pass 