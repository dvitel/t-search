

from typing import TYPE_CHECKING, Sequence

import torch
from t_search.term import Term
from .base import Operator

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class Elitism(Operator): 
    ''' Passes through the population but stores elite terms'''

    def __init__(self, name: str = "elitism", size: int = 10):
        super().__init__(name)
        self.elite_size = size

    def __call__(self, solver: 'GPSolver', population: Sequence[Term], next_ops: list['Operator'] = []):
        elite: list[Term] = []
        fitness = solver.eval(population, return_fitness="tensor").fitness
        sorted_ids = torch.argsort(fitness, dim=0)
        elite_ids = sorted_ids[:self.elite_size].tolist()
        del fitness, sorted_ids
        elite = [population[i] for i in elite_ids]
        # bad_ids = sorted_ids[-self.elite_size:].tolist()
        # passed_population = [] 
        # bad_id_set = set(bad_ids)
        # for i, term in enumerate(population):
        #     if i not in bad_id_set:
        #         passed_population.append(term)
        children = self.call_next(solver, population, next_ops)
        # children are not evaluated yet - add elite anyway
        children.extend(elite) 
        return children