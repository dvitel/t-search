

from typing import Sequence

import torch
from t_search.evaluators.evaluator import Evaluator
from t_search.syntax import Term
from .base import Operator

class Elitism(Operator): 
    ''' Passes through the population but stores elite terms'''

    def __init__(self, *,
                 evaluator: Evaluator,
                 size: int = 10):
        self.elite_size = size
        self.evaluator = evaluator

    def __call__(self, population: Sequence[Term], next_ops: list['Operator'] = []):
        elite: list[Term] = []
        fitness = self.evaluator.eval(population, return_fitness="tensor").fitness
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
        children = self.call_next(population, next_ops)
        # children are not evaluated yet - add elite anyway
        if len(children) > len(elite):
            children[-len(elite):] = elite # TODO: should we replace worst?
        return children