

from typing import TYPE_CHECKING, Sequence

import torch

from term import Term
from .base import Selection

if TYPE_CHECKING:
    from gp import GPSolver

class TS(Selection):
    ''' Tournament selection operator '''
    
    def __init__(self, name: str = "tournament", tournament_size: int = 7):
        super().__init__(name)
        self.tournament_size = tournament_size

    def select(self, solver: 'GPSolver', population, selection_size: int) -> Sequence[Term]:
        ''' Fitness is 1d tensor of fitness selected for tournament '''
        fitness = solver.eval(population, return_fitness="tensor").fitness
        selected_ids = torch.randint(fitness.shape[0], (selection_size, self.tournament_size), dtype=torch.int, device=fitness.device,
                                    generator=solver.torch_gen)
        selected_fitnesses = fitness[selected_ids]
        best_id_id = torch.argmin(selected_fitnesses, dim=-1)
        best_ids = torch.gather(selected_ids, dim=-1, index = best_id_id.unsqueeze(-1)).squeeze(-1)
        del selected_ids, selected_fitnesses, best_id_id
        del fitness
        children = [population[best_id] for best_id in best_ids.tolist()]
        return children
