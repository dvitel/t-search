''' Selection operators '''

from typing import TYPE_CHECKING

import torch

from term import Term
from util import Operator, stack_rows

if TYPE_CHECKING:
    from gp import GPSolver  # Import only for type checking


class Selection(Operator):
    ''' Base class for selection operators '''
    
    def get_size_without_elite(self, solver: 'GPSolver'):
        size = solver.pop_size
        for e in solver.elitism:
            size -= e.elite_size
        size = max(1, size)
        return size
    
    def _select(self, solver: 'GPSolver', population):
        children = solver.rnd.choice(population, self.get_size_without_elite(solver), replace=False)
        return children
    
    def __call__(self, solver: 'GPSolver', population):
        self.on_start()
        children = self._select(solver, population)
        return children
    

class TournamentSelection(Selection):
    ''' Tournament selection operator '''
    
    def __init__(self, name: str = "tournament", tournament_size: int = 7):
        super().__init__(name)
        self.tournament_size = tournament_size

    def _select(self, solver: 'GPSolver', population):
        ''' Fitness is 1d tensor of fitness selected for tournament '''
        size = self.get_size_without_elite(solver)
        present_terms, fitness_list = solver.get_terms_fitness(population)
        if len(fitness_list) == 0:
            return population
        fitness = torch.stack(fitness_list, dim=0)
        selected_ids = torch.randint(fitness.shape[0], (size, self.tournament_size), dtype=torch.int, device=fitness.device,
                                    generator=solver.torch_gen)
        selected_fitnesses = fitness[selected_ids]
        best_id_id = torch.argmin(selected_fitnesses, dim=-1)
        best_ids = torch.gather(selected_ids, dim=-1, index = best_id_id.unsqueeze(-1)).squeeze(-1)
        del selected_ids, selected_fitnesses, best_id_id
        del fitness
        children = [present_terms[best_id] for best_id in best_ids.tolist()]
        return children
    
class LexicaseSelection(Selection):
    ''' Lexicase selection operator '''
    
    def __init__(self, name: str = "lexicase", nan_error = torch.inf,):
        super().__init__(name)
        self.nan_error = nan_error

    def _select(self, solver: 'GPSolver', population):
        size = self.get_size_without_elite(solver)
        outputs_list = [solver.get_cached_output(term) for term in population]
        outputs = stack_rows(outputs_list, target_size=solver.target.shape[0])

        nan_interactions = torch.abs(outputs - solver.target)

        interactions = torch.nan_to_num(nan_interactions, nan=self.nan_error)
        del nan_interactions
        
        selected_ids = torch.zeros(size, dtype=torch.int, device=outputs.device)

        for pos_i in range(size):
            shuffled_test_ids = torch.randperm(interactions.shape[-1], device=interactions.device,
                                                generator=solver.torch_gen)
            candidate_ids = torch.arange(interactions.shape[0], device=interactions.device) # all candidates
            for test_id in shuffled_test_ids:
                test_min_diff = torch.min(interactions[candidate_ids, test_id])
                candidate_id_ids, = torch.where(interactions[candidate_ids, test_id] == test_min_diff)
                candidate_ids = candidate_ids[candidate_id_ids]
                if len(candidate_ids) == 1:
                    break
            if len(candidate_ids) == 1:
                selected_ids[pos_i] = candidate_ids[0]
                continue
            best_id_id = torch.randint(len(candidate_ids), (1,), device=interactions.device,
                                        generator=solver.torch_gen)
            selected_ids[pos_i] = candidate_ids[best_id_id]
        del interactions, shuffled_test_ids
        del outputs
        children = [population[term_id] for term_id in selected_ids.tolist()]
        return children

class Elitism(Selection): 
    ''' Passes through the population but stores elite terms'''

    def __init__(self, name: str = "elitism", size: int = 10):
        super().__init__(name)
        self.elite_size = size
        self.elite: list[Term] = []

    def get_elite(self):
        return self.elite

    def _select(self, solver: 'GPSolver', population: list[Term]):
        self.elite.clear()
        present_terms, present_fitness = solver.get_terms_fitness(population)
        if len(present_fitness) == 0:
            return population
        fitness = torch.stack(present_fitness, dim=0)
        sorted_ids = torch.argsort(fitness, dim=0)
        elite_ids = sorted_ids[:self.elite_size].tolist()
        self.elite = [present_terms[i] for i in elite_ids]
        return population
    
class Finite(Selection):
    ''' Selects only children that have finite or unknown outputs 
        Resorts back to  full population if all outputs are infinie or nan
    '''
    def __init__(self, name: str = "finite"):
        super().__init__(name)

    def _select(self, solver: 'GPSolver', population: list[Term]):
        children = [ch for ch in population if ch not in solver.invalid_term_outputs]
        if len(children) == 0:
            print("WARN: all population has nans or infs")
            return population
        return children