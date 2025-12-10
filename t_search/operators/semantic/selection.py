from typing import Sequence
import torch
from t_search.evaluators.semantics import Semantics
from t_search.operators.classic.ts import TS
from t_search.syntax import Term

class SemanticTournamentSelection(TS):
    ''' Galván-López et al. (2013)
        First parent with TS, second is the best parent != to first one or first one. 
        "Using Semantics in Selection Mechanism in Genetic Programming: a Simple Methods for Promoting Semantic Diversity"
    '''
    def __init__(self, *, 
                 semantics: Semantics,
                 rtol=1e-5, 
                 atol=1e-4, 
                 **kwargs):
        super().__init__(**kwargs)   
        self.semantics = semantics 
        self.rtol = rtol
        self.atol = atol

    def __call__(self, population: Sequence[Term]) -> Sequence[Term]:
        half_size = self.selection_size // 2
        half_parents = super().select(population, half_size + (self.selection_size % 2)) 
        half_parents_sems = self.semantics.get_outputs(half_parents, return_type="tensor")
        children = []
        for i in range(half_size):
            first_parent = half_parents[i]
            first_sem = half_parents_sems[i]
            # find second parent
            candidiates = self.rnd.choice(population, size = self.tournament_size)
            candidate_sem = self.semantics.get_outputs(candidiates, return_type="tensor")
            mask = torch.isclose(candidate_sem, first_sem, rtol=self.rtol, atol=self.atol).all(dim=-1)
            del candidate_sem
            filter_ids, = torch.where(~mask)
            filtered_candidates = [candidiates[i] for i in filter_ids.tolist()]
            c_fitness = self.semantics.get_outputs(filtered_candidates, return_type="tensor")
            best_id = torch.argmin(c_fitness).item()
            best_parent = candidiates[best_id]
            del c_fitness
            children.append(first_parent)
            children.append(best_parent)

        del half_parents_sems

        if self.selection_size % 2 == 1:
            children.append(half_parents[-1])
        return children