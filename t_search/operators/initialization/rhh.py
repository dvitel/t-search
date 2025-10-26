

from typing import TYPE_CHECKING, Optional

from syntax import Term
from syntax.generation import grow
from .base import Initialization

if TYPE_CHECKING:
    from t_search.solver import GPSolver
    
class RHH(Initialization):
    ''' Ramped Half and Half initialization operator '''

    def __init__(self, name: str = "rhh", *, 
                min_depth = 1, max_depth = 5, grow_proba = 0.5,
                leaf_proba: Optional[float] = 0.1,
                freq_skew: bool = False):
        super().__init__(name)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.grow_proba = grow_proba
        self.leaf_proba = leaf_proba
        self.freq_skew = freq_skew

    def _rhh(self, solver: 'GPSolver'):
        depth = solver.rnd.randint(self.min_depth, self.max_depth + 1)
        leaf_prob = self.leaf_proba if solver.rnd.rand() < self.grow_proba else 0
        term = grow(solver.builders, grow_depth = depth, 
                    grow_leaf_prob = leaf_prob, rnd = solver.rnd, gen_metrics=self.metrics,
                    freq_skew = self.freq_skew)
        return term
    
    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        population = []
        for _ in range(pop_size):
            term = self._rhh(solver)
            # print(str(term))
            if term is not None:
                population.append(term)
        return population

class RHHCached(RHH):
    ''' Considers inner terms of solver syntax cache '''
    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        if not solver.cache_terms:
            return super().pop_init(solver, pop_size)
        none_count = 0
        sz = pop_size - len(solver.vars)
        while len(solver.syntax) < sz:
            term = self._rhh(solver)
            if term is None:
                none_count += 1
            if none_count == pop_size:
                break 
        population = list(solver.syntax.values())[:sz]
        population.extend(solver.vars.values())    
        return population