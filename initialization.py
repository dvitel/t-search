''' Population initialization operators '''

from typing import Optional
from term import Term, grow

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gp import GPSolver  # Import only for type checking

class Initialization:

    def __init__(self, name: str):
        self.name = name 
        self.metrics = {}

    def _init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        return []

    def __call__(self, solver: 'GPSolver', pop_size: int):
        ''' Use to trigger initialization, _init should not be called directly '''
        self.metrics = {}
        population = self._init(solver, pop_size)
        return population
    
class RHH(Initialization):
    ''' Ramped Half and Half initialization operator '''

    def __init__(self, name: str = "rhh", *, 
                min_depth = 1, max_depth = 5, grow_proba = 0.5,
                leaf_proba: Optional[float] = 0.1):
        super().__init__(name)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.grow_proba = grow_proba
        self.leaf_proba = leaf_proba

    def _rhh(self, solver: 'GPSolver'):
        depth = solver.rnd.randint(self.min_depth, self.max_depth + 1)
        leaf_prob = self.leaf_proba if solver.rnd.rand() < self.grow_proba else 0
        term = grow(solver.builders, grow_depth = depth, 
                    grow_leaf_prob = leaf_prob, rnd = solver.rnd, gen_metrics=self.metrics)
        return term
    
    def _init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        population = []
        for _ in range(pop_size):
            term = self._rhh(solver)
            # print(str(term))
            if term is not None:
                population.append(term)
        return population
    
class CachedRHH(RHH):
    ''' Considers inner terms of solvver syntax cache '''
    def _init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        if not solver.cache_terms:
            return super()._init(solver, pop_size)
        none_count = 0
        sz = pop_size - len(solver.vars)
        while len(solver.syntax) < sz:
            term = self._rhh(solver)
            if term is None:
                none_count += 1
            if none_count == pop_size:
                break 
        population = list(solver.syntax.values())[:sz]
        population.extend(solver.vars)    
        return population