from typing import Sequence

import numpy as np

from t_search.operators.operator import Operator
from t_search.syntax import Term

class Selection(Operator):
    ''' Base class for selection operators '''

    def __init__(self, *, selection_size: int, rnd: np.random.Generator):
        self.rnd: np.random.Generator = rnd
        self.selection_size: int = selection_size
    
    def __call__(self, population: Sequence[Term], size: int | None = None) -> Sequence[Term]:
        children = self.rnd.choice(population, size or self.selection_size).tolist()
        return children