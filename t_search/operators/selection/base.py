from typing import Sequence

import numpy as np

from t_search.syntax import Term
from ..base import Operator

class Selection(Operator):
    ''' Base class for selection operators '''

    def __init__(self, *, selection_size: int, rnd: np.random.Generator):
        self.rnd: np.random.Generator = rnd
        self.selection_size: int = selection_size
    
    def exec(self, population):
        return self.select(population, self.selection_size)
    
    def select(self, population: Sequence[Term], selection_size: int) -> Sequence[Term]:
        children = self.rnd.choice(population, selection_size).tolist()
        return children