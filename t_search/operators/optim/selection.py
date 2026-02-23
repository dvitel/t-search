from typing import Sequence

import torch

from t_search.operators.classic.ts import TS
from t_search.syntax.term import Term

class FrontierSelection(TS):
    ''' TS with ensured presense of frontier terms  '''
    def __init__(self,
                 term_frontier: set[Term],
                  **kwargs):
        super().__init__(**kwargs)
        self.term_frontier = term_frontier

    def __call__(self, population: Sequence[Term]) -> Sequence[Term]:
        children = super().__call__(population, size=self.selection_size)
        # children = list(self.term_frontier)
        # left_size = self.selection_size - len(children)
        # other_children = super().__call__(population, size=left_size) 
        # children.extend(other_children)
        return children