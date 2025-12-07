
from typing import Sequence

import torch

from t_search.syntax import Term

class GenListener: 
    
    def on_gen_start(self, gen: int, population: Sequence[Term]):
        ''' Called at start of each generation '''
        pass 

    def on_gen_end(self, gen: int, population: Sequence[Term]):
        ''' Called at end of each generation '''
        pass    

class EvalListener:  

    def on_eval(self, term: Term, semantics: torch.Tensor, fitness: torch.Tensor | None = None) -> Sequence[Term] | None:
        ''' Called on new evaluation '''
        pass