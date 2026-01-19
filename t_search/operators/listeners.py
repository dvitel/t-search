
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

    def on_eval(self, terms: list[Term], semantics: torch.Tensor) -> None:
        ''' Called on new evaluation '''
        pass