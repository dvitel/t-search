''' Implementation of optimizer '''

from typing import TYPE_CHECKING

import torch

from .optimization import OptimState
from t_search.syntax.term import Term

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class Optimizer:

    def reset(self) -> None:
        pass

    def get_optim_state(self, 
                        solver: 'GPSolver', 
                        term: Term, 
                        initial_term_loss: torch.Tensor | None = None) -> OptimState:
        pass

    def optimize(self,
        solver: 'GPSolver',
        term: Term,
        initial_term_loss: torch.Tensor | None = None,
    ) -> Term:
        pass    