''' Implementation of optimizer '''
from __future__ import annotations

from t_search.syntax.term import Term

class Optimizer:

    # def get_optim_state(self, 
    #                     term: Term, 
    #                     initial_term_loss: torch.Tensor | None = None) -> OptimState:
    #     pass

    def optimize(self,
        # evaluator: Evaluator,
        term: Term,
        # initial_term_loss: torch.Tensor | None = None,
    ) -> Term:
        pass    