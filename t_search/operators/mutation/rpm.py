from t_search.syntax import Term, TermPos
from t_search.syntax.generation import grow
from t_search.syntax.syntax import Syntax

from .base import PositionMutation

class RPM(PositionMutation):
    """One Random Position Mutation"""

    def __init__(self, *, 
                 syntax: Syntax,
                 max_grow_depth=5, 
                 freq_skew: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_grow_depth = max_grow_depth
        self.syntax = syntax
        self.freq_skew = freq_skew

    def mutate_position(
        self, term: Term, position: TermPos
    ) -> Term | None:
        start_context, arg_counts = self.syntax.get_gen_constraints(term, position)

        new_term = self.syntax.grow(
            min(self.max_grow_depth, self.syntax.max_term_depth - position.at_depth),
            start_context=start_context,
            arg_counts=arg_counts,
            freq_skew=self.freq_skew,
        )

        mutated_term = self.syntax.replace_position(
            term, position, new_term, with_validation=False
        )
        return mutated_term
