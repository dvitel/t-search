from t_search.operators.mutation import PositionMutation
from t_search.syntax import Term, TermPos
from t_search.syntax.syntax import Syntax

class RPM(PositionMutation):
    """One Random Position Mutation"""

    def __init__(self, *, 
                 max_grow_depth=5, 
                 freq_skew: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_grow_depth = max_grow_depth
        self.freq_skew = freq_skew

    def mutate_position(
        self, term: Term, position: TermPos
    ) -> Term | None:

        new_term = self.syntax.grow(
            min(self.max_grow_depth, self.syntax.max_term_depth - position.at_depth),
            start_pos=(term, position),
            freq_skew=self.freq_skew,
        )

        mutated_term = self.syntax.replace_position(
            term, position, new_term, with_validation=False
        )
        return mutated_term
