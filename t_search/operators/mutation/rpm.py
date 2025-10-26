from typing import TYPE_CHECKING

from syntax import Term, TermPos
from syntax.generation import grow

from .base import PositionMutation

if TYPE_CHECKING:
    from t_search.solver import GPSolver


class RPM(PositionMutation):
    """One Random Position Mutation"""

    def __init__(self, name="RPM", *, max_grow_depth=5, **kwargs):
        super().__init__(name, **kwargs)
        self.max_grow_depth = max_grow_depth

    def mutate_position(
        self, solver: "GPSolver", term: Term, position: TermPos
    ) -> Term | None:
        start_context, arg_counts = solver.get_gen_constraints(term, position)

        new_term = grow(
            grow_depth=min(
                self.max_grow_depth, solver.max_term_depth - position.at_depth
            ),
            builders=solver.builders,
            start_context=start_context,
            arg_counts=arg_counts,
            gen_metrics=self.metrics,
            rnd=solver.rnd,
        )

        mutated_term = solver.replace_position(
            term, position, new_term, with_validation=False
        )
        return mutated_term
