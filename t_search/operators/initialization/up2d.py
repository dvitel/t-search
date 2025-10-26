

from typing import TYPE_CHECKING
from syntax import Term
from syntax.generation import TermGenContext, gen_all_terms
from .base import Initialization

if TYPE_CHECKING:
    from t_search.solver import GPSolver

class Up2D(Initialization):
    ''' All trees (without constants) up to specified depth '''

    def __init__(self, name: str = "up2d", *, depth = 2, force_pop_size: bool = False):
        super().__init__(name)
        self.depth = depth
        self.gen_context: TermGenContext | None = None
        self.force_pop_size = force_pop_size

    def pop_init(self, solver: 'GPSolver', pop_size: int) -> list[Term]:
        if self.gen_context is None:
            self.gen_context = TermGenContext(solver.builders.default_gen_context.min_counts,
                                            solver.builders.default_gen_context.max_counts.copy(),
                                            solver.builders.default_gen_context.arg_limits)
            self.gen_context.max_counts[solver.const_builder.id] = 0  # no constants
        population = gen_all_terms(solver.builders, depth=self.depth, start_context=self.gen_context)
        if self.force_pop_size:
            if len(population) > pop_size:
                population = solver.rnd.choice(population, size=pop_size, replace=False).tolist()
            elif len(population) < pop_size:
                pop_extend = pop_size - len(population)
                population.extend(solver.rnd.choice(population, size=pop_extend, replace=True))
        return population