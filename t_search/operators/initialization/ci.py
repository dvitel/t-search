from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull

from syntax import Term
from spatial import TermVectorStorage

from .sdi import SDI

if TYPE_CHECKING:
    from t_search.solver import GPSolver


class CI(SDI):
    """Competent semantic initialization"""

    def __init__(self, name: str = "CI", *, index: TermVectorStorage):
        super().__init__(name, index=index)

    def is_point_inside_hull(
        self, point: np.ndarray, hull: ConvexHull, tolerance: float = 1e-12
    ) -> bool:
        results = np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1]
        return np.all(results <= tolerance).item()

    def pop_init(self, solver: "GPSolver", pop_size: int) -> list[Term]:
        ci_population: list[Term] = []
        max_try_count = 3
        i = 0
        target_inside_hull = False
        target = solver.target.cpu().numpy()
        while len(ci_population) < pop_size and (i < max_try_count):
            i += 1
            population = super().pop_init(solver, i * pop_size)
            semantics = solver.eval(population, return_outputs="tensor").outputs
            np_semantics = semantics.cpu().numpy()
            del semantics
            convex_hull = ConvexHull(np_semantics)
            vertex_ids = convex_hull.vertices
            ci_population = [population[vid] for vid in vertex_ids]
            target_inside_hull = self.is_point_inside_hull(target, convex_hull)
        res = ci_population
        self.metrics["target_inside_hull"] = target_inside_hull
        # res = ci_population[:pop_size]
        return res
