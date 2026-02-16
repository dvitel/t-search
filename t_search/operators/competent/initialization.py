from typing import Callable
import numpy as np
from scipy.spatial import ConvexHull
import torch

from t_search.operators.semantic.initialization import SemanticallyDrivenInitialization
from t_search.syntax import Term

# NOTE: does not currently work - ConvexHull is too expensive - just resort to SemanticallyDrivenInitialization

class CompetentInitialization(SemanticallyDrivenInitialization):
    """Competent semantic initialization"""

    def __init__(self, *, 
                 add_metrics: Callable,
                 target: torch.Tensor, 
                 size: int = 1000,
                 **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.target_cpu = target.cpu().numpy()
        self.add_metrics = add_metrics
        self.size = size

    def is_point_inside_hull(
        self, point: np.ndarray, hull: ConvexHull, tolerance: float = 1e-12
    ) -> bool:
        results = np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1]
        return np.all(results <= tolerance).item()

    def __call__(self) -> list[Term]:
        ci_population: list[Term] = []
        max_try_count = 3
        i = 0
        target_inside_hull = False
        while len(ci_population) < self.size and (i < max_try_count):
            i += 1
            population = super().__call__(size=i * self.size)
            self.evaluator.eval(population)
            semantics = self.semantics.get_outputs(population, return_type="tensor")
            # first, filter by distance to target
            l2 = ((semantics.unsqueeze(1) - self.target.unsqueeze(0)) ** 2).sum(dim=-1)
            close_mask  = l2 < 1e6
            filtered_ids = torch.where(close_mask)[0]
            filtered_semantics = semantics[filtered_ids]
            filtered_population = [population[i] for i in filtered_ids.tolist()]
            pass

            np_semantics = filtered_semantics.cpu().numpy()
            del semantics, filtered_semantics
            convex_hull = ConvexHull(np_semantics) 
            vertex_ids = convex_hull.vertices
            ci_population = [filtered_population[vid] for vid in vertex_ids]
            target_inside_hull = self.is_point_inside_hull(self.target_cpu, convex_hull)
        res = ci_population
        self.add_metrics(target_inside_hull = [1 if target_inside_hull else 0])
        # res = ci_population[:size]
        return res
