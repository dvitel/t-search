from typing import Callable
import numpy as np
from scipy.spatial import ConvexHull
import torch

from t_search.syntax import Term

from .sdi import SDI

class CI(SDI):
    """Competent semantic initialization"""

    def __init__(self, *, 
                 add_metrics: Callable,
                 target: torch.Tensor, 
                 size: int = 1000,
                 **kwargs):
        super().__init__(**kwargs)
        self.target = target.cpu().numpy()
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
            population = super().__call__(i * self.size)
            semantics = self.evaluator.eval(population, return_outputs="tensor").outputs
            np_semantics = semantics.cpu().numpy()
            del semantics
            convex_hull = ConvexHull(np_semantics)
            vertex_ids = convex_hull.vertices
            ci_population = [population[vid] for vid in vertex_ids]
            target_inside_hull = self.is_point_inside_hull(self.target, convex_hull)
        res = ci_population
        self.add_metrics(target_inside_hull = [1 if target_inside_hull else 0])
        # res = ci_population[:size]
        return res
