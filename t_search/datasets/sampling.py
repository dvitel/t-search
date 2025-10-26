from typing import Optional
import torch

def lexsort(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() == 2, "Input tensor must be 2D"
    k, n = tensor.shape

    sorted_indices = torch.argsort(tensor[-1, :])

    for row in range(k - 2, -1, -1):
        sorted_tensor = tensor[:, sorted_indices]
        sorted_indices = sorted_indices[torch.argsort(sorted_tensor[row, :])]

    return sorted_indices


# tensor = torch.tensor([[2, 2, 1, 2], [2, 1, 2, 1], [1, 2, 4, 3]], dtype=torch.float32)
# sorted_indices = lexsort(tensor)
# pass

# tests1 = torch.meshgrid([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])], indexing='ij')
# test1 = torch.stack(tests1, dim=-1)
# pass


def get_full_grid(grid_values: list[torch.Tensor]) -> torch.Tensor:
    """grid_values - per each dimension/variable, specifies allowed values for each dimension"""
    assert len(grid_values) > 0, "Grid values should not be empty"
    meshes = torch.meshgrid(*[v for v in grid_values], indexing="ij")
    grid_nd = torch.stack(meshes, dim=-1)
    grid = grid_nd.reshape(-1, grid_nd.shape[-1])
    return grid


# t1 = get_full_grid([torch.tensor([1,2]), torch.tensor([1,2]), torch.tensor([4,5])])
# pass


def get_rand_grid_point(grid_values: list[torch.Tensor], *, generator: torch.Generator | None = None) -> torch.Tensor:
    assert len(grid_values) > 0, "Grid values should not be empty"
    values = [v[torch.randint(0, len(v), (1,), device=v.device, generator=generator)] for v in grid_values]
    stacked = torch.cat(values, dim=0)
    return stacked


# t2 = get_rand_grid_point([torch.tensor([1,2]), torch.tensor([1,2]), torch.tensor([4,5])])
# pass


def get_rand_points(num_samples: int, ranges: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """ranges: tensor 1d - free var, 2d - [min, max]
    return rand sample of values in ranges"""
    mins = ranges[:, 0]
    maxs = ranges[:, 1]
    dist = maxs - mins
    values = mins[:, torch.newaxis] + dist[:, torch.newaxis] * torch.rand(
        len(ranges), num_samples, device=ranges.device, generator=generator
    )
    return values


# t3 = get_rand_points(10, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass


def get_rand_full_grid(num_samples, ranges: torch.Tensor) -> torch.Tensor:
    """ranges: tensor 1d - free var, 2d - [min, max]
    From rand sample per dimension builds full grid
    """
    grid_values = get_rand_points(num_samples, ranges)
    grid = get_full_grid(grid_values)
    return grid


# t4 = get_rand_full_grid(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass


def get_interval_points(
    steps: torch.Tensor | float,
    ranges: torch.Tensor,
    deltas: Optional[torch.Tensor] = None,
    rand_deltas=False,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:

    # mins = ranges[:, 0]
    # maxs = ranges[:, 1]
    # steps = (maxs - mins) / num_samples_per_dim
    if not torch.is_tensor(steps):
        steps = torch.full_like(ranges[:, 0], steps)
    if rand_deltas:
        if deltas is None:
            deltas = steps.clone()
        deltas *= torch.rand(ranges.shape[0], device=ranges.device, generator=generator)
    if deltas is None:
        deltas = torch.zeros_like(steps)
    values = [torch.arange(r[0] + d, r[1], s, device=r.device, dtype=r.dtype) for r, s, d in zip(ranges, steps, deltas)]
    return values


# t5 = get_interval_points(0.5, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass


def get_interval_grid(
    steps: torch.Tensor | float,
    ranges: torch.Tensor,
    deltas: Optional[torch.Tensor] = None,
    rand_deltas=False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    grid_values = get_interval_points(steps, ranges, deltas, rand_deltas, generator)
    grid = get_full_grid(grid_values)
    return grid


# t6 = get_interval_grid(0.5, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass


def get_rand_interval_points(
    num_samples: int,
    ranges: torch.Tensor,
    steps: Optional[torch.Tensor | float] = None,
    deltas: Optional[torch.Tensor] = None,
    rand_deltas=True,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    if steps is None:
        steps = (ranges[:, 1] - ranges[:, 0]) / (num_samples + 1)
    grid_values = get_interval_points(steps, ranges, deltas, rand_deltas, generator=generator)
    points = [get_rand_grid_point(grid_values, generator=generator) for _ in range(num_samples)]
    # points = torch.stack(points, dim=0)
    return points


# t7 = get_rand_interval_points(10, torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), 0.5)
# pass


# https://en.wikipedia.org/wiki/Chebyshev_nodes
def get_chebyshev_points(
    num_samples, ranges: torch.Tensor, rand_deltas=False, generator: torch.Generator | None = None
) -> torch.Tensor:
    assert num_samples > 0, "Number of samples should be greater than 1"
    mins = ranges[:, 0]
    maxs = ranges[:, 1]
    dist = maxs - mins
    indexes = torch.arange(
        1, num_samples + 1, dtype=float, device=ranges.device
    )  # torch.tile(torch.arange(0, num_samples), (ranges.shape[0], 1))
    if rand_deltas:
        indexes = torch.rand(ranges.shape[0], device=ranges.device, generator=generator)[:, torch.newaxis] + indexes
    else:
        indexes = torch.zeros(ranges.shape[0], device=ranges.device)[:, torch.newaxis] + indexes
    index_vs = torch.cos((2.0 * indexes - 1) / (2.0 * num_samples) * torch.pi)
    values = (maxs[:, torch.newaxis] + mins[:, torch.newaxis]) / 2 + dist[:, torch.newaxis] / 2 * index_vs
    return values


# t8 = get_chebyshev_points(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass


def get_chebyshev_grid(num_samples, ranges: torch.Tensor, rand_deltas=False):
    grid_values = get_chebyshev_points(num_samples, ranges, rand_deltas)
    grid = get_full_grid(grid_values)
    return grid


# t9 = get_chebyshev_grid(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass


def get_rand_chebyshev_points(
    num_samples, ranges: torch.Tensor, num_samples_per_dim: torch.Tensor | int = 0, rand_deltas=True
):
    num_samples_per_dim = num_samples if num_samples_per_dim == 0 else num_samples_per_dim
    grid_values = get_chebyshev_points(num_samples_per_dim, ranges, rand_deltas)
    points = [get_rand_grid_point(grid_values) for _ in range(num_samples)]
    points = torch.stack(points, dim=0)
    return points


# t10 = get_rand_chebyshev_points(20, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass
