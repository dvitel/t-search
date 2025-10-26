import torch
from .sampling import get_interval_grid, get_rand_points
from .benchmark import Benchmark


test_0 = Benchmark("test_0", lambda x: x + 74.3, get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

koza_1 = Benchmark(
    "koza_1",
    lambda x: x * x * x * x + x * x * x + x * x + x,
    get_rand_points,
    {"num_samples": 20, "ranges": [(-1.0, 1.0)]},
)

koza_2 = Benchmark(
    "koza_2",
    lambda x: x * x * x * x * x - 2.0 * x * x * x + x,
    get_rand_points,
    {"num_samples": 20, "ranges": [(-1.0, 1.0)]},
)


koza_3 = Benchmark(
    "koza_3",
    lambda x: x * x * x * x * x * x - 2.0 * x * x * x * x + x * x,
    get_rand_points,
    {"num_samples": 20, "ranges": [(-1.0, 1.0)]},
)

nguyen_1 = Benchmark(
    "nguyen_1", lambda x: x * x * x + x * x + x, get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]}
)


nguyen_2 = Benchmark(
    "nguyen_2",
    lambda x: x * x * x * x + x * x * x + x * x + x,
    get_rand_points,
    {"num_samples": 20, "ranges": [(-1.0, 1.0)]},
)

nguyen_3 = Benchmark(
    "nguyen_3",
    lambda x: x * x * x * x * x + x * x * x * x + x * x + x,
    get_rand_points,
    {"num_samples": 20, "ranges": [(-1.0, 1.0)]},
)

nguyen_4 = Benchmark(
    "nguyen_4",
    lambda x: x * x * x * x * x * x + x * x * x * x * x + x * x * x * x + x * x * x + x * x + x,
    get_rand_points,
    {"num_samples": 20, "ranges": [(-1.0, 1.0)]},
)

nguyen_5 = Benchmark(
    "nguyen_5",
    lambda x: torch.sin(x * x) * torch.cos(x) - 1.0,
    get_rand_points,
    {"num_samples": 20, "ranges": [(-1.0, 1.0)]},
)

nguyen_6 = Benchmark(
    "nguyen_6",
    lambda x: torch.sin(x) + torch.sin(x + x * x),
    get_rand_points,
    {"num_samples": 20, "ranges": [(-1.0, 1.0)]},
)

nguyen_7 = Benchmark(
    "nguyen_7",
    lambda x: torch.log(x + 1.0) + torch.log(x * x + 1.0),
    get_rand_points,
    {"num_samples": 20, "ranges": [(0.0, 2.0)]},
)

nguyen_8 = Benchmark("nguyen_8", lambda x: torch.sqrt(x), get_rand_points, {"num_samples": 20, "ranges": [(0.0, 4.0)]})

nguyen_9 = Benchmark(
    "nguyen_9",
    lambda x, y: torch.sin(x) + torch.sin(y * y),
    get_rand_points,
    {"num_samples": 100, "ranges": [(0.0, 1.0), (0.0, 1.0)]},
)

nguyen_10 = Benchmark(
    "nguyen_10",
    lambda x, y: 2.0 * torch.sin(x) + torch.cos(y),
    get_rand_points,
    {"num_samples": 100, "ranges": [(0.0, 1.0), (0.0, 1.0)]},
)

pagie_1 = Benchmark(
    "pagie_1",
    lambda x, y: 1.0 / (1.0 + 1.0 / (x * x * x * x)) + 1.0 / (1.0 + 1.0 / (y * y * y * y)),
    get_interval_grid,
    {"steps": 0.4, "ranges": [[-5.0, 5.0], [-5.0, 5.0]]},
)

pagie_2 = Benchmark(
    "pagie_2",
    lambda x, y, z: 1.0 / (1.0 + 1.0 / (x * x * x * x))
    + 1.0 / (1.0 + 1.0 / (y * y * y * y))
    + 1.0 / (1.0 + 1.0 / (z * z * z * z)),
    get_interval_grid,
    {"steps": 0.4, "ranges": [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]},
)

korns_sampling = get_rand_points, {
    "num_samples": 10000,
    "ranges": [[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]],
}

korns_1 = Benchmark("korns_1", lambda *xs: 1.57 + (24.3 * xs[3]), *korns_sampling)
korns_2 = Benchmark("korns_2", lambda *xs: 0.23 + (14.2 * ((xs[3] + xs[1]) / (3.0 * xs[4]))), *korns_sampling)
korns_3 = Benchmark(
    "korns_3", lambda *xs: -5.41 + (4.9 * (((xs[3] - xs[0]) + (xs[1] / xs[4])) / (3 * xs[4]))), *korns_sampling
)
korns_4 = Benchmark("korns_4", lambda *xs: -2.3 + (0.13 * torch.sin(xs[2])), *korns_sampling)
korns_5 = Benchmark("korns_5", lambda *xs: 3.0 + (2.13 * torch.log(xs[4])), *korns_sampling)
korns_6 = Benchmark("korns_6", lambda *xs: 1.3 + (0.13 * torch.sqrt(xs[0])), *korns_sampling)
korns_7 = Benchmark(
    "korns_7", lambda *xs: 213.80940889 - (213.80940889 * torch.exp(-0.54723748542 * xs[0])), *korns_sampling
)
korns_8 = Benchmark("korns_8", lambda *xs: 6.87 + (11.0 * torch.sqrt(7.23 * xs[0] * xs[3] * xs[4])), *korns_sampling)
korns_9 = Benchmark(
    "korns_9", lambda *xs: torch.sqrt(xs[0]) / torch.log(xs[1]) * torch.exp(xs[2]) / (xs[3] * xs[3]), *korns_sampling
)
korns_10 = Benchmark(
    "korns_10",
    lambda *xs: 0.81
    + (
        24.3
        * (
            ((2.0 * xs[1]) + (3.0 * (xs[2] * xs[2])))
            / ((4.0 * (xs[3] * xs[3] * xs[3])) + (5.0 * (xs[4] * xs[4] * xs[4] * xs[4])))
        )
    ),
    *korns_sampling,
)
korns_11 = Benchmark("korns_11", lambda *xs: 6.87 + (11.0 * torch.cos(7.23 * xs[0] * xs[0] * xs[0])), *korns_sampling)
korns_12 = Benchmark(
    "korns_12", lambda *xs: 2.0 - (2.1 * (torch.cos(9.8 * xs[0]) * torch.sin(1.3 * xs[4]))), *korns_sampling
)
korns_13 = Benchmark(
    "korns_13",
    lambda *xs: 32.0 - (3.0 * ((torch.tan(xs[0]) / torch.tan(xs[1])) * (torch.tan(xs[2]) / torch.tan(xs[3])))),
    *korns_sampling,
)
korns_14 = Benchmark(
    "korns_14",
    lambda *xs: 22.0 - (4.2 * ((torch.cos(xs[0]) - torch.tan(xs[1])) * (torch.tanh(xs[2]) / torch.sin(xs[3])))),
    *korns_sampling,
)
korns_15 = Benchmark(
    "korns_15",
    lambda *xs: 12.0 - (6.0 * ((torch.tan(xs[0]) / torch.exp(xs[1])) * (torch.log(xs[2]) - torch.tan(xs[3])))),
    *korns_sampling,
)

keijzer_1 = Benchmark(
    "keijzer_1",
    lambda x: 0.3 * x * torch.sin(2.0 * torch.pi * x),
    get_interval_grid,
    {"steps": 0.1, "ranges": [[-1.0, 1.0]]},
    get_interval_grid,
    {"steps": 0.001, "ranges": [[-1.0, 1.0]]},
)

keijzer_2 = Benchmark(
    "keijzer_2",
    lambda x: 0.3 * x * torch.sin(2.0 * torch.pi * x),
    get_interval_grid,
    {"steps": 0.1, "ranges": [[-2.0, 2.0]]},
    get_interval_grid,
    {"steps": 0.001, "ranges": [[-2.0, 2.0]]},
)

keijzer_3 = Benchmark(
    "keijzer_3",
    lambda x: 0.3 * x * torch.sin(2.0 * torch.pi * x),
    get_interval_grid,
    {"steps": 0.1, "ranges": [[-3.0, 3.0]]},
    get_interval_grid,
    {"steps": 0.001, "ranges": [[-3.0, 3.0]]},
)

keijzer_4 = Benchmark(
    "keijzer_4",
    lambda x: x * torch.exp(-x) * torch.cos(x) * torch.sin(x) * (torch.sin(x) * torch.sin(x) * torch.cos(x) - 1),
    get_interval_grid,
    {"steps": 0.05, "ranges": [[0.0, 10.0]]},
    get_interval_grid,
    {"steps": 0.05, "ranges": [[0.05, 10.05]]},
)

keijzer_5 = Benchmark(
    "keijzer_5",
    lambda x, y, z: (30.0 * x * z) / ((x - 10.0) * y * y),
    get_rand_points,
    {"num_samples": 1000, "ranges": [[-1.0, 1.0], [1.0, 2.0], [-1.0, 1.0]]},
    get_rand_points,
    {"num_samples": 10000, "ranges": [[-1.0, 1.0], [1.0, 2.0], [-1.0, 1.0]]},
)

keijzer_6 = Benchmark(
    "keijzer_6",
    lambda *xs: torch.stack([torch.sum(1.0 / torch.arange(1, torch.floor(x) + 1)) for x in xs]),
    get_interval_grid,
    {"steps": 1.0, "ranges": [[1.0, 50.0]]},
    get_interval_grid,
    {"steps": 1.0, "ranges": [[1.0, 120.0]]},
)

keijzer_7 = Benchmark(
    "keijzer_7",
    lambda x: torch.log(x),
    get_interval_grid,
    {"steps": 1.0, "ranges": [[1.0, 100.0]]},
    get_interval_grid,
    {"steps": 0.1, "ranges": [[1.0, 100.0]]},
)

keijzer_8 = Benchmark(
    "keijzer_8",
    lambda x: torch.sqrt(x),
    get_interval_grid,
    {"steps": 1.0, "ranges": [[0.0, 100.0]]},
    get_interval_grid,
    {"steps": 0.1, "ranges": [[0.0, 100.0]]},
)

keijzer_9 = Benchmark(
    "keijzer_9",
    lambda x: torch.arcsinh(x),
    get_interval_grid,
    {"steps": 1.0, "ranges": [[0.0, 100.0]]},
    get_interval_grid,
    {"steps": 0.1, "ranges": [[0.0, 100.0]]},
)

keijzer_10 = Benchmark(
    "keijzer_10",
    lambda x, y: torch.float_power(x, y),
    get_rand_points,
    {"num_samples": 100, "ranges": [[0.0, 1.0], [0.0, 1.0]]},
    get_interval_grid,
    {"steps": 0.01, "ranges": [[0.0, 1.0], [0.0, 1.0]]},
)


keijzer_11 = Benchmark(
    "keijzer_11",
    lambda x, y: x * y + torch.sin((x - 1.0) * (y - 1.0)),
    get_rand_points,
    {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
    get_interval_grid,
    {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
)

keijzer_12 = Benchmark(
    "keijzer_12",
    lambda x, y: x * x * x * x - x * x * x + y * y / 2.0 - y,
    get_rand_points,
    {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
    get_interval_grid,
    {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
)

keijzer_13 = Benchmark(
    "keijzer_13",
    lambda x, y: 6.0 * torch.sin(x) * torch.cos(y),
    get_rand_points,
    {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
    get_interval_grid,
    {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
)

keijzer_14 = Benchmark(
    "keijzer_14",
    lambda x, y: 8.0 / (2.0 + x * x + y * y),
    get_rand_points,
    {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
    get_interval_grid,
    {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
)

keijzer_15 = Benchmark(
    "keijzer_15",
    lambda x, y: x * x * x / 5.0 + y * y * y / 2.0 - y - x,
    get_rand_points,
    {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
    get_interval_grid,
    {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
)

vladislavleva_1 = Benchmark(
    "vladislavleva_1",
    lambda x, y: torch.exp(-(x - 1) * (x - 1)) / (1.2 + (y - 2.5) * (y - 2.5)),
    get_rand_points,
    {"num_samples": 100, "ranges": [[0.3, 4.0], [0.3, 4.0]]},
    get_interval_grid,
    {"steps": 0.1, "ranges": [[-0.2, 4.2], [-0.2, 4.2]]},
)

vladislavleva_2 = Benchmark(
    "vladislavleva_2",
    lambda x: torch.exp(-x) * x * x * x * torch.cos(x) * torch.sin(x) * (torch.cos(x) * torch.sin(x) * torch.sin(x) - 1),
    get_interval_grid,
    {"steps": 0.1, "ranges": [[0.05, 10]]},
    get_interval_grid,
    {"steps": 0.05, "ranges": [[-0.5, 10.5]]},
)

vladislavleva_3 = Benchmark(
    "vladislavleva_3",
    lambda x, y: torch.exp(-x)
    * x
    * x
    * x
    * torch.cos(x)
    * torch.sin(x)
    * (torch.cos(x) * torch.sin(x) * torch.sin(x) - 1)
    * (y - 5),
    get_interval_grid,
    {"steps": [0.1, 2.0], "ranges": [[0.05, 10], [0.05, 10.05]]},
    get_interval_grid,
    {"steps": [0.05, 0.5], "ranges": [[-0.5, 10.5], [-0.5, 10.5]]},
)

vladislavleva_4 = Benchmark(
    "vladislavleva_4",
    lambda *xs: 10.0 / (5.0 + torch.sum((xs - 3.0) ** 2, axis=0)),
    get_rand_points,
    {"num_samples": 1024, "ranges": [[0.05, 6.05]] * 5},
    get_rand_points,
    {"num_samples": 5000, "ranges": [[-0.25, 6.35]] * 5},
)


vladislavleva_5 = Benchmark(
    "vladislavleva_5",
    lambda x, y, z: (30.0 * (x - 1.0) * (z - 1.0)) / (y * y * (x - 10.0)),
    get_rand_points,
    {"num_samples": 300, "ranges": [[0.05, 2.0], [1.0, 2.0], [0.05, 2.0]]},
    get_interval_grid,
    {"steps": [0.15, 0.15, 0.1], "ranges": [[-0.05, 2.1], [0.95, 2.05], [-0.05, 2.1]]},
)

vladislavleva_6 = Benchmark(
    "vladislavleva_6",
    lambda x, y: 6.0 * torch.sin(x) * torch.cos(y),
    get_rand_points,
    {"num_samples": 30, "ranges": [[0.1, 5.9], [0.1, 5.9]]},
    get_interval_grid,
    {"steps": [0.02, 0.02], "ranges": [[-0.05, 6.05], [-0.05, 6.05]]},
)


vladislavleva_7 = Benchmark(
    "vladislavleva_7",
    lambda x, y: (x - 3.0) * (y - 3.0) + 2 * torch.sin((x - 4.0) * (y - 4.0)),
    get_rand_points,
    {"num_samples": 300, "ranges": [[0.05, 6.05], [0.05, 6.05]]},
    get_rand_points,
    {"num_samples": 1000, "ranges": [[-0.25, 6.35], [-0.25, 6.35]]},
)


vladislavleva_8 = Benchmark(
    "vladislavleva_8",
    lambda x, y: ((x - 3.0) * (x - 3.0) * (x - 3.0) * (x - 3.0) + (y - 3.0) * (y - 3.0) * (y - 3.0) - (y - 3.0))
    / ((y - 2.0) * (y - 2.0) * (y - 2.0) * (y - 2.0) + 10.0),
    get_rand_points,
    {"num_samples": 50, "ranges": [[0.05, 6.05], [0.05, 6.05]]},
    get_interval_grid,
    {"steps": [0.2, 0.2], "ranges": [[-0.25, 6.35], [-0.25, 6.35]]},
)


# all_benchmarks = [
#     koza_1,
#     koza_2,
#     koza_3,
#     nguyen_1,
#     nguyen_2,
#     nguyen_3,
#     nguyen_4,
#     nguyen_5,
#     nguyen_6,
#     nguyen_7,
#     nguyen_8,
#     nguyen_9,
#     nguyen_10,
#     pagie_1,
#     pagie_2,
#     korns_1,
#     korns_2,
#     korns_3,
#     korns_4,
#     korns_5,
#     korns_6,
#     korns_7,
#     korns_8,
#     korns_9,
#     korns_10,
#     korns_11,
#     korns_12,
#     korns_13,
#     korns_14,
#     korns_15,
#     keijzer_1,
#     keijzer_2,
#     keijzer_3,
#     keijzer_4,
#     keijzer_5,
#     keijzer_6,
#     keijzer_7,
#     keijzer_8,
#     keijzer_9,
#     keijzer_10,
#     keijzer_11,
#     keijzer_12,
#     keijzer_13,
#     keijzer_14,
#     keijzer_15,
#     vladislavleva_1,
#     vladislavleva_2,
#     vladislavleva_3,
#     vladislavleva_4,
#     vladislavleva_5,
#     vladislavleva_6,
#     vladislavleva_7,
#     vladislavleva_8,
# ]
