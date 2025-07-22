
from dataclasses import dataclass, field
from functools import partial
import inspect
from typing import Any, Callable, Literal, Optional
import torch

from term import Term, cache_term, dict_alloc_id, evaluate, get_leaves, parse_term

alg_ops = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "pow": lambda a, b: a ** b,
    "neg": lambda a: -a,
    "inv": lambda a: 1 / a,
    "exp": lambda a: torch.exp(a),
    "log": lambda a: torch.log(a),
    "sin": lambda a: torch.sin(a),
    "cos": lambda a: torch.cos(a),
}

# tests1 = torch.meshgrid([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])], indexing='ij')
# test1 = torch.stack(tests1, dim=-1)
# pass 

def get_full_grid(grid_values: list[torch.Tensor]) -> torch.Tensor:
    ''' grid_values - per each dimension/variable, specifies allowed values for each dimension '''
    assert len(grid_values) > 0, "Grid values should not be empty"
    meshes = torch.meshgrid(*[v for v in grid_values], indexing='ij')
    grid_nd = torch.stack(meshes, dim=-1)
    grid = grid_nd.reshape(-1, grid_nd.shape[-1])
    return grid 

# t1 = get_full_grid([torch.tensor([1,2]), torch.tensor([1,2]), torch.tensor([4,5])])
# pass 

def get_rand_grid_point(grid_values: list[torch.Tensor], 
                        *, generator: torch.Generator | None = None) -> torch.Tensor:
    assert len(grid_values) > 0, "Grid values should not be empty"
    values = [v[torch.randint(0, len(v), (1,), generator=generator)] for v in grid_values]
    stacked = torch.cat(values, dim=0)
    return stacked

# t2 = get_rand_grid_point([torch.tensor([1,2]), torch.tensor([1,2]), torch.tensor([4,5])])
# pass

def get_rand_points(num_samples: int, ranges: torch.Tensor,
                        *, generator: torch.Generator | None = None) -> torch.Tensor:
    ''' ranges: tensor 1d - free var, 2d - [min, max] 
        return rand sample of values in ranges '''
    mins = ranges[:, 0]
    maxs = ranges[:, 1]
    dist = maxs - mins
    values = mins[:, torch.newaxis] + dist[:, torch.newaxis] * torch.rand(len(ranges), num_samples, device=ranges.device,
                                                                            generator = generator)
    return values

# t3 = get_rand_points(10, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_rand_full_grid(num_samples, ranges: torch.Tensor) -> torch.Tensor:
    ''' ranges: tensor 1d - free var, 2d - [min, max] 
        From rand sample per dimension builds full grid
    '''
    grid_values = get_rand_points(num_samples, ranges)
    grid = get_full_grid(grid_values)
    return grid

# t4 = get_rand_full_grid(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_interval_points(steps: torch.Tensor | float, ranges: torch.Tensor, 
                        deltas: Optional[torch.Tensor] = None, rand_deltas = False,
                        generator: torch.Generator | None = None) -> list[torch.Tensor]:
    
    
    # mins = ranges[:, 0]
    # maxs = ranges[:, 1]
    # steps = (maxs - mins) / num_samples_per_dim
    if not torch.is_tensor(steps): 
        steps = torch.full_like(ranges[:, 0], steps)
    if rand_deltas:
        deltas = deltas or steps
        deltas *= torch.rand(ranges.shape[0], device=ranges.device, generator = generator)
    if deltas is None:
        deltas = torch.zeros_like(steps)
    values = [torch.arange(r[0] + d, r[1], s) for r, s, d in zip(ranges, steps, deltas)]
    return values

# t5 = get_interval_points(0.5, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_interval_grid(steps: torch.Tensor | float, ranges: torch.Tensor, 
                      deltas: Optional[torch.Tensor] = None, rand_deltas = False) -> torch.Tensor:
    grid_values = get_interval_points(steps, ranges, deltas, rand_deltas)
    grid = get_full_grid(grid_values)
    return grid

# t6 = get_interval_grid(0.5, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_rand_interval_points(num_samples: int, ranges: torch.Tensor, 
                             steps: Optional[torch.Tensor | float] = None, deltas: Optional[torch.Tensor] = None, rand_deltas = True) -> torch.Tensor:
    if steps is None:
        steps = (ranges[:, 1] - ranges[:, 0]) / num_samples
    grid_values = get_interval_points(steps, ranges, deltas, rand_deltas)
    points = [get_rand_grid_point(grid_values) for _ in range(num_samples)]
    points = torch.stack(points, dim=0)
    return points

# t7 = get_rand_interval_points(10, torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), 0.5)
# pass

# https://en.wikipedia.org/wiki/Chebyshev_nodes
def get_chebyshev_points(num_samples, ranges: torch.Tensor, rand_deltas = False,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    assert num_samples > 0, "Number of samples should be greater than 1"
    mins = ranges[:, 0]
    maxs = ranges[:, 1]
    dist = maxs - mins
    indexes = torch.arange(1, num_samples + 1, dtype=float, device=ranges.device) #torch.tile(torch.arange(0, num_samples), (ranges.shape[0], 1))
    if rand_deltas:
        indexes = torch.rand(ranges.shape[0], device=ranges.device, generator=generator)[:, torch.newaxis] + indexes
    else:
        indexes = torch.zeros(ranges.shape[0], device=ranges.device)[:, torch.newaxis] + indexes
    index_vs = torch.cos((2.0 * indexes - 1) / (2.0 * num_samples) * torch.pi)
    values = (maxs[:, torch.newaxis] + mins[:, torch.newaxis]) / 2 + dist[:, torch.newaxis] / 2 * index_vs
    return values

# t8 = get_chebyshev_points(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_chebyshev_grid(num_samples, ranges: torch.Tensor, rand_deltas = False):
    grid_values = get_chebyshev_points(num_samples, ranges, rand_deltas)
    grid = get_full_grid(grid_values)
    return grid

# t9 = get_chebyshev_grid(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_rand_chebyshev_points(num_samples, ranges: torch.Tensor, num_samples_per_dim: torch.Tensor | int = 0, rand_deltas = True):
    num_samples_per_dim = num_samples if num_samples_per_dim == 0 else num_samples_per_dim
    grid_values = get_chebyshev_points(num_samples_per_dim, ranges, rand_deltas)
    points = [get_rand_grid_point(grid_values) for _ in range(num_samples)]
    points = torch.stack(points, dim=0)
    return points

# t10 = get_rand_chebyshev_points(20, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def lin_comb_value_builder(term: Term, semantics: torch.Tensor, 
                           leaf_semantic_ids: dict[Term, int], 
                           branch_semantic_ids: dict[Term, int],
                           const_ranges: torch.Tensor, num_points: int, 
                           sample_strategy: Callable = get_rand_interval_points):
    '''
        Represents symbol x as linear combination of free variables and constant - one W per symbol position in term
        sample_grid_strategy - one of get_rand_chebyshev_points, get_rand_interval_points, get_rand_points
        do not use full grid - too expensive
        free_vars - tensor for variables x, y, z, etc and constant 1.
        Convension - constant is first in semantics - should be reflected in free_vars and ranges

        Goal is to find W - cocnstants that would bring us to target semantics
        term is ignored - all vars and constants are samely repreesented
    '''
    # TODO: ranges - are const ranges always - except of original free var grid 
    if len(term.args) == 0: # leaf 
        semantic_ids = [*leaf_semantic_ids.values()]
    else: # branch - experimental - idea is to replace args A under (f A) by (f W * A), only applicable if all branches are evaluated
        semantic_ids = [0, *(branch_semantic_ids.get(arg, None) for arg in term.args)]
        assert all(bid is not None for bid in semantic_ids), "All branch semantic ids should be defined"
    ranges = torch.tile(const_ranges, (len(semantic_ids), 1))
    W = sample_strategy(num_points, ranges) #what should we use as ranges here?
    W.requires_grad = True # we optimize this tensor    
    X = semantics[semantic_ids]
    def term_op(W = W, X = X):
        output = torch.matmul(W, X) # W = 20 x 4, X = 4 x 256 ==> W * X = 20 x 256
        return output 
    return (W, term_op)

def const_value_builder(term: Term, semantics: torch.Tensor, 
                        leaf_semantic_ids: dict[Term, int], 
                        branch_semantic_ids: dict[Term, int], 
                        const_ranges: torch.Tensor, num_points: int, 
                        sample_strategy: Callable = get_rand_interval_points):
    '''
        Represents constant c symbol only as weight. 
        Variables 
    '''
    if len(term.args) == 0: # leaf
        semantic_ids = [leaf_semantic_ids[term]]
    else: # branch - scale branch result
        semantic_ids = [branch_semantic_ids[term]]
    W = sample_strategy(num_points, const_ranges[torch.newaxis, :]) # ranges for constant
    W.requires_grad = True # we optimize this tensor
    X = semantics[semantic_ids]
    def term_op(W = W, X = X):
        output = torch.matmul(W, X)
        return output
    return (W, term_op)

def default_leaf_op(term: Term):
    X = semantics[leaf_semantic_ids[term]]
    return X

def lbfgs_optimize(term: Term, gold_outputs: torch.Tensor, semantics: torch.Tensor, 
                leaf_semantic_ids: dict[Term, int],
                branch_semantic_ids: dict[Term, int],
                const_ranges: torch.Tensor,
                epsilon: float = 1e-4, num_points = 10, num_steps = 20,
                value_builder = lin_comb_value_builder,
                sample_strategy = get_rand_interval_points,
                leaves_cache = {}):
    ''' Searches for tensor values to bring term closer to gold outputs 
        Initial bindings should be already created
    '''

    leaves = get_leaves(term, leaves_cache=leaves_cache) # all adjustable leaves 
    Ws_ops = [value_builder(leaf, semantics, leaf_semantic_ids, branch_semantic_ids,
                        const_ranges, num_points, sample_strategy) for leaf in leaves[-2:]]

    Ws, ops = zip(*Ws_ops) # Ws - list of leaf tensors, Opts - list of bindings at term leaf
    # bindings = bind_terms(leaves, list(values))

    leaf_ops = {(leaf, len(leaves) - 2 + leaf_id): op for leaf_id, (leaf, op) in enumerate(zip(leaves, ops))}
    other_ops = {(leaf, leaf_id): partial(default_leaf_op, leaf) for leaf_id, leaf in enumerate(leaves[:-2])}

    last_bindings = None
    last_outputs = None
    last_errors = None

    def closure_builder(optimizer):
        nonlocal last_outputs, last_errors, last_bindings
        optimizer.zero_grad()        
        last_bindings = {**other_ops, **leaf_ops}
        outputs = evaluate(term, alg_ops, last_bindings, last)
        assert outputs is not None, "Term evaluation should be full. Term is evaluated partially"
        last_outputs = outputs
        loss_els = (outputs - gold_outputs) ** 2
        loss = torch.sum(loss_els, dim=-1) # 1d tensor per batch element
        # del loss_els
        last_errors = loss
        # loss = mse_loss(outputs, gold_outputs)
        loss.backward(gradient=torch.ones_like(loss))
        total_loss = loss.median()
        return total_loss

    optimizer = torch.optim.LBFGS(Ws, lr=1, max_iter=20,
                                    # max_eval = 1.5 * num_steps,
                                    # tolerance_change=epsilon,
                                    # tolerance_grad=epsilon / 10,
                                    # history_size=100,
                                    # line_search_fn='strong_wolfe'
                                    )

    closure = partial(closure_builder, optimizer)
            
    total_loss = optimizer.step(closure)
    print(f"LBFGS optimization finished with loss {total_loss.item()}")
    # res = evaluate(term, all_ops, {})
    pass

class Benchmark: 

    def __init__(self, name: str | None, fn: Callable,
                 train_sampling: Callable = get_rand_points, 
                 train_args: dict[str, Any] = None,
                 test_sampling: Optional[Callable] = None, 
                 test_args: Optional[dict[str, Any]] = None):
        self.name = name or fn.__name__
        self.fn = fn
        self.train_sampling: Callable = train_sampling
        self.train_args: dict[str, Any] = train_args or {}
        self.test_sampling: Optional[Callable] = test_sampling
        self.test_args: Optional[dict[str, Any]] = test_args
        self.sampled = {}

    def with_train_sampling(self, train_sampling = None, **kwargs):
        return Benchmark(self.name, self.fn, train_sampling or self.train_sampling, kwargs)
    
    def with_test_sampling(self, test_sampling = None, **kwargs):
        return Benchmark(self.name, self.fn, self.train_sampling, self.train_args,
                         test_sampling or self.test_sampling, kwargs)

    def sample_set(self, set_name: Literal["train", "test"], 
                            device = "cpu", dtype = torch.float32,
                            generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if set_name in self.sampled:
            return self.sampled[set_name]
        if set_name == "test" and self.test_sampling is None:
            return self.sample_set("train", device)
        sample_args = self.train_args if set_name == "train" else self.test_args
        prepared_args = {k:(torch.tensor(v, device = device, dtype=dtype) if type(v) is list else v) for k, v in sample_args.items()}
        sample_fn = self.train_sampling if set_name == "train" else self.test_sampling
        signature = inspect.signature(sample_fn)
        if 'generator' in signature.parameters:
            prepared_args['generator'] = generator
        free_vars = sample_fn(**prepared_args) 
        gold_outputs = self.fn(*free_vars)
        self.sampled[set_name] = (free_vars, gold_outputs)
        return free_vars, gold_outputs


koza_1 = Benchmark("koza_1", lambda x: x*x*x*x + x*x*x + x*x + x,
                   get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

koza_2 = Benchmark("koza_2", lambda x: x*x*x*x*x - 2.0*x*x*x + x,
                    get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})


koza_3 = Benchmark("koza_3", lambda x: x*x*x*x*x*x - 2.0*x*x*x*x + x*x,
                    get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

nguyen_1 = Benchmark("nguyen_1", lambda x: x*x*x + x*x + x,
                    get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})


nguyen_2 = Benchmark("nguyen_2", lambda x: x*x*x*x + x*x*x + x*x + x,
                    get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

nguyen_3 = Benchmark("nguyen_3", lambda x: x*x*x*x*x + x*x*x*x + x*x + x,
                    get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

nguyen_4 = Benchmark("nguyen_4", lambda x: x*x*x*x*x*x + x*x*x*x*x + x*x*x*x + x*x*x + x*x + x,
                    get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

nguyen_5 = Benchmark("nguyen_5", lambda x: torch.sin(x*x) * torch.cos(x) - 1.0,
                    get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

nguyen_6 = Benchmark("nguyen_6", lambda x: torch.sin(x) + torch.sin(x + x*x),
                    get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

nguyen_7 = Benchmark("nguyen_7", lambda x: torch.log(x + 1.0) + torch.log(x*x + 1.0),
                    get_rand_points, {"num_samples": 20, "ranges": [(0.0, 2.0)]})

nguyen_8 = Benchmark("nguyen_8", lambda x: torch.sqrt(x),
                    get_rand_points, {"num_samples": 20, "ranges": [(0.0, 4.0)]})

nguyen_9 = Benchmark("nguyen_9", lambda x, y: torch.sin(x) + torch.sin(y * y),
                    get_rand_points, {"num_samples": 100, "ranges": [(0.0, 1.0), (0.0, 1.0)]})

nguyen_10 = Benchmark("nguyen_10", lambda x, y: 2.0 * torch.sin(x) + torch.cos(y),
                    get_rand_points, {"num_samples": 100, "ranges": [(0.0, 1.0), (0.0, 1.0)]})

pagie_1 = Benchmark("pagie_1", lambda x, y: 1.0 / (1.0 + 1.0 / (x * x * x * x)) + 1.0 / (1.0 + 1.0 / (y * y * y * y)),
                    get_interval_grid, {"steps": 0.4, "ranges": [[-5.0, 5.0], [-5.0, 5.0]]})

pagie_2 = Benchmark("pagie_2", lambda x, y, z: 1.0 / (1.0 + 1.0 / ( x * x * x * x)) + 1.0 / (1.0 + 1.0 / (y * y * y * y)) + 1.0 / (1.0 + 1.0 / (z * z * z * z)),
                    get_interval_grid, {"steps": 0.4, "ranges": [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]})

korns_sampling = get_rand_points, {"num_samples": 10000, "ranges": [[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]]}

korns_1 = Benchmark("korns_1", lambda *xs: 1.57 + (24.3 * xs[3]), *korns_sampling)
korns_2 = Benchmark("korns_2", lambda *xs: 0.23 + (14.2 * ((xs[3] + xs[1]) / (3.0 * xs[4]))), *korns_sampling)
korns_3 = Benchmark("korns_3", lambda *xs: -5.41 + (4.9 * (((xs[3] - xs[0]) + (xs[1]/xs[4])) / (3 * xs[4]))), *korns_sampling)
korns_4 = Benchmark("korns_4", lambda *xs: -2.3 + (0.13 * torch.sin(xs[2])), *korns_sampling)
korns_5 = Benchmark("korns_5", lambda *xs: 3.0 + (2.13 * torch.log(xs[4])), *korns_sampling)
korns_6 = Benchmark("korns_6", lambda *xs: 1.3 + (0.13 * torch.sqrt(xs[0])), *korns_sampling)
korns_7 = Benchmark("korns_7", lambda *xs: 213.80940889 - (213.80940889 * torch.exp(-0.54723748542 * xs[0])), *korns_sampling)
korns_8 = Benchmark("korns_8", lambda *xs: 6.87 + (11.0 * torch.sqrt(7.23 * xs[0] * xs[3] * xs[4])), *korns_sampling)
korns_9 = Benchmark("korns_9", lambda *xs: torch.sqrt(xs[0]) / torch.log(xs[1]) * torch.exp(xs[2]) / (xs[3] * xs[3]), *korns_sampling)
korns_10 = Benchmark("korns_10", lambda *xs: 0.81 + (24.3 * (((2.0 * xs[1]) + (3.0 * (xs[2] * xs[2]))) / ((4.0 * (xs[3]*xs[3]*xs[3])) + (5.0 * (xs[4]*xs[4]*xs[4]*xs[4]))))), *korns_sampling)
korns_11 = Benchmark("korns_11", lambda *xs: 6.87 + (11.0 * torch.cos(7.23 * xs[0]*xs[0]*xs[0])), *korns_sampling)
korns_12 = Benchmark("korns_12", lambda *xs: 2.0 - (2.1 * (torch.cos(9.8 * xs[0]) * torch.sin(1.3 * xs[4]))), *korns_sampling)
korns_13 = Benchmark("korns_13", lambda *xs: 32.0 - (3.0 * ((torch.tan(xs[0]) / torch.tan(xs[1])) * (torch.tan(xs[2])/torch.tan(xs[3])))), *korns_sampling)
korns_14 = Benchmark("korns_14", lambda *xs: 22.0 - (4.2 * ((torch.cos(xs[0]) - torch.tan(xs[1]))*(torch.tanh(xs[2])/torch.sin(xs[3])))), *korns_sampling)
korns_15 = Benchmark("korns_15", lambda *xs: 12.0 - (6.0 * ((torch.tan(xs[0])/torch.exp(xs[1])) * (torch.log(xs[2]) - torch.tan(xs[3])))), *korns_sampling) 

keijzer_1 = Benchmark("keijzer_1", lambda x: 0.3 * x * torch.sin(2.0 * torch.pi * x),
                        get_interval_grid, {"steps": 0.1, "ranges": [[-1.0, 1.0]]},
                        get_interval_grid, {"steps": 0.001, "ranges": [[-1.0, 1.0]]})
    
keijzer_2 = Benchmark("keijzer_2", lambda x: 0.3 * x * torch.sin(2.0 * torch.pi * x),
                        get_interval_grid, {"steps": 0.1, "ranges": [[-2.0, 2.0]]},
                        get_interval_grid, {"steps": 0.001, "ranges": [[-2.0, 2.0]]})

keijzer_3 = Benchmark("keijzer_3", lambda x: 0.3 * x * torch.sin(2.0 * torch.pi * x),
                        get_interval_grid, {"steps": 0.1, "ranges": [[-3.0, 3.0]]},
                        get_interval_grid, {"steps": 0.001, "ranges": [[-3.0, 3.0]]})

keijzer_4 = Benchmark("keijzer_4", lambda x: x * torch.exp(-x) * torch.cos(x) * torch.sin(x) * (torch.sin(x) * torch.sin(x) * torch.cos(x) - 1),
                        get_interval_grid, {"steps": 0.05, "ranges": [[0.0, 10.0]]},
                        get_interval_grid, {"steps": 0.05, "ranges": [[0.05, 10.05]]})

keijzer_5 = Benchmark("keijzer_5", lambda x, y, z: (30.0 * x * z) / ((x - 10.0) * y * y),
                        get_rand_points, {"num_samples": 1000, "ranges": [[-1.0, 1.0],[1.0,2.0],[-1.0,1.0]]},
                        get_rand_points, {"num_samples": 10000, "ranges": [[-1.0, 1.0],[1.0,2.0],[-1.0,1.0]]})

keijzer_6 = Benchmark("keijzer_6", lambda *xs: torch.stack([torch.sum(1.0 / torch.arange(1, torch.floor(x) + 1)) for x in xs]),
                        get_interval_grid, {"steps": 1.0, "ranges": [[1.0, 50.0]]},
                        get_interval_grid, {"steps": 1.0, "ranges": [[1.0, 120.0]]})

keijzer_7 = Benchmark("keijzer_7", lambda x: torch.log(x),
                        get_interval_grid, {"steps": 1.0, "ranges": [[1.0, 100.0]]},
                        get_interval_grid, {"steps": 0.1, "ranges": [[1.0, 100.0]]})

keijzer_8 = Benchmark("keijzer_8", lambda x: torch.sqrt(x),
                        get_interval_grid, {"steps": 1.0, "ranges": [[0.0, 100.0]]},
                        get_interval_grid, {"steps": 0.1, "ranges": [[0.0, 100.0]]})                      

keijzer_9 = Benchmark("keijzer_9", lambda x: torch.arcsinh(x),
                        get_interval_grid, {"steps": 1.0, "ranges": [[0.0, 100.0]]},
                        get_interval_grid, {"steps": 0.1, "ranges": [[0.0, 100.0]]})

keijzer_10 = Benchmark("keijzer_10", lambda x, y: torch.float_power(x, y),
                        get_rand_points, {"num_samples": 100, "ranges": [[0.0, 1.0], [0.0, 1.0]]},
                        get_interval_grid, {"steps": 0.01, "ranges": [[0.0, 1.0], [0.0, 1.0]]})
    

keijzer_11 = Benchmark("keijzer_11", lambda x, y: x * y + torch.sin((x - 1.0) * (y - 1.0)),
                        get_rand_points, {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
                        get_interval_grid, {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]})
                       
keijzer_12 = Benchmark("keijzer_12", lambda x, y: x*x*x*x - x*x*x + y*y/2.0 - y,
                        get_rand_points, {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
                        get_interval_grid, {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]})

keijzer_13 = Benchmark("keijzer_13", lambda x, y: 6.0 * torch.sin(x) * torch.cos(y),
                        get_rand_points, {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
                        get_interval_grid, {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]})

keijzer_14 = Benchmark("keijzer_14", lambda x, y: 8.0 / (2.0 + x*x + y*y),
                        get_rand_points, {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
                        get_interval_grid, {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]})

keijzer_15 = Benchmark("keijzer_15", lambda x, y: x*x*x / 5.0 + y*y*y/2.0 - y - x,
                        get_rand_points, {"num_samples": 20, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]},
                        get_interval_grid, {"steps": 0.01, "ranges": [[-3.0, 3.0], [-3.0, 3.0]]})

vladislavleva_1 = Benchmark("vladislavleva_1", lambda x, y: torch.exp(-(x-1)*(x-1)) / (1.2 + (y - 2.5)*(y-2.5)),
                        get_rand_points, {"num_samples": 100, "ranges": [[0.3, 4.0], [0.3, 4.0]]},
                        get_interval_grid, {"steps": 0.1, "ranges": [[-0.2, 4.2], [-0.2, 4.2]]})

vladislavleva_2 = Benchmark("vladislavleva_2", lambda x: torch.exp(-x) * x*x*x * torch.cos(x) * torch.sin(x) * (torch.cos(x) * torch.sin(x) * torch.sin(x) - 1),
                        get_interval_grid, {"steps": 0.1, "ranges": [[0.05, 10]]},
                        get_interval_grid, {"steps": 0.05, "ranges": [[-0.5, 10.5]]})

vladislavleva_3 = Benchmark("vladislavleva_3", lambda x, y: torch.exp(-x)*x*x*x*torch.cos(x)*torch.sin(x)*(torch.cos(x)*torch.sin(x)*torch.sin(x) - 1) * (y - 5),
                        get_interval_grid, {"steps": [0.1, 2.0], "ranges": [[0.05, 10], [0.05, 10.05]]},
                        get_interval_grid, {"steps": [0.05, 0.5], "ranges": [[-0.5, 10.5], [-0.5, 10.5]]})

vladislavleva_4 = Benchmark("vladislavleva_4", lambda *xs: 10.0 / (5.0 + torch.sum((xs - 3.0) ** 2, axis=0)),
                        get_rand_points, {"num_samples": 1024, "ranges": [[0.05, 6.05]] * 5},
                        get_rand_points, {"num_samples": 5000, "ranges": [[-0.25, 6.35]] * 5})


vladislavleva_5 = Benchmark("vladislavleva_5", lambda x, y, z: (30.0 * (x - 1.0) * (z - 1.0)) / (y * y * (x - 10.0)),
                        get_rand_points, {"num_samples": 300, "ranges": [[0.05, 2.0], [1.0, 2.0], [0.05, 2.0]]},
                        get_interval_grid, {"steps": [0.15, 0.15, 0.1], "ranges": [[-0.05, 2.1], [0.95, 2.05], [-0.05, 2.1]]})

vladislavleva_6 = Benchmark("vladislavleva_6", lambda x, y: 6.0 * torch.sin(x) * torch.cos(y),
                        get_rand_points, {"num_samples": 30, "ranges": [[0.1, 5.9], [0.1, 5.9]]},
                        get_interval_grid, {"steps": [0.02, 0.02], "ranges": [[-0.05, 6.05], [-0.05, 6.05]]})


vladislavleva_7 = Benchmark("vladislavleva_7", lambda x, y: (x - 3.0) * (y - 3.0) + 2 * torch.sin((x - 4.0) * (y - 4.0)),
                        get_rand_points, {"num_samples": 300, "ranges": [[0.05, 6.05], [0.05, 6.05]]},
                        get_rand_points, {"num_samples": 1000, "ranges": [[-0.25, 6.35], [-0.25, 6.35]]})


vladislavleva_8 = Benchmark("vladislavleva_8", lambda x, y: ((x - 3.0) * (x - 3.0) * (x - 3.0) * (x - 3.0) + (y - 3.0) * (y - 3.0) * (y - 3.0) - (y - 3.0)) / ((y - 2.0) * (y - 2.0) * (y - 2.0) * (y - 2.0) + 10.0),
                        get_rand_points, {"num_samples": 50, "ranges": [[0.05, 6.05], [0.05, 6.05]]},
                        get_interval_grid, {"steps": [0.2, 0.2], "ranges": [[-0.25, 6.35], [-0.25, 6.35]]})


all_benchmarks = [
    koza_1, koza_2, koza_3,
    nguyen_1, nguyen_2, nguyen_3, nguyen_4, nguyen_5, nguyen_6, nguyen_7, nguyen_8, nguyen_9, nguyen_10,
    pagie_1, pagie_2,
    korns_1, korns_2, korns_3, korns_4, korns_5, korns_6, korns_7, korns_8, korns_9, korns_10,
    korns_11, korns_12, korns_13, korns_14, korns_15,
    keijzer_1, keijzer_2, keijzer_3,
    keijzer_4, keijzer_5, keijzer_6,
    keijzer_7, keijzer_8, keijzer_9,
    keijzer_10, keijzer_11, keijzer_12,
    keijzer_13, keijzer_14, keijzer_15,
    vladislavleva_1, vladislavleva_2, vladislavleva_3,
    vladislavleva_4, vladislavleva_5, vladislavleva_6,
    vladislavleva_7, vladislavleva_8  ]

# NOTE: it makes sense to optimize one leaf at a time to avoid introducing too many weights to optimize 
# example: when optimized (add (mul x x) x) - 3 places gives suboptimal results - local minima hit in 10 tries. 
# with 2 variables - 1 and 2 places - optimization gives perfect result.
# TODO: optimizator config and metrics - counts of evals! lr ??? 
# Think: what places should be extended with weights. 

if __name__ == "__main__":
    term_cache = {}

    device = "cpu"

    names_to_ids_cache = {}
    ids_to_names_cache = {}
    alloc_id = partial(dict_alloc_id, names_to_ids_cache=names_to_ids_cache,
                       ids_to_names_cache=ids_to_names_cache)

    def f1(x:torch.Tensor) -> torch.Tensor:
        # return 3.13 * x + 1.42 #
        return x * x + 3.131 * x + 1.42
    
    const_ranges = torch.tensor([0, 10], dtype=torch.float32, device=device)
    grid = get_interval_grid(1, const_ranges[torch.newaxis, :], rand_deltas=False)
    gold_outputs = f1(grid[:, 0])

    c_term = cache_term(term_cache, "c")
    term, _ = parse_term(term_cache, alloc_id, "(add (mul x x) x)")

    max_num_syntaxes = 10000
    semantics = torch.zeros((max_num_syntaxes, gold_outputs.shape[0]), dtype=torch.float32, device=device)
    semantics[0] = 1 # 0 is constant 1
    semantics[1] = grid[:, 0]
    x_term = cache_term(term_cache, "x")
    leaf_semantic_ids = {c_term: 0, x_term: 1}  # x is at index 1 in semantics
    branch_semantic_ids = {}
    
    lbfgs_optimize(term, gold_outputs, semantics, leaf_semantic_ids,
                   branch_semantic_ids, const_ranges)
    
    pass


# IDEAS: 
# 1. syntactic simplification 
# 2. FFT for learning rate?? 
# 3. Machcine learnring for patterns 
# 4. Spearman correlation coefficient for monotonicity  

# about monotonicity:
#   compositions o monotonic functions is monotonic 
#   we can establish monotonicity with Spearman correlation and propagate this property through compositions
#   if we have monotonicity - we can use it to reduce search space

# TODO: need to add constraints in term generation - properties like idempotence could be used to prohibit some terms or simplify terms
# think about commutativity - how to establish for black box and its usefullness (cutting search space)
# commutativity w.r.t 2 positions in the tree - most are not commutative, should we heavily test?
