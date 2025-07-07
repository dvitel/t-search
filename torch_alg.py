
from functools import partial
from typing import Callable, Optional
import torch

from term import Term, build_term, evaluate, get_leaves, parse_term

alg_ops_torch = {
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

def get_full_grid(grid_values: list[torch.Tensor]):
    ''' grid_values - per each dimension/variable, specifies allowed values for grid '''
    assert len(grid_values) > 0, "Grid values should not be empty"
    meshes = torch.meshgrid(*[v for v in grid_values], indexing='ij')
    grid_nd = torch.stack(meshes, dim=-1)
    grid = grid_nd.reshape(-1, grid_nd.shape[-1])
    return grid 

# t1 = get_full_grid([torch.tensor([1,2]), torch.tensor([1,2]), torch.tensor([4,5])])
# pass 

def get_rand_grid_point(grid_values: list[torch.Tensor]):
    assert len(grid_values) > 0, "Grid values should not be empty"
    values = [v[torch.randint(0, len(v), (1,))] for v in grid_values]
    stacked = torch.cat(values, dim=0)
    return stacked

# t2 = get_rand_grid_point([torch.tensor([1,2]), torch.tensor([1,2]), torch.tensor([4,5])])
# pass

def get_rand_points(num_samples: int, ranges: torch.Tensor):
    ''' ranges: tensor 1d - free var, 2d - [min, max] '''
    mins = ranges[:, 0]
    maxs = ranges[:, 1]
    dist = maxs - mins
    values = mins[:, torch.newaxis] + dist[:, torch.newaxis] * torch.rand(len(ranges), num_samples, device=ranges.device)
    return values

# t3 = get_rand_points(10, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_rand_full_grid(num_samples, ranges: torch.Tensor):
    ''' ranges: tensor 1d - free var, 2d - [min, max] '''
    grid_values = get_rand_points(num_samples, ranges)
    grid = get_full_grid(grid_values)
    return grid

# t4 = get_rand_full_grid(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_interval_points(num_samples_per_dim: torch.Tensor | int, ranges: torch.Tensor, deltas: Optional[torch.Tensor] = None, rand_deltas = False):
    mins = ranges[:, 0]
    maxs = ranges[:, 1]
    steps = (maxs - mins) / num_samples_per_dim
    if rand_deltas:
        deltas = deltas or steps
        deltas *= torch.rand(ranges.shape[0], device=ranges.device)
    if deltas is not None:
        mins += deltas
    one_range = torch.arange(0, num_samples_per_dim, device=ranges.device)
    values = mins[:, torch.newaxis] + steps[:, torch.newaxis] * one_range
    return values

# t5 = get_interval_points(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_interval_grid(num_samples_per_dim: torch.Tensor | int, ranges: torch.Tensor, deltas: Optional[torch.Tensor] = None, rand_deltas = False):
    grid_values = get_interval_points(num_samples_per_dim, ranges, deltas, rand_deltas)
    grid = get_full_grid(grid_values)
    return grid

# t6 = get_interval_grid(4, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_rand_interval_points(num_samples: int, ranges: torch.Tensor, num_samples_per_dim: torch.Tensor | int = 0, deltas: Optional[torch.Tensor] = None, rand_deltas = True):
    num_samples_per_dim = num_samples if num_samples_per_dim == 0 else num_samples_per_dim
    grid_values = get_interval_points(num_samples_per_dim, ranges, deltas, rand_deltas)
    points = [get_rand_grid_point(grid_values) for _ in range(num_samples)]
    points = torch.stack(points, dim=0)
    return points

# t7 = get_rand_interval_points(10, torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
# pass

# https://en.wikipedia.org/wiki/Chebyshev_nodes
def get_chebyshev_points(num_samples, ranges: torch.Tensor, rand_deltas = False):
    assert num_samples > 0, "Number of samples should be greater than 1"
    mins = ranges[:, 0]
    maxs = ranges[:, 1]
    dist = maxs - mins
    indexes = torch.arange(1, num_samples + 1, dtype=float, device=ranges.device) #torch.tile(torch.arange(0, num_samples), (ranges.shape[0], 1))
    if rand_deltas:
        indexes = torch.rand(ranges.shape[0], device=ranges.device)[:, torch.newaxis] + indexes
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
        outputs = evaluate(term, alg_ops_torch, last_bindings, last)
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

# NOTE: it makes sense to optimize one leaf at a time to avoid introducing too many weights to optimize 
# example: when optimized (add (mul x x) x) - 3 places gives suboptimal results - local minima hit in 10 tries. 
# with 2 variables - 1 and 2 places - optimization gives perfect result.
# TODO: optimizator config and metrics - counts of evals! lr ??? 
# Think: what places should be extended with weights. 

if __name__ == "__main__":
    term_cache = {}

    device = "cpu"

    def f1(x:torch.Tensor) -> torch.Tensor:
        # return 3.13 * x + 1.42 #
        return x * x + 3.131 * x + 1.42
    
    const_ranges = torch.tensor([0, 10], dtype=torch.float32, device=device)
    grid = get_interval_grid(10, const_ranges[torch.newaxis, :], rand_deltas=False)
    gold_outputs = f1(grid[:, 0])

    c_term = build_term(term_cache, "c")
    term, _ = parse_term(term_cache, "(add (mul x x) x)")

    max_num_syntaxes = 10000
    semantics = torch.zeros((max_num_syntaxes, gold_outputs.shape[0]), dtype=torch.float32, device=device)
    semantics[0] = 1 # 0 is constant 1
    semantics[1] = grid[:, 0]
    x_term = build_term(term_cache, "x")
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
