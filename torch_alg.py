
from dataclasses import dataclass, field
from functools import partial
import inspect
import math
from typing import Any, Callable, Literal, Optional
import numpy as np
import torch

from term import Builders, Term, TermPos, Value, Variable, collect_terms, evaluate, parse_term, replace_fn, replace_pos

alg_ops = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    # "pow": lambda a, b: a ** b,
    # "neg": lambda a: -a,
    # "inv": lambda a: 1 / a,
    # "exp": lambda a: torch.exp(a),
    # "log": lambda a: torch.log(a),
    # "sin": lambda a: torch.sin(a),
    # "cos": lambda a: torch.cos(a),
}

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
    values = [v[torch.randint(0, len(v), (1,), device = v.device, generator=generator)] for v in grid_values]
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
        if deltas is None:
            deltas = steps.clone()
        deltas *= torch.rand(ranges.shape[0], device=ranges.device, generator = generator)
    if deltas is None:
        deltas = torch.zeros_like(steps)
    values = [torch.arange(r[0] + d, r[1], s, device=r.device, dtype=r.dtype) for r, s, d in zip(ranges, steps, deltas)]
    return values

# t5 = get_interval_points(0.5, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_interval_grid(steps: torch.Tensor | float, ranges: torch.Tensor, 
                      deltas: Optional[torch.Tensor] = None, rand_deltas = False,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    grid_values = get_interval_points(steps, ranges, deltas, rand_deltas, generator)
    grid = get_full_grid(grid_values)
    return grid

# t6 = get_interval_grid(0.5, torch.tensor([[1, 2], [3, 4], [5, 6]]))
# pass

def get_rand_interval_points(num_samples: int, ranges: torch.Tensor, 
                             steps: Optional[torch.Tensor | float] = None, 
                             deltas: Optional[torch.Tensor] = None, rand_deltas = True,
                             generator: torch.Generator | None = None) -> list[torch.Tensor]:
    if steps is None:
        steps = (ranges[:, 1] - ranges[:, 0]) / (num_samples + 1)
    grid_values = get_interval_points(steps, ranges, deltas, rand_deltas, generator = generator)
    points = [get_rand_grid_point(grid_values, generator = generator) for _ in range(num_samples)]
    # points = torch.stack(points, dim=0)
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

def mse_loss_builder(target):
    return lambda output: torch.mean((output - target) ** 2, dim=-1)

def nmse_loss_builder(target):
    ''' we follow R^2 normalization: NMSE = 1 - R^2 '''
    # norm = torch.mean(target ** 2, dim=-1) # TODO: could be different norms: std dev 
    norm = torch.var(target, dim=-1, unbiased=False)
    # mse = torch.mean((output - target) ** 2, dim=-1)
    def loss_fn(output):
        mse = torch.mean((output - target) ** 2, dim=-1)
        nmse = mse / norm
        return nmse
    return loss_fn

# def mse_loss_nan_v(predictions, target, *, nan_error = torch.inf):
#     loss = torch.mean((predictions - target) ** 2, dim=-1)
#     loss = torch.where(torch.isnan(loss), torch.tensor(nan_error, device=loss.device, dtype=loss.dtype), loss)
#     return loss     

# def mse_loss_nan_vf(predictions, target, *, 
#                     nan_value_fn = lambda m,t: torch.tensor(torch.inf, 
#                                                     device = t.device, dtype=t.dtype), 
#                     nan_frac = 0.5):
#     nan_frac_count = math.floor(target.shape[0] * nan_frac)
#     nan_mask = torch.isnan(predictions)
#     err_rows: torch.Tensor = nan_mask.sum(dim=-1) > nan_frac_count
#     bad_positions = nan_mask & err_rows.unsqueeze(-1)
#     fixed_predictions = torch.where(bad_positions, 
#                                     nan_value_fn(bad_positions, target),
#                                     predictions)
#     err_rows.logical_not_()
#     fixed_positions = nan_mask & err_rows.unsqueeze(-1)
#     fully_fixed_predictions = torch.where(fixed_positions, target, fixed_predictions)
#     loss = torch.mean((fully_fixed_predictions - target) ** 2, dim=-1)
#     del fully_fixed_predictions, fixed_predictions, fixed_positions, bad_positions, err_rows, nan_mask
#     return loss         

def l1_loss_builder(target):
    return lambda outputs: torch.mean(torch.abs(outputs - target), dim=-1)  

# @dataclass(frozen=False, eq=False, unsafe_hash=False, repr=False)
# @dataclass(frozen=True)
# class OptimPoint(Term): # guarantee ref equality instead of Value eq
#     occur: int

@dataclass(frozen=True)
class OptimPoint(Term):
    point_id: int # optim point in root term 

@dataclass 
class OptimState:
    optim_term: Term
    optim_points: list[OptimPoint] # starts of optim paths
    binding: dict[Term, torch.Tensor] # collected path bindings
    best_binding: dict[Term, torch.Tensor] | None = None # intermediate bindings of the optimization
    best_loss: torch.Tensor | None = None
    best_term: Term | None = None
    max_tries: int = 1

    def dec(self):
        self.max_tries -= 1
        if self.max_tries <= 0:
            for v in self.binding.values():
                del v
            self.binding.clear()
            # self.optim_points.clear()
    
class LRAdjust(Exception):
    pass

optim_id = -1 # for debugging
def optimize(optim_state: OptimState, loss_fn: Callable, given_ops: dict[str, Callable], 
                get_binding: Callable, *, eval_fn = evaluate,
                num_best: int = 1, lr: float = 1.0, max_evals: int = 10, 
                collect_inner_binding: bool = False, loss_threshold: float = 0.1,
                ):
    global optim_id
    optim_id += 1

    num_evals = 0
    num_root_evals = 0

    print(f">>> [{optim_id}] {optim_state.optim_term}")
    
    # print(f"--- {term}")
        
    # cur_lr = lr 
    # cur_best_lr = lr

    # for c, cv in zip(optim_state.optim_points, const_vectors):
    #     c.requires_grad = False 
    #     c.copy_(cv) # copy new value to optim point
    #     c.requires_grad = True

    params = []
    for optim_point in optim_state.optim_points:
        point_binding = optim_state.binding[optim_point]
        point_binding.requires_grad = True
        params.append(point_binding)

    # print(f"\t === {optim_state.max_tries} {cur_lr}")

    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_evals,
                                    max_eval=max_evals,
                                    # max_eval = 1.5 * num_steps,
                                    tolerance_change=1e-6, # TODO - should be parameters???
                                    tolerance_grad=1e-3,
                                    # history_size=100,
                                    line_search_fn='strong_wolfe'
                                    )

    best_loss = None 

    iter_loss = []
    iter_binding = {}

    if optim_state.best_loss is not None:
        iter_loss.append(optim_state.best_loss)    

    if optim_state.best_binding is not None:
        for k, v in optim_state.best_binding.items():
            iter_binding[k] = [v]

    def closure_builder(optimizer: torch.optim.Optimizer):
        nonlocal num_root_evals, best_loss, max_evals

        # cur_lr = optimizer.param_groups[0]['lr']
        # print(f"LR: {cur_lr}")
        if num_root_evals >= max_evals:
            raise LRAdjust(None)
        num_root_evals += 1
        optimizer.zero_grad()

        def _redirected_get_binding(root: Term, term: Term):
            if isinstance(term, OptimPoint):
                return optim_state.binding[term]
            return get_binding(root, term)   

        def _set_binding(root: Term, term: Term, output: torch.Tensor):
            nonlocal num_evals
            num_evals += 1
            if collect_inner_binding and (root != term):
                if term in optim_state.binding:
                    del optim_state.binding[term]
                optim_state.binding[term] = output
            return         
                
        outputs: torch.Tensor = eval_fn(optim_state.optim_term, given_ops, _redirected_get_binding, _set_binding)
        # assert outputs is not None, "Term evaluation should be full. Term is evaluated partially"
        loss: torch.Tensor = loss_fn(outputs) 
        finite_loss_mask = torch.isfinite(loss)
        if not torch.any(finite_loss_mask):
            raise LRAdjust(None)
        
        finite_loss_ids, = torch.where(finite_loss_mask)

        finite_loss = loss[finite_loss_ids]

        # if best_loss.numel() == 1: # pick best loss 
        #     # finit_loss_ids = finite_ids[finit_loss_id_ids]
        #     new_min_loss_id_id = torch.argmin(finite_loss)
        #     new_min_loss_id = finite_loss_ids[new_min_loss_id_id]
        #     new_min_loss = finite_loss[new_min_loss_id_id]
        #     if new_min_loss < best_loss:
        #         best_loss.copy_(new_min_loss)
        #         for k, v in binding.items():
        #             if k in best_binding:
        #                 del best_binding[k]
        #                 best_binding[k].copy_(v[new_min_loss_id])
        #             else:
        #                 best_binding[k] = v[new_min_loss_id].detach().clone()
        #             pass 
        # else:
        #     new_min_loss = None
        #     # stacked_loss = torch.concat([finite_loss.detach().clone(), best_loss], dim=0)
        #     stacked_loss = torch.concat([finite_loss, best_loss], dim=0)
        #     sort_ids = torch.argsort(stacked_loss)[:best_loss.shape[0]]
        #     best_loss.copy_(stacked_loss[sort_ids])
        #     del stacked_loss
        #     new_mask = sort_ids < finite_loss.shape[0]
        #     new_ids, = torch.where(new_mask)
        #     if len(new_ids) > 0:
        #         new_sort_ids = sort_ids[new_ids]
        #         for k, v in binding.items():
        #             if k in best_binding:
        #                 best_binding[k][new_ids] = v[new_sort_ids]
        #             else:
        #                 best_binding[k] = v
        #         for cur_b, last_b in zip(optim_state.best_binding, optim_state.optim_points):
        #             cur_b[new_ids] = last_b[new_sort_ids]
        
        min_loss = finite_loss.min()

        print(f"\tLoss {min_loss.item()}, evals {num_root_evals}")

        if min_loss < loss_threshold:
            iter_loss.append(loss.detach().clone())
            for k,v in optim_state.binding.items():
                iter_binding.setdefault(k, []).append(v.detach().clone())        
        
        # TODO: experiment more with early exit
        # if best_loss is not None:
        #     # if torch.allclose(new_min_loss, last_min_loss, rtol=rtol, atol=atol):
        #     #     raise LRAdjust(None)
        #     # elif new_min_loss > last_min_loss:
        #     #     # optimizer.param_groups[0]['lr'] *= 0.5
        #     #     pass
        #     # if min_loss >= best_loss:
        #     #     raise LRAdjust(None)
        #     pass

        best_loss = min_loss

        total_loss = finite_loss.mean()
        total_loss.backward()

        return total_loss

    closure = partial(closure_builder, optimizer)
        
    try:
        first_loss = optimizer.step(closure)
    except ZeroDivisionError as e:
        # print(f"LBFGS optimization failed with ZeroDivisionError")
        pass # just use last loss
    except LRAdjust as e:
        pass
        # if e.args[0] is None:
        #     break 
        # cur_lr *= e.args[0]
        # lr_try -= 1
        # continue

    # NOTE: optimizer actually returns first loss

    # assert torch.allclose(last_loss, final_loss)

    if len(iter_loss) > 0:

        all_iter_loss = torch.concat(iter_loss)
        all_iter_loss.nan_to_num_(torch.inf)
        if num_best == 1:
            best_ids = torch.argmin(all_iter_loss).unsqueeze(0)
        else:
            best_ids = torch.argsort(all_iter_loss)[:num_best]
        best_loss = all_iter_loss[best_ids]
        best_id_ids, = torch.where(best_loss < loss_threshold)
        best_ids = best_ids[best_id_ids]
        del all_iter_loss
        for il in iter_loss:
            del il

        best_binding = {}
        for k, v in iter_binding.items():
            v_tensor = torch.concat(v, dim=0)
            best_binding[k] = v_tensor[best_ids]
            del v_tensor
            for vi in v:
                del vi

        optim_state.best_loss = best_loss[best_id_ids]
        optim_state.best_binding = best_binding

    optim_state.dec()

    return num_evals, num_root_evals

def optimize_consts(term: Term, term_loss: torch.Tensor,
    loss_fn: Callable, builders: Builders,
    given_ops: dict[str, Callable], get_binding: Callable, start_range: torch.Tensor,
    *,
    eval_fn = evaluate, 
    num_vals = 10, max_tries = 1, max_evals = 20, num_best: int = 1,
    lr = 0.1,
    loss_threshold: float = 0.1,
    torch_gen: torch.Generator | None = None,
    term_values_cache: dict[Term, list[Value]],
    optim_term_cache: dict[Term, Term | None],
    optim_state_cache: dict[Term, OptimState],) -> Optional[tuple[OptimState, int, int]]:
    ''' Searches for the term const values that would bring it closer to the target outputs.
        Restarts will reinitialize the constants.
    '''
    
    if term not in optim_term_cache: # need to build optim term with optim points

        optim_points: list[OptimPoint] = []    
        binding = {}    
        values = []

        def const_to_optim_point(term, *_):
            if isinstance(term, Value):
                point_id = len(optim_points)
                point = OptimPoint(point_id)
                optim_points.append(point)
                value = torch.zeros((num_vals, 1 if len(term.value.shape) == 0 else term.value.shape[0]), dtype=term.value.dtype, device=term.value.device)
                binding[point] = value
                values.append(term)
                return point

        optim_term = replace_fn(term, const_to_optim_point, builders)

        if len(optim_points) == 0:
            optim_term = None
        optim_term_cache[term] = optim_term
        if optim_term is None:
            return None         
        term_values_cache[term] = values
        if optim_term not in optim_state_cache:
            optim_state = OptimState(optim_term, optim_points, binding, max_tries=max_tries)
            optim_state_cache[optim_term] = optim_state
        else:
            optim_state = optim_state_cache[optim_term]
            # if term_loss < optim_state.loss:
            #     optim_state.loss.copy_(term_loss)
            #     optim_state.binding = binding
            #     optim_state.final_term = term
    else:         
        optim_term = optim_term_cache[term]
        if optim_term is None:
            return None
        optim_state = optim_state_cache[optim_term]

    if optim_state.max_tries <= 0:
        return optim_state, 0, 0
    
    starts_to_attempt = []

    rand_points_to_attempt = num_vals
    if (optim_state.best_loss is None) or (term_loss < torch.min(optim_state.best_loss)): # at first try we also optimize current values
        starts_to_attempt.append([v.value for v in term_values_cache[term]])
        rand_points_to_attempt -= 1

    if rand_points_to_attempt > 0: # we use grid sampling with rand shifts 
        should_del_ranges = False
        if len(start_range.shape) == 1: # 1d range
            should_del_ranges = True 
            start_range = torch.tile(start_range, (len(optim_state.optim_points), 1))
        steps = (start_range[:, 1] - start_range[:, 0]) / (rand_points_to_attempt + 1)
        rand_points = get_interval_grid(steps, start_range, rand_deltas=True, generator=torch_gen)
        if rand_points.shape[0] > rand_points_to_attempt:
            selected_ids = torch.randperm(rand_points.shape[0], device=rand_points.device, generator=torch_gen)[:rand_points_to_attempt]
            new_rand_points = rand_points[selected_ids, :]
            del rand_points
            rand_points = new_rand_points
        starts_to_attempt.extend([[v for v in p] for p in rand_points])
        if should_del_ranges:
            del start_range

    const_vectors = []
    for point in optim_state.optim_points:
        const_values = torch.tensor([[p[point.point_id]] for p in starts_to_attempt], device=term_loss.device, dtype=term_loss.dtype)
        const_vectors.append(const_values)

    for p, cv in zip(optim_state.optim_points, const_vectors):
        binding = optim_state.binding[p]
        binding.requires_grad = False
        binding.copy_(cv) # copy new value to optim point
        binding.requires_grad = True

    best_loss_before = optim_state.best_loss if optim_state.best_loss is not None else None
    
    num_evals, num_root_evals = \
        optimize(optim_state, loss_fn, given_ops, get_binding, 
                 eval_fn = eval_fn, loss_threshold = loss_threshold,
                 collect_inner_binding = False,
                 lr=lr, max_evals=max_evals, num_best = num_best)

    if optim_state.best_loss is not None and \
        (best_loss_before is None or optim_state.best_loss[0] < best_loss_before[0]):
                
        def bind_optim_points(term, occur, **_):
            if isinstance(term, OptimPoint):
                return Value(optim_state.best_binding[term][0])
            
        optim_state.best_term = replace_fn(optim_state.optim_term, bind_optim_points, builders)

        optim_term_cache[optim_state.best_term] = optim_state.optim_term
    
    return optim_state, num_evals, num_root_evals

def get_pos_optim_state(term: Term, positions: list[TermPos], *,
    optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term | None],
    optim_state_cache: dict[Term, OptimState], builders: Builders,
    num_vals: int = 10, output_size: int = 1, max_tries: int = 1,
    dtype = torch.float16, device = "cuda") -> Optional[OptimState]:

    key = (term, *((p.term, p.occur) for p in positions))

    if key not in optim_term_cache:
        
        if len(positions) == 1:
            value = torch.zeros((num_vals, output_size), dtype=dtype, device=device)
            optim_points = [OptimPoint(0)]
            binding = {optim_points[0]: value}
            # pos_to_point = {(pos.term, pos.occur): point.point_id}
            optim_term = replace_pos(positions[0], optim_points[0], builders)
        else:

            prersent_pos = set((p.term, p.occur) for p in positions)
            optim_points = []
            binding = {}
            def pos_to_optim_point(term, occur):
                if (term, occur) in prersent_pos:
                    value = torch.zeros((num_vals, output_size), dtype=dtype, device=device)
                    point_id = len(optim_points)
                    point = OptimPoint(point_id)
                    optim_points.append(point)
                    binding[point] = value
                    return point
                
            optim_term = replace_fn(positions, pos_to_optim_point, builders)

        if len(optim_points) == 0:
            optim_term = None        
        optim_term_cache[key] = optim_term
        if optim_term is None:
            return None
        if optim_term not in optim_state_cache:   
            optim_state = OptimState(optim_term, optim_points, binding, max_tries=max_tries)
            optim_state_cache[optim_term] = optim_state
        else:
            optim_state = optim_state_cache[optim_term]             
    else:
        optim_term = optim_term_cache[key]
        if optim_term is None:
            return None
        optim_state = optim_state_cache[optim_term]
    return optim_state

def optimize_positions(optim_state: OptimState,
    loss_fn: Callable,
    given_ops: dict[str, Callable], get_binding: Callable, start_range: torch.Tensor,
    eval_fn = evaluate,
    pos_outputs: list[tuple[torch.Tensor]] = [],
    num_vals = 10, max_evals = 20, num_best: int = 5, collect_inner_binding: bool = False,
    lr = 1.0, loss_threshold: float = 0.1,
    torch_gen: torch.Generator | None = None) -> tuple[int, int]:
    ''' Searches for the term const values that would bring it closer to the target outputs.
        Restarts will reinitialize the constants.
    '''
    
    starts_to_attempt = [pos_outputs]

    rand_points_to_attempt = num_vals - len(starts_to_attempt)
    if rand_points_to_attempt > 0: # we use grid sampling with rand shifts 
        pos_rand_attempt = []
        for _ in optim_state.optim_points:
            rand_points = get_rand_interval_points(rand_points_to_attempt, start_range.t(), rand_deltas=True, generator=torch_gen)
            pos_rand_attempt.append(rand_points)
        starts_to_attempt.extend(zip(*pos_rand_attempt))

    for op_id, op in enumerate(optim_state.optim_points):
        binding = optim_state.binding[op]
        binding.requires_grad = False
        for opt_id, start_to_attempt in enumerate(starts_to_attempt):
            # for att_id, att in enumerate(start_to_attempt):                
            binding[opt_id] = start_to_attempt[op_id]
        binding.requires_grad = True    

    optim_res = optimize(optim_state, loss_fn, given_ops, get_binding, 
                         eval_fn = eval_fn, loss_threshold = loss_threshold,
                         collect_inner_binding = collect_inner_binding,
                         lr=lr, max_evals=max_evals, num_best=num_best)

    return optim_res

class Benchmark: 

    def __init__(self, name: str | None, fn: Callable,
                 train_sampling: Callable = get_rand_points, 
                 train_args: dict[str, Any] = None,
                 test_sampling: Optional[Callable] = None, 
                 test_args: Optional[dict[str, Any]] = None):
        self.name = name
        if name is None:
            self.name = fn.__name__
        self.fn = fn
        self.train_sampling: Callable = train_sampling
        self.train_args: dict[str, Any] = {} if train_args is None else train_args
        self.test_sampling: Optional[Callable] = test_sampling
        self.test_args: Optional[dict[str, Any]] = test_args
        self.sampled = {}

    def with_train_sampling(self, train_sampling = None, **kwargs):
        return Benchmark(self.name, self.fn, (train_sampling if train_sampling is not None else self.train_sampling), kwargs)
    
    def with_test_sampling(self, test_sampling = None, **kwargs):
        return Benchmark(self.name, self.fn, self.train_sampling, self.train_args,
                         (test_sampling if test_sampling is not None else self.test_sampling), kwargs)

    def sample_set(self, set_name: Literal["train", "test"], 
                            device = "cpu", dtype = torch.float32,
                            generator: torch.Generator | None = None,
                            sorted = False) -> tuple[torch.Tensor, torch.Tensor]:
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
        if sorted:
            indices = lexsort(free_vars)
            new_free_vars = free_vars[:, indices]
            new_gold_outputs = gold_outputs[indices]
            del free_vars, gold_outputs
            free_vars = new_free_vars
            gold_outputs = new_gold_outputs
        self.sampled[set_name] = (free_vars, gold_outputs)
        return free_vars, gold_outputs

test_0 = Benchmark("test_0", lambda x: x + 74.3,
                   get_rand_points, {"num_samples": 20, "ranges": [(-1.0, 1.0)]})

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

    def f1(x:torch.Tensor) -> torch.Tensor:
        # return 3.13 * x + 1.42 #
        return x * x + 3.131 * x + 1.42
    
    const_ranges = torch.tensor([0, 10], dtype=torch.float32, device=device)
    grid = get_interval_grid(1, const_ranges[torch.newaxis, :], rand_deltas=False)
    gold_outputs = f1(grid[:, 0])

    term, _ = parse_term("(add (mul x x) x)")

    max_num_syntaxes = 10000
    semantics = torch.zeros((max_num_syntaxes, gold_outputs.shape[0]), dtype=torch.float32, device=device)
    semantics[0] = 1 # 0 is constant 1
    semantics[1] = grid[:, 0]
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
