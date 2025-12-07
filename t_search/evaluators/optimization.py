from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch

from t_search.datasets.sampling import get_interval_grid, get_rand_interval_points
from t_search.syntax import Term, TermPos, Value, evaluate
from t_search.syntax.generation import Builders
from t_search.syntax.replacement import replace_fn, replace_pos


@dataclass(frozen=True)
class OptimPoint(Term):
    point_id: int  # optim point in root term


@dataclass
class OptimState:
    optim_term: Term
    optim_points: list[OptimPoint]  # starts of optim paths
    binding: dict[Term, torch.Tensor]  # collected path bindings
    best_binding: dict[Term, torch.Tensor] | None = None  # intermediate bindings of the optimization
    best_loss: torch.Tensor | None = None
    best_term: Term | None = None
    is_optimized: bool = False
    loss_fn: Callable | None = None
    
    def get_binding(self, root: Term, term: Term):
        if isinstance(term, OptimPoint):
            return self.binding[term]
        # NOTE: we still allow evaluator _get_binding to speedup computations  

class LRAdjust(Exception):
    pass


optim_id = -1  # for debugging


def optimize(
    optim_state: OptimState,
    *,
    # num_best: int = 1,
    lr: float = 1.0,
    max_evals: int = 10,
    tolerance_change: float = 1e-6,
    tolerance_grad: float = 1e-3,
    # loss_threshold: float = 0.1,
) -> None:
    global optim_id

    if optim_state.is_optimized:
        return 0, 0
    
    assert optim_state.loss_fn is not None, "Optimization loss function is not set"
        
    optim_id += 1

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

    optimizer = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=max_evals,
        max_eval=max_evals,
        # max_eval = 1.5 * num_steps,
        tolerance_change=tolerance_change,
        tolerance_grad=tolerance_grad,
        # history_size=100,
        line_search_fn="strong_wolfe",
    )

    best_loss = None
    best_binding = None

    # iter_loss = []
    # iter_binding = {}

    if optim_state.best_loss is not None:
        # iter_loss.append(optim_state.best_loss)
        best_loss = optim_state.best_loss

    if optim_state.best_binding is not None:
        best_binding = dict(optim_state.best_binding)
        # for k, v in optim_state.best_binding.items():
        #     iter_binding[k] = [v]

    num_root_evals = 0

    def closure_builder(optimizer: torch.optim.Optimizer):
        nonlocal best_loss, max_evals, best_binding, num_root_evals

        # cur_lr = optimizer.param_groups[0]['lr']
        # print(f"LR: {cur_lr}")        
        if num_root_evals >= max_evals:
            raise LRAdjust(None)
        optimizer.zero_grad()


        loss: torch.Tensor = optim_state.loss_fn(optim_state.optim_term)
        num_root_evals += 1
        fixed_loss = loss.nan_to_num_(torch.inf)
        # finite_loss_mask = torch.isfinite(loss)
        if not torch.all(torch.isfinite(fixed_loss)):
            raise LRAdjust(None)

        # (finite_loss_ids,) = torch.where(finite_loss_mask)

        # finite_loss = loss[finite_loss_ids]

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

        loss_min_pos = fixed_loss.argmin()
        min_loss = fixed_loss[loss_min_pos]

        print(f"\tLoss {min_loss.item()}, evals {num_root_evals}")

        # if min_loss < loss_threshold:
        #     iter_loss.append(loss.detach().clone())
        #     for k, v in optim_state.binding.items():
        #         iter_binding.setdefault(k, []).append(v.detach().clone())

        if best_loss is None or min_loss < best_loss:
            best_loss = min_loss.detach().clone()
            for k, v in optim_state.binding.items():
                best_binding[k] = v[loss_min_pos].detach().clone()

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

        finite_loss = fixed_loss[torch.isfinite(fixed_loss)]
        total_loss = finite_loss.mean()
        total_loss.backward()

        return total_loss

    closure = partial(closure_builder, optimizer)

    try:
        first_loss = optimizer.step(closure)
    except ZeroDivisionError as e:
        # print(f"LBFGS optimization failed with ZeroDivisionError")
        pass  # just use last loss
    except LRAdjust as e:
        pass
        # if e.args[0] is None:
        #     break
        # cur_lr *= e.args[0]
        # lr_try -= 1
        # continue

    # NOTE: optimizer actually returns first loss

    # assert torch.allclose(last_loss, final_loss)

    if best_loss is not None:

        optim_state.best_loss = best_loss
        optim_state.best_binding = best_binding
    
    optim_state.is_optimized = True

def get_pos_optim_state(
    term: Term,
    positions: list[TermPos],
    *,
    optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term | None],
    optim_state_cache: dict[Term, OptimState],
    builders: Builders,
    num_vals: int = 10,
    output_size: int = 1,
    dtype=torch.float16,
    device="cuda",
) -> Optional[OptimState]:

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
            optim_state = OptimState(optim_term, optim_points, binding)
            optim_state_cache[optim_term] = optim_state
        else:
            optim_state = optim_state_cache[optim_term]
    else:
        optim_term = optim_term_cache[key]
        if optim_term is None:
            return None
        optim_state = optim_state_cache[optim_term]
    return optim_state


def optimize_positions(
    optim_state: OptimState,
    loss_fn: Callable,
    given_ops: dict[str, Callable],
    get_binding: Callable,
    start_range: torch.Tensor,
    eval_fn=evaluate,
    pos_outputs: list[tuple[torch.Tensor]] = [],
    num_vals=10,
    max_evals=20,
    num_best: int = 5,
    collect_inner_binding: bool = False,
    lr=1.0,
    loss_threshold: float = 0.1,
    torch_gen: torch.Generator | None = None,
) -> tuple[int, int]:
    """Searches for the term const values that would bring it closer to the target outputs.
    Restarts will reinitialize the constants.
    """

    starts_to_attempt = [pos_outputs]

    rand_points_to_attempt = num_vals - len(starts_to_attempt)
    if rand_points_to_attempt > 0:  # we use grid sampling with rand shifts
        pos_rand_attempt = []
        for _ in optim_state.optim_points:
            rand_points = get_rand_interval_points(
                rand_points_to_attempt, start_range.t(), rand_deltas=True, generator=torch_gen
            )
            pos_rand_attempt.append(rand_points)
        starts_to_attempt.extend(zip(*pos_rand_attempt))

    for op_id, op in enumerate(optim_state.optim_points):
        binding = optim_state.binding[op]
        binding.requires_grad = False
        for opt_id, start_to_attempt in enumerate(starts_to_attempt):
            # for att_id, att in enumerate(start_to_attempt):
            binding[opt_id] = start_to_attempt[op_id]
        binding.requires_grad = True

    optim_res = optimize(
        optim_state,
        loss_fn,
        given_ops,
        get_binding,
        eval_fn=eval_fn,
        loss_threshold=loss_threshold,
        collect_inner_binding=collect_inner_binding,
        lr=lr,
        max_evals=max_evals,
        num_best=num_best,
    )

    return optim_res