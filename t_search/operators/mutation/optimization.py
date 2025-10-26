from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import torch

from t_search.datasets.sampling import get_interval_grid, get_rand_interval_points
from syntax import Term, TermPos, Value, evaluate
from syntax.generation import Builders
from syntax.replacement import replace_fn, replace_pos


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


optim_id = -1  # for debugging


def optimize(
    optim_state: OptimState,
    loss_fn: Callable,
    given_ops: dict[str, Callable],
    get_binding: Callable,
    *,
    eval_fn=evaluate,
    num_best: int = 1,
    lr: float = 1.0,
    max_evals: int = 10,
    collect_inner_binding: bool = False,
    loss_threshold: float = 0.1,
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

    optimizer = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=max_evals,
        max_eval=max_evals,
        # max_eval = 1.5 * num_steps,
        tolerance_change=1e-6,  # TODO - should be parameters???
        tolerance_grad=1e-3,
        # history_size=100,
        line_search_fn="strong_wolfe",
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

        (finite_loss_ids,) = torch.where(finite_loss_mask)

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
            for k, v in optim_state.binding.items():
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

    if len(iter_loss) > 0:

        all_iter_loss = torch.concat(iter_loss)
        all_iter_loss.nan_to_num_(torch.inf)
        if num_best == 1:
            best_ids = torch.argmin(all_iter_loss).unsqueeze(0)
        else:
            best_ids = torch.argsort(all_iter_loss)[:num_best]
        best_loss = all_iter_loss[best_ids]
        (best_id_ids,) = torch.where(best_loss < loss_threshold)
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


def optimize_consts(
    term: Term,
    term_loss: torch.Tensor,
    loss_fn: Callable,
    builders: Builders,
    given_ops: dict[str, Callable],
    get_binding: Callable,
    start_range: torch.Tensor,
    *,
    eval_fn=evaluate,
    num_vals=10,
    max_tries=1,
    max_evals=20,
    num_best: int = 1,
    lr=0.1,
    loss_threshold: float = 0.1,
    torch_gen: torch.Generator | None = None,
    term_values_cache: dict[Term, list[Value]],
    optim_term_cache: dict[Term, Term | None],
    optim_state_cache: dict[Term, OptimState],
) -> Optional[tuple[OptimState, int, int]]:
    """Searches for the term const values that would bring it closer to the target outputs.
    Restarts will reinitialize the constants.
    """

    if term not in optim_term_cache:  # need to build optim term with optim points

        optim_points: list[OptimPoint] = []
        binding = {}
        values = []

        def const_to_optim_point(term, *_):
            if isinstance(term, Value):
                point_id = len(optim_points)
                point = OptimPoint(point_id)
                optim_points.append(point)
                value = torch.zeros(
                    (num_vals, 1 if len(term.value.shape) == 0 else term.value.shape[0]),
                    dtype=term.value.dtype,
                    device=term.value.device,
                )
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
    if (optim_state.best_loss is None) or (
        term_loss < torch.min(optim_state.best_loss)
    ):  # at first try we also optimize current values
        starts_to_attempt.append([v.value for v in term_values_cache[term]])
        rand_points_to_attempt -= 1

    if rand_points_to_attempt > 0:  # we use grid sampling with rand shifts
        should_del_ranges = False
        if len(start_range.shape) == 1:  # 1d range
            should_del_ranges = True
            start_range = torch.tile(start_range, (len(optim_state.optim_points), 1))
        steps = (start_range[:, 1] - start_range[:, 0]) / (rand_points_to_attempt + 1)
        rand_points = get_interval_grid(steps, start_range, rand_deltas=True, generator=torch_gen)
        if rand_points.shape[0] > rand_points_to_attempt:
            selected_ids = torch.randperm(rand_points.shape[0], device=rand_points.device, generator=torch_gen)[
                :rand_points_to_attempt
            ]
            new_rand_points = rand_points[selected_ids, :]
            del rand_points
            rand_points = new_rand_points
        starts_to_attempt.extend([[v for v in p] for p in rand_points])
        if should_del_ranges:
            del start_range

    const_vectors = []
    for point in optim_state.optim_points:
        const_values = torch.tensor(
            [[p[point.point_id]] for p in starts_to_attempt], device=term_loss.device, dtype=term_loss.dtype
        )
        const_vectors.append(const_values)

    for p, cv in zip(optim_state.optim_points, const_vectors):
        binding = optim_state.binding[p]
        binding.requires_grad = False
        binding.copy_(cv)  # copy new value to optim point
        binding.requires_grad = True

    best_loss_before = optim_state.best_loss if optim_state.best_loss is not None else None

    num_evals, num_root_evals = optimize(
        optim_state,
        loss_fn,
        given_ops,
        get_binding,
        eval_fn=eval_fn,
        loss_threshold=loss_threshold,
        collect_inner_binding=False,
        lr=lr,
        max_evals=max_evals,
        num_best=num_best,
    )

    if optim_state.best_loss is not None and (
        best_loss_before is None or optim_state.best_loss[0] < best_loss_before[0]
    ):

        def bind_optim_points(term, occur, **_):
            if isinstance(term, OptimPoint):
                return Value(optim_state.best_binding[term][0])

        optim_state.best_term = replace_fn(optim_state.optim_term, bind_optim_points, builders)

        optim_term_cache[optim_state.best_term] = optim_state.optim_term

    return optim_state, num_evals, num_root_evals


def get_pos_optim_state(
    term: Term,
    positions: list[TermPos],
    *,
    optim_term_cache: dict[tuple[Term, tuple[Term, int]], Term | None],
    optim_state_cache: dict[Term, OptimState],
    builders: Builders,
    num_vals: int = 10,
    output_size: int = 1,
    max_tries: int = 1,
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