from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Optional

import torch

from t_search.datasets.sampling import get_rand_interval_points
from t_search.syntax import Term, Value
from t_search.syntax.term import Variable


@dataclass(frozen=True)
class OptimPoint(Term):
    point_id: int  # optim point in root term

class LRAdjust(Exception):
    pass

optim_id = -1  # for debugging

def optimize(
    optim_term: Term,
    start_range: torch.Tensor,
    start_binding: dict[OptimPoint, torch.Tensor],
    loss_fn_builder: Callable,
    *,
    num_starts: int = 10,
    lr: float = 1.0,
    max_evals: int = 10,
    tolerance_change: float = 1e-6,
    tolerance_grad: float = 1e-3,
    torch_gen: torch.Generator | None = None,
    num_best_binings: int = 1,
    debug: bool = False
    # loss_threshold: float = 0.1,
) -> tuple[torch.Tensor | None, dict[Term, torch.Tensor] | None]:
    global optim_id
    
    # assert optim_state.loss_fn is not None, "Optimization loss function is not set"
        
    if debug:
        optim_id += 1

        print(f">>> [{optim_id}] {optim_term}")

    params = []
    binding = {}
    for optim_point, optim_value in start_binding.items():
        value = torch.zeros(
            (num_starts, 1 if len(optim_value.shape) == 0 else optim_value.shape[0]), 
            dtype=optim_value.dtype, device=optim_value.device
        )
        value[0] = optim_value

        if num_starts > 1:
            rand_points = get_rand_interval_points(
                num_starts-1, start_range,
                rand_deltas=True, generator=torch_gen
            )
            for rp_id, rp in enumerate(rand_points):
                value[rp_id+1:] = rp
                
        value.requires_grad_(True)
        binding[optim_point] = value
        params.append(value)

    # print(f"\t === {optim_state.max_tries} {cur_lr}")

    def get_binding(root: Term, term: Term):
        if isinstance(term, OptimPoint):
            return binding[term]
        # NOTE: we still allow evaluator _get_binding to speedup computations

    loss_fn = loss_fn_builder(get_binding=get_binding)

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
    best_binding = {}

    # iter_loss = []
    # iter_binding = {}

    # if optim_state.best_loss is not None:
    #     # iter_loss.append(optim_state.best_loss)
    #     best_loss = optim_state.best_loss

    # if optim_state.best_binding is not None:
    #     best_binding = dict(optim_state.best_binding)
    #     # for k, v in optim_state.best_binding.items():
    #     #     iter_binding[k] = [v]

    num_root_evals = 0

    def closure_builder(optimizer: torch.optim.Optimizer):
        nonlocal best_loss, debug, max_evals, num_root_evals

        # cur_lr = optimizer.param_groups[0]['lr']
        # print(f"LR: {cur_lr}")        
        if num_root_evals >= max_evals:
            raise LRAdjust(None)
        optimizer.zero_grad()


        loss: torch.Tensor = loss_fn(optim_term, binding=binding)
        num_root_evals += 1
        fixed_loss = loss.nan_to_num_(torch.inf)
        # finite_loss_mask = torch.isfinite(loss)
        if not torch.all(torch.isfinite(fixed_loss)):
            raise LRAdjust(None)

        if num_best_binings == 1:
            loss_min_pos = fixed_loss.argmin()
            min_loss = fixed_loss[loss_min_pos]

            if debug:
                print(f"\tLoss {min_loss.item()}, evals {num_root_evals}")

            # if min_loss < loss_threshold:
            #     iter_loss.append(loss.detach().clone())
            #     for k, v in optim_state.binding.items():
            #         iter_binding.setdefault(k, []).append(v.detach().clone())

            if best_loss is None or min_loss < best_loss:
                best_loss = min_loss.detach().clone()
                for k, v in binding.items():
                    best_binding[k] = v[loss_min_pos].detach().clone()
        else: # best_loss is 1d tensor of size num_best_binings and best_binding is 2d (best_binding, values)
            if best_loss is None: # take num_best_binings best 
                sort_ids = torch.argsort(fixed_loss)
                best_sort_ids = sort_ids[:num_best_binings]
                best_loss = fixed_loss[best_sort_ids].detach().clone()
                for k, v in binding.items():
                    best_binding[k] = v[best_sort_ids].detach().clone()
                del sort_ids, best_sort_ids
            else: # need to combine current and prev best_loss 
                both_loss = torch.cat([best_loss, fixed_loss], dim=0)
                sort_ids = torch.argsort(both_loss)
                best_sort_ids = sort_ids[:num_best_binings]
                if any(best_sort_ids >= best_loss.shape[0]):
                    # some new losses are among best 
                    best_loss = both_loss[best_sort_ids].detach().clone()
                    for k, v in binding.items():
                        both_bindings = torch.cat([best_binding[k], v], dim=0)
                        best_binding[k] = both_bindings[best_sort_ids].detach().clone()
                        del both_bindings
                    del both_loss

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

    # if best_loss is not None:

    #     optim_state.best_loss = best_loss
    #     optim_state.best_binding = best_binding

    for p in params:
        del p.grad
        del p

    return (best_loss, best_binding)

def get_all_grads(term: Term,
                  var_bindings: dict[str, torch.Tensor],
                  get_loss_fn: Callable,
                  dtype: torch.dtype = torch.float32,
                  device: Literal["cpu", "cuda"] = "cpu") -> dict[tuple[Term, int], torch.Tensor]:
    ''' Collecting gradients of loss w.r.t each position in Term '''
    collected_term_pos: dict[tuple[Term, int], torch.Tensor] = {}
    occurs: dict[Term, int] = {}
    def get_binding(root: Term, term: Term) -> Optional[torch.Tensor]:
        if isinstance(term, Value):
            outputs = torch.tensor(term.value, dtype=dtype, device=device, requires_grad=True)
        elif isinstance(term, Variable):
            outputs = var_bindings[term.var_id].clone().detach()
            outputs = outputs.to(dtype=dtype, device=device)
            outputs.requires_grad_(True)
        if outputs is not None:
            cur_occur = occurs.setdefault(term, 0)
            collected_term_pos[(term, cur_occur)] = outputs
            occurs[term] = cur_occur + 1
        return outputs
    def set_binding(root: Term, term: Term, value: torch.Tensor) -> None:
        cur_occur = occurs.setdefault(term, 0)
        collected_term_pos[(term, cur_occur)] = value
        value.requires_grad_(True)
        occurs[term] = cur_occur + 1        
    
    loss_fn = get_loss_fn(get_binding=get_binding, set_binding=set_binding, no_cache=True)

    loss = loss_fn(term)
    loss.backward()

    grads: dict[tuple[Term, int], torch.Tensor] = {}
    for (t, occ), binding in collected_term_pos.items():
        if binding.grad is not None:
            grads[(t, occ)] = binding.grad.clone()
        else:
            raise ValueError(f"Gradient for term {t} occur {occ} is None")
        
    return grads



