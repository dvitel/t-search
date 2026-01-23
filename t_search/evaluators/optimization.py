from dataclasses import dataclass, field
from functools import partial
from itertools import product
from math import prod
from typing import Callable, Literal, Optional

import torch

from t_search.datasets.sampling import get_rand_interval_points
from t_search.syntax.term import Term, Value, Variable


@dataclass(frozen=True)
class OptimPoint(Term):
    point_id: int  # optim point in root term

class LRAdjust(Exception):
    pass

# optim_id = -1  # for debugging

@dataclass(frozen=False)
class OptimResult:
    loss: torch.Tensor # one loss per one test and one start (num_starts, num_tests)
    binding: dict[OptimPoint, torch.Tensor] # per pos of interest, binding (num_starts, num_tests)
    additional_binding: dict[tuple[Term, int], torch.Tensor] = field(default_factory=dict)
    def agg_loss(self):
        return self.loss.mean(dim=-1)  # (num_starts,) or (1,)

# TODO 1: optimize path root --> OptimPoint(s)
# TODO 2: return optimization per test, select N slowest solution functions. 
# OPTIONAL: operator that find contant at point 

def optimize(
    optim_term: Term,
    start_range: torch.Tensor,
    start_binding: dict[OptimPoint, torch.Tensor],
    loss_fn_builder: Callable,
    *,
    pos_to_collect: set[tuple[Term, int]] = set(),
    num_starts: int = 10,
    lr: float = 1.0,
    max_evals: int = 10,
    tolerance_change: float = 1e-6,
    tolerance_grad: float = 1e-3,
    torch_gen: torch.Generator | None = None,
    # debug: bool = False
    # loss_threshold: float = 0.1,
) -> OptimResult:
    # global optim_id
    
    # assert optim_state.loss_fn is not None, "Optimization loss function is not set"
        
    # if debug:
    #     optim_id += 1

    #     print(f">>> [{optim_id}] {optim_term}")

    params = []
    binding = {}
    for optim_point, optim_value in start_binding.items():
        value = torch.zeros(
            (num_starts, start_range.shape[0]), 
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

    additional_binding = {}
    occurs = {}
    def set_binding(root: Term, term: Term, value: torch.Tensor):
        ''' NOTE: only path to OptimPoint should be set here '''
        cur_occur = occurs.setdefault(term, 0)
        if (term, cur_occur) in pos_to_collect:
            additional_binding[(term, cur_occur)] = value.detach().clone()
            occurs[term] = cur_occur + 1
        return

    loss_fn = loss_fn_builder(get_binding=get_binding, set_binding=set_binding)

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

    best_loss = torch.full((num_starts, start_range.shape[0]), torch.inf, dtype=start_range.dtype, device=start_range.device)
    best_binding = {k:v.detach().clone() for k, v in binding.items()}
    best_additional_binding = {k: None for k in additional_binding}

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
        nonlocal best_loss, max_evals, num_root_evals

        # cur_lr = optimizer.param_groups[0]['lr']
        # print(f"LR: {cur_lr}")        
        if num_root_evals >= max_evals:
            raise LRAdjust(None)
        optimizer.zero_grad()

        occurs.clear()
        loss: torch.Tensor = loss_fn(optim_term)
        num_root_evals += 1
        fixed_loss = loss.nan_to_num_(torch.inf)
        # finite_loss_mask = torch.isfinite(loss)

        # if not torch.all(torch.isfinite(fixed_loss)):
        #     raise LRAdjust(None)

        where_better = fixed_loss < best_loss
        best_loss.data[where_better] = fixed_loss[where_better]
        for k, v in binding.items():
            best_binding[k].data[where_better] = v[where_better]
        for k, v in additional_binding.items():
            if k in best_additional_binding:
                best_additional_binding[k].data[where_better] = additional_binding[k][where_better]                
            else:
                best_additional_binding[k] = additional_binding[k].clone()
        for v in additional_binding.values():
            del v
        # if debug:
        #     print(f"{num_root_evals}\tL {' '.join([f'{f:.1e}' for f in best_loss.mean(dim=1).tolist()])}")        
        
        # if num_best_binings == 1:
        #     loss_min_pos = fixed_loss.argmin()
        #     min_loss = fixed_loss[loss_min_pos]

        #     if debug:
        #         print(f"\tLoss {min_loss.item()}, evals {num_root_evals}")

        #     # if min_loss < loss_threshold:
        #     #     iter_loss.append(loss.detach().clone())
        #     #     for k, v in optim_state.binding.items():
        #     #         iter_binding.setdefault(k, []).append(v.detach().clone())

        #     if best_loss is None or min_loss < best_loss:
        #         best_loss = min_loss.detach().clone()
        #         for k, v in binding.items():
        #             best_binding[k] = v[loss_min_pos].detach().clone()
        # else: # best_loss is 1d tensor of size num_best_binings and best_binding is 2d (best_binding, values)
        #     if best_loss is None: # take num_best_binings best 
        #         sort_ids = torch.argsort(fixed_loss)
        #         best_sort_ids = sort_ids[:num_best_binings]
        #         best_loss = fixed_loss[best_sort_ids].detach().clone()
        #         for k, v in binding.items():
        #             best_binding[k] = v[best_sort_ids].detach().clone()
        #         del sort_ids, best_sort_ids
        #     else: # need to combine current and prev best_loss 
        #         both_loss = torch.cat([best_loss, fixed_loss], dim=0)
        #         sort_ids = torch.argsort(both_loss)
        #         best_sort_ids = sort_ids[:num_best_binings]
        #         if any(best_sort_ids >= best_loss.shape[0]):
        #             # some new losses are among best 
        #             best_loss = both_loss[best_sort_ids].detach().clone()
        #             for k, v in binding.items():
        #                 both_bindings = torch.cat([best_binding[k], v], dim=0)
        #                 best_binding[k] = both_bindings[best_sort_ids].detach().clone()
        #                 del both_bindings
        #             del both_loss

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

        # finite_loss = fixed_loss[torch.isfinite(fixed_loss)]
        # total_loss = finite_loss.mean()
        total_loss = fixed_loss.sum() # use all losses including inf
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

    for p in params:
        del p.grad
        del p

    res = OptimResult(best_loss, best_binding, best_additional_binding)

    return res

def clean_optim_result(optim_result: OptimResult) -> None:
    del optim_result.loss
    for v in optim_result.binding.values():
        del v
    for v in optim_result.additional_binding.values():
        del v   

def gather_optim_pos(optim_result: OptimResult, pos: torch.Tensor) -> OptimResult:
    ''' pos has shape (k, num_tests) - specicify what loss and binding to collect 
        k is frequently 1
    '''
    best_loss = torch.gather(optim_result.loss, dim=0, index=pos) # (k, num_tests)
    best_binding = {}    
    for k, v in optim_result.binding.items():
        new_v = torch.gather(v, dim=0, index=pos)
        best_binding[k] = new_v
    additional_binding = {}
    for k, v in optim_result.additional_binding.items():
        if v is None:
            additional_binding[k] = None
            continue
        new_v = torch.gather(v, dim=0, index=pos)
        additional_binding[k] = new_v
    res = OptimResult(best_loss, best_binding, additional_binding)
    return res

def get_best_optim_result(optim_result: OptimResult) -> OptimResult:
    ''' Extract smallest loss positions.
        optim_result is 2d tensor (num_starts, num_tests)
    '''
    min_pos = optim_result.loss.argmin(dim=0, keepdim=True) # (1, num_tests)
    res = gather_optim_pos(optim_result, min_pos)
    return res

def get_slowest_funs_dp(fit_ids: list[torch.Tensor],
                        optim_result: OptimResult,
                        max_num_funs: int) -> OptimResult:
    ''' Dynamic programming routine to find smallest variational trace '''
    dists = []
    num_tests = len(fit_ids)
    max_fits = max(len(ids) for ids in fit_ids)
    for test_i in range(num_tests-1):
        cur_ids = fit_ids[test_i]
        next_ids = fit_ids[test_i+1]
        dist_matrix = torch.zeros((len(cur_ids), len(next_ids)),
                                  dtype=optim_result.loss.dtype,
                                  device=optim_result.loss.device)
        for k, v in optim_result.binding.items():
            dist_matrix += torch.abs(
                v[cur_ids, test_i].unsqueeze(1) - v[next_ids, test_i + 1].unsqueeze(0)
            )            
        dist_matrix /= len(optim_result.binding)
        dists.append(dist_matrix) # (len(cur_ids), len(next_ids))

    # TV represents minimal total variation: note that we search for max_num_funs smallest series
    TV = torch.full((num_tests, max_fits, max_num_funs), torch.inf, 
                   dtype=optim_result.loss.dtype, device=optim_result.loss.device)    
    Paths = torch.full((num_tests, max_fits, max_num_funs), -1, 
                      dtype=torch.long, device=optim_result.loss.device)
    PathRanks = torch.full((num_tests, max_fits, max_num_funs), -1,
                          dtype=torch.long, device=optim_result.loss.device)        
    
    TV[0, :len(fit_ids[0]), 0] = 0.0  # No variation yet at first test

    # forward dynamic programming pass
    for test_i in range(1, num_tests):
        prev_fits = fit_ids[test_i - 1]
        curr_fits = fit_ids[test_i]
        cur_dists = dists[test_i - 1]  # (len(prev_fits), len(curr_fits))
        cur_TV = TV[test_i] # (max_fits, max_num_funs)
        prev_TV = TV[test_i - 1, :len(prev_fits)] # (len(prev_fits), max_num_funs)

        new_TV = prev_TV.unsqueeze(2) + cur_dists.unsqueeze(1) # (len(prev_fits), max_num_funs, len(curr_fits))
        all_new_TVs_flat = new_TV.reshape(-1, len(curr_fits))  # (num_prev * max_num_funs, len(curr_fits))

        top_k_values, top_k_indices = torch.topk(
            all_new_TVs_flat, 
            k=max_num_funs, 
            dim=0,  # Top-k along the candidates dimension
            largest=False
        )  # (max_num_funs, len(curr_fits))       

        TV[test_i, :len(curr_fits), :] = top_k_values.T

        prev_fit_indices = top_k_indices // max_num_funs
        prev_rank_indices = top_k_indices % max_num_funs 

        Paths[test_i, :len(curr_fits), :] = prev_fit_indices.T 
        PathRanks[test_i, :len(curr_fits), :] = prev_rank_indices.T
          
    # backward pass to reconstruct paths
    num_final = len(fit_ids[-1])
    final_candidates = TV[-1, :num_final, :].reshape(-1) # (num_final, max_num_funs)
    num_to_select = min(max_num_funs, (final_candidates < torch.inf).sum().item())
    final_candidate_values, final_candidate_ids = torch.topk(final_candidates, k=num_to_select, largest=False)
    final_candidate_real_ids = final_candidate_ids // max_num_funs # (max_num_funs,)
    final_rank_indices = final_candidate_ids % max_num_funs

    paths = torch.full((num_to_select, num_tests), -1, 
                      dtype=torch.long, device=optim_result.loss.device)
    
    paths[:, -1] = final_candidate_real_ids
    curr_fits = final_candidate_real_ids
    curr_ranks = final_rank_indices

    for test_i in range(num_tests - 1, 0, -1):
        # Get previous fit indices for all paths
        # Paths[test_i, curr_fits[path_id], curr_ranks[path_id]]
        prev_fits = Paths[test_i, curr_fits, curr_ranks]  # (num_to_select,)
        prev_ranks = PathRanks[test_i, curr_fits, curr_ranks]  # (num_to_select,)
        
        # Check for invalid paths
        if torch.any(prev_fits == -1):
            # Some paths are invalid - need to handle this
            valid_mask = prev_fits != -1
            if not valid_mask.any():
                return None
            # Filter to valid paths only
            paths = paths[valid_mask]
            curr_fits = prev_fits[valid_mask]
            curr_ranks = prev_ranks[valid_mask]
            num_to_select = paths.shape[0] 
        else:
            curr_fits = prev_fits
            curr_ranks = prev_ranks
        
        # Store previous fit indices
        paths[:, test_i - 1] = curr_fits    

    actual_paths = torch.empty_like(paths)
    for test_i in range(num_tests):
        actual_paths[:, test_i] = fit_ids[test_i][paths[:, test_i]]

    res = gather_optim_pos(optim_result, actual_paths)
        
    return res   

def threshold_optim_result_(optim_result: OptimResult,
                           loss_threshold: float = 1e-3) -> OptimResult:
    '''
    Sets loss above threshold to torch.inf in optim_result in-place
    '''
    mask = optim_result.loss >= loss_threshold
    optim_result.loss[mask] = torch.inf
    return optim_result

def get_slowest_funs(optim_result: OptimResult, 
                     max_num_funs: int = 1,
                     high_num_funs: int = 1024, # we may attempt to generate this num of funcs and then pick max_num_funs
                     ) -> OptimResult | None:
    '''
    For given combinatorial semantics in optim_result.binding,
    finds slowest functions which fits the loss requirement optim_result.loss < loss_threshold

    Returns: loss of selected points, constructed functions in binding and additional_binding     
             None if no fit points for every test
    '''

    # TODO: operations should be vectorized
    num_tests = optim_result.loss.shape[-1]
    fit_mask = torch.isfinite(optim_result.loss) # (num_starts, num_tests)
    num_fit = torch.sum(fit_mask, dim=0)
    if torch.any(num_fit == 0):
        return None # some tests cannot be fitted 
    if torch.all(num_fit == 1): # exactly one function 
        pos = torch.argmax(fit_mask.long(), dim=0, keepdim=True) # (1, num_tests)
        res = gather_optim_pos(optim_result, pos)
        return res 
    # there could be combinations - we check distances to neighbours
    fit_ids = [torch.where(fit_mask[:, test_i])[0] for test_i in range(num_tests)]
    num_funcs = prod(len(l) for l in fit_ids)
    if num_funcs <= max_num_funs: # return all possible binding combinations 
        pos = torch.cartesian_prod(*fit_ids) #(num_funcs, num_tests)
        res = gather_optim_pos(optim_result, pos)
        del pos
        return res 
    # here we need to pick some funcs only from variety of posible combinations
    if num_funcs <= high_num_funs: # we generate all and then pick by smallest variation
        pos = torch.cartesian_prod(*fit_ids) #(num_funcs, num_tests)
        intermediate_res = gather_optim_pos(optim_result, pos)
        variations = torch.zeros(num_funcs, 
                                 dtype=optim_result.loss.dtype, 
                                 device=optim_result.loss.device)
        for v in intermediate_res.binding.values():
            total_variation = torch.abs(v[:, 1:] - v[:, :-1]).sum(dim=1) # (num_funcs,)
            variations += total_variation #torch.var(v, dim=1) # (num_funcs,)
        best_ids = torch.topk(variations, k=max_num_funs, largest=False).indices
        best_pos = pos[best_ids]
        res = gather_optim_pos(optim_result, best_pos)
        del pos, best_pos, best_ids, variations
        return res 
    # this case is the most difficult, we have alot of funcs to pick from
    # we resort to random exploration for some time to find smallet by variation functions 

    res = get_slowest_funs_dp(fit_ids, optim_result, max_num_funs)

    return res
         
def set_local_minimas_(optim_result: OptimResult,
                        rtol: float = 1e-3,
                        atol: float = 1e-4) -> None:
    '''
        For each test in (num_start, num_tests) of optim_result.binding, picks unique local minimas:
        1. Smallest by loss
        2. torch.inf to closest to smallest by loss
        3. Repeats until no more local minimas found
    '''
    # local_minima_vectors = []
    # local_minimas = []

    sort_ids = torch.argsort(optim_result.loss, dim=0)  # (num_starts, num_tests)
    sorted_loss = torch.gather(optim_result.loss, dim=0, index=sort_ids)  # (num_starts, num_tests)
    vectors = [
        torch.gather(v, dim=0, index=sort_ids)  # (num_starts, num_tests)
        for v in optim_result.binding.values()
    ] 
    assert len(vectors) > 0, "No binding vectors in optim_result"

    for start_id in range(1, sort_ids.shape[0]): # check if previous starts are close by binding vectors and set the loss to inf in such cases
        prev_values = [v[:start_id] for v in vectors] # (start_id, num_tests) each
        cur_value = [v[start_id:start_id+1] for v in vectors] #(1, num_tests) each 
        are_close = [ torch.isclose(cv, pv, rtol=rtol, atol=atol) 
                      for cv, pv in zip(cur_value, prev_values)]
        close_mask = are_close[0]
        for ac in are_close[1:]:
            close_mask &= ac 
        start_close_mask = torch.any(close_mask, dim=0)  # (num_tests,)
        start_close_index = sort_ids[start_id]
        # scatter inf according to start_close_mask
        new_loss = sorted_loss[start_id]
        new_loss[start_close_mask] = torch.inf
        optim_result.loss.scatter_(0, start_close_index.unsqueeze(0), new_loss.unsqueeze(0))
        del close_mask, start_close_mask, are_close, prev_values, cur_value
    del sort_ids, vectors, sorted_loss

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



