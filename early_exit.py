
''' Not finished - left for later 
We consider two version of comparison - 
1.  vectorize which would require checking all dimenstions
2.  early exit which would stop on first mismatch.
'''

import torch

def __eq_v(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    return torch.equal(t1, t2)

def __eq_early_exit(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    # assert t1.shape[-1] == t2.shape[-1], "Number of dimensions should be equal"
    for dim_id in range(t1.shape[-1]):
        if not torch.equal(t1[..., dim_id], t2[..., dim_id]):
            return False
    return True

def __le_v(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return torch.all(t1 <= t2, dim=-1)

def __le_early_exit(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """ Early exit by last dimension """
    assert t1.shape[-1] == t2.shape[-1], "Number of dimensions should be equal"
    dims = t1.shape[-1]
    if len(t1.shape) == 1 and len(t2.shape) == 1: # comparison to one vector 
        t1 = t1.unsqueeze(0)
    if len(t2.shape) == 1:
        mask = torch.ones(t1.shape[:-1], dtype=torch.bool)
        t1v = t1.view(-1, t1.shape[-1])  
        maskv = mask.view(-1)
        cur_mask = maskv
        for dim_id in range(t2.shape[-1]):
            cur_mask = t1v[cur_mask, dim_id] <= t2[dim_id]
            t1v = t1v[cur_mask]
            if t1v.numel() == 0:
                break
    elif len(t1.shape) == 1:
        mask = torch.ones(t2.shape[:-1], dtype=torch.bool)
        t2v = t2.view(-1, t2.shape[-1])  
        maskv = mask.view(-1)
        cur_mask = maskv
        for dim_id in range(t1.shape[-1]):
            cur_mask = t1[dim_id] <= t2v[cur_mask, dim_id]
            t2v = t2v[cur_mask]
            if t2v.numel() == 0:
                break
    else:
        mask = torch.ones(t1.shape[:-1], dtype=torch.bool)
        t1v = t1.view(-1, t1.shape[-1])  
        t2v = t2.view(-1, t2.shape[-1])  
        maskv = mask.view(-1)
        cur_mask = maskv
        for dim_id in range(t2.shape[-1]):
            cur_mask = t1v[cur_mask, dim_id] <= t2v[cur_mask, dim_id]
            t2v = t2v[cur_mask]
            if t2v.numel() == 0:
                break
    return mask

def running_update(tensor: torch.Tensor, mask: torch.Tensor, update: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
                    dim_permutation: Optional[torch.Tensor] = None) -> None:
    ''' Runs through all dimensions in usual order or by dim_permutation (shape (dims))
        On each step (dim_id), updates mask view with tensor view.
        Param tensor shape is (N, dims), mask shape i (N, K)
        Func update params: dim_id, mask_view, tensor_view, where mask_view, tensor_view not just view of dim_id, 
            but also previously filtered by filters produced by previous infovations of func update (return value)
            mask_view shape (L <= N, K), tensor_view shape (L <= N, 1), returns next iteration filter of shape (S <= N)
        The result is in mask, as mask is gradually updated by updater.
    '''
    cur_mask = mask
    cur_tensor = tensor
    for dim_id in (dim_permutation or range(tensor.shape[-1])):
        cur_filter_mask = update(dim_id, cur_mask, cur_tensor[:, dim_id])
        # cur_mask.bitwise_and_(update(dim_id, cur_tensors[cur_mask, dim_id]))
        cur_mask = cur_mask[cur_filter_mask] # viewing only successful passes of predicate
        cur_tensor = cur_tensor[cur_filter_mask]
        if cur_mask.numel() == 0:
            break
    return mask

def find_eq(tensors: torch.Tensor, t: torch.Tensor, rtol=1e-5, atol=1e-4) -> torch.Tensor:
    ''' Find indices where rows matches t.
        tensors shape (N, dims), t shape (K, dims), or (dims).
        returns 0 1 mask of shape (N, K)
    '''
    if len(t.shape) == 1:  # t is a single vector
        t = t.unsqueeze(0)
    tq = t.unsqueeze(0)  # (1, K, dims)
    mask = torch.ones(tensors.shape[0], t.shape[0], dtype=torch.bool, device=tensors.device) # (N, K)
    def _update(dim_id: int, cur_mask: torch.Tensor, cur_tensors: torch.Tensor) -> torch.Tensor:
        cur_close = torch.isclose(cur_tensors.unsqueeze(1), tq[...,dim_id], rtol=rtol, atol=atol) # (L, K)
        cur_mask.bitwise_and_(cur_close)
        return torch.any(cur_close, dim=1) # (L)
    running_update(tensors, mask, _update)
    return mask

def find_in(tensors: torch.Tensor, tmin: torch.Tensor, tmax: torch.Tensor) -> torch.Tensor:
    ''' Find indices where rows  are in between of tmin and tmax, tmin <= row <= tmax.
        tensors shape (N, dims), tmin and tmax shape (dims) or (K, dims).
        Result is of shape (N, K) mask 
    '''
    if len(tmin.shape) == 1:  # t is a single vector
        tmin = tmin.unsqueeze(0)
    if len(tmax.shape) == 1:  # t is a single vector
        tmax = tmax.unsqueeze(0)
    tminq = tmin.unsqueeze(0)  # (1, K, dims)
    tmaxq = tmax.unsqueeze(0)  # (1, K, dims)
    mask = torch.ones(tensors.shape[0], tmin.shape[0], dtype=torch.bool, device=tensors.device) # (N, K)
    def _update(dim_id: int, cur_mask: torch.Tensor, cur_tensors: torch.Tensor) -> torch.Tensor:
        cur_above = (cur_tensors >= tminq[..., dim_id])
        cur_below = (cur_tensors <= tmaxq[..., dim_id])
        cur_in = cur_above & cur_below
        cur_mask.bitwise_and_(cur_in)
        return torch.any(cur_in, dim=1) # (L)    
    running_update(tensors, mask, _update)
    return mask 
