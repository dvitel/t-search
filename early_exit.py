
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