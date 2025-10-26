from typing import Callable
import torch


def mse_loss_builder(target):
    return lambda output: torch.mean((output - target) ** 2, dim=-1)


def nmse_loss_builder(target) -> Callable[[torch.Tensor], torch.Tensor]:
    """we follow R^2 normalization: NMSE = 1 - R^2"""
    # norm = torch.mean(target ** 2, dim=-1) # TODO: could be different norms: std dev
    norm = torch.var(target, dim=-1, unbiased=False)

    # mse = torch.mean((output - target) ** 2, dim=-1)
    def loss_fn(output: torch.Tensor) -> torch.Tensor:
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

def l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    el_dist = (a - b) ** 2
    el_dist.nan_to_num_(nan=torch.inf)
    return torch.sqrt(torch.sum(el_dist, dim=-1))