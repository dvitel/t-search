import math
import torch

from .bin import BinIndex


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_rank = torch.argsort(torch.argsort(x)).to(dtype=x.dtype)
    y_rank = torch.argsort(torch.argsort(y)).to(dtype=y.dtype)

    # Compute Pearson correlation on the ranks
    x_mean = x_rank.mean(dim=-1).unsqueeze(-1)  # Ensure x_mean is a column vector
    y_mean = y_rank.mean(dim=-1).unsqueeze(-1)  # Ensure y_mean is a column vector
    numerator = torch.sum((x_rank - x_mean) * (y_rank - y_mean), dim=-1)
    denominator = torch.sqrt(torch.sum((x_rank - x_mean)**2) * torch.sum((y_rank - y_mean)**2, dim=-1))

    cor = numerator / denominator
    return cor
        
class SpearmanCorIndex(BinIndex):
    ''' vector to spearman correlation with target 
    
        NOTE: reduction of dims to 1 leads to very small epsilon, machine error and fails due to float precision.
        Use higher number of children is required here: max_children = 1000-10000,
        Works only with float32 (see RCosIndex problem)

        NOTE: very slow - same to absent index situation, 
             Many points land to small region.
        FIXME: ??epsilon should not be constant, starting from small value it should increase
               ??Do not hard fix max_children for this index - make it relaxed - max children only in near target region
    ''' 

    def __init__(self, target: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = 1
        if not isinstance(target, torch.Tensor):
            self.target = torch.full((self.vectors.shape[-1], ), target, dtype=self.vectors.dtype, device=self.vectors.device)
        else:
            self.target = target
    
    def get_bin_index(self, vectors: torch.Tensor) -> torch.Tensor:
        cors: torch.Tensor = 1 - spearman_correlation(vectors, self.target)
        bin_id = torch.floor(cors / self.epsilon).to(dtype=torch.int64)
        return bin_id.unsqueeze(-1)
    
    def on_rebuild(self, trigger_bin_id: tuple):
        # bin_tensor = self.get_vectors(self.bins[trigger_bin_id])

        approx_num_dims = math.floor(math.log2(len(self.bins[trigger_bin_id]) / self.max_bin_size)) + 1

        nums_dims = math.ceil(approx_num_dims / 2)

        self.epsilon /= 2 ** nums_dims 

        # cors: torch.Tensor = spearman_correlation(bin_tensor, self.target)
        # cors.abs_()
        # cor_start = trigger_bin_id[0] * self.epsilon
        # cor_end = (trigger_bin_id[0] + 1) * self.epsilon
        # cor_median = torch.median(cors).item()
        # self.epsilon = max(cor_end - cor_median, cor_median - cor_start)
        pass

    def get_bins_range(self, qrange):
        ''' qrange is distance in cor index - 2d tensor (1, 2), start-end of expected cor '''
        cor_delta = torch.floor(qrange / self.epsilon).to(dtype=torch.int64)
        min_bin_id = (cor_delta[0, 0].item(), )
        max_bin_id = (cor_delta[1, 0].item(), )
        selected_bin_ids = [bin_id for bin_id in self.bins.keys() if min_bin_id <= bin_id <= max_bin_id]
        return selected_bin_ids, None