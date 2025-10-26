import math
import torch

from .base import find_in_range

from .bin import BinIndex


def get_cos_distance(v1: torch.Tensor, v2: torch.Tensor, v1_norms = None, v2_norms = None,
                        rtol = 1e-5, atol=1e-4, zero=None) -> torch.Tensor:
    if v1_norms is None:
        v1_norms = torch.norm(v1, dim=-1)
    if v2_norms is None:
        v2_norms = torch.norm(v2, dim=-1)
    norm_prod = v1_norms * v2_norms
    if zero is None:
        zero = torch.zeros(1, dtype=v1_norms.dtype, device=v1_norms.device)
    cos_distance = 1 - torch.sum(v1 * v2, dim=-1) / norm_prod
    zero_ids, = torch.where(torch.isclose(norm_prod, zero, atol = atol, rtol=rtol))
    cos_distance[zero_ids] = zero
    return cos_distance

    
class RCosIndex(BinIndex):
    ''' Represents torch.Tensor with only radius vector and cosine distance to target vector 
        Splits spalce onto cones by angle and radius.

        Note: does not work with float16, as too many points land into same range and angle region --> splits epsilon till zeroes
    ''' 
    def __init__(self, target: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        if not torch.is_tensor(target):
            target = torch.full((self.vectors.shape[-1], ), target, dtype=self.vectors.dtype, device=self.vectors.device)
        self.target = target
        self.target_norm = torch.norm(self.target)
        self.zero = torch.zeros_like(self.target_norm)
        assert not torch.isclose(self.target_norm, self.zero, atol=self.atol, rtol=self.rtol), "Target vector cannot be zero vector."
        # if torch.is_tensor(epsilons):
        #     self.epsilons = epsilons
        # else:
        self.epsilons = torch.tensor([1, 1], dtype=self.vectors.dtype, device=self.vectors.device)

    def get_bin_index(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Get bin index for a given vector '''
        norms = torch.norm(vectors, dim=-1)
        cos_distance = get_cos_distance(vectors, self.target, v1_norms=norms, v2_norms=self.target_norm, 
                                        rtol=self.rtol, atol=self.atol, zero=self.zero)
        norm_bin_index = torch.floor(norms / self.epsilons[0]).to(dtype=torch.int64)
        cos_bin_index = torch.floor(cos_distance / self.epsilons[1]).to(dtype=torch.int64)
        bin_index = torch.stack((norm_bin_index, cos_bin_index), dim=-1) # shape (n, 2)
        del norms, cos_distance
        return bin_index
    
    def on_rebuild(self, trigger_bin_id: tuple):
        # bin_tensor = self.get_vectors(self.bins[trigger_bin_id])
        # norms = torch.norm(bin_tensor, dim=1)
        # cos_distance = get_cos_distance(bin_tensor, self.target, v1_norms=norms, v2_norms=self.target_norm,
        #                                 rtol=self.rtol, atol=self.atol, zero=self.zero)

        approx_num_dims = math.floor(math.log2(len(self.bins[trigger_bin_id]) / self.max_bin_size)) + 1

        nums_dims = math.ceil(approx_num_dims / 2)

        self.epsilons /= 2 ** nums_dims 
        pass

        # norms_median = norms.median()
        # norms_start = trigger_bin_id[0] * self.epsilons[0]
        # new_norm_epsilon = torch.maximum(norms_median - norms_start, norms_start + self.epsilons[0] - norms_median)
        # if new_norm_epsilon <= 0:
        #    new_norm_epsilon = self.epsilons[0] / 2 
        # cos_median = cos_distance.median()
        # cos_start = trigger_bin_id[1] * self.epsilons[1]
        # new_cos_epsilon = torch.maximum(cos_median - cos_start, cos_start + self.epsilons[1] - cos_median)
        # if new_cos_epsilon <= 0:
        #     new_cos_epsilon = self.epsilons[1] / 2
        # self.epsilons = torch.tensor([new_norm_epsilon, new_cos_epsilon], dtype=self.vectors.dtype, device=self.vectors.device)
        pass
        # epsilon_ids, = torch.where(~torch.isclose(new_epsilon, self.zero, atol=self.atol, rtol=self.rtol))
        # self.epsilons[epsilon_ids] = new_epsilon[epsilon_ids]

    def get_bins_range(self, qrange):
        ''' qrange is 2 reference vectors that specify radius and angle from target '''
        bin_qrange = self.get_bin_index(qrange)
        bin_qrange[:] = torch.sort(bin_qrange, dim=0).values
        all_bin_ids_list = [bin_id for bin_id in self.bins.keys()]
        all_bin_ids = torch.tensor(all_bin_ids_list, dtype = torch.int64, device=qrange.device)
        bin_id_ids = find_in_range(all_bin_ids, bin_qrange)
        selected_bin_ids = [all_bin_ids_list[i] for i in bin_id_ids.tolist()]
        del bin_id_ids, all_bin_ids, bin_qrange
        return selected_bin_ids, None        