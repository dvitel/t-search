import math
import torch

from .bin import BinIndex


def pack_bits(bits: torch.Tensor, dtype = torch.int64) -> torch.Tensor:
    ''' Splits bits (*ns, m) 0 1 uint8 tensor into tensor of shape (*ns, ceil(m / size(dtype))) '''
    sz = torch.iinfo(dtype).bits
    bitmask = (1 << torch.arange(sz - 1, -1, -1, dtype = dtype, device = bits.device)) # shape [64]
    max_packed_values = math.ceil(bits.shape[-1] / sz) # number of packed values
    packed = []
    # bits_tp = bits.to(dtype=tp, device=bits.device) # convert bits to target type
    for i in range(max_packed_values):
        start = i * sz
        end = start + sz
        if end > bits.shape[-1]:
            end = bits.shape[-1]
        packed_chunk = torch.sum(bits[..., start:end] * bitmask[:end - start], dtype = dtype, dim=-1)
        packed.append(packed_chunk)
    return torch.stack(packed, dim=-1)


def unpack_bits(packed: torch.Tensor, clamp_sz: int) -> torch.Tensor:
    ''' Converts packed (list of shape (n) tensors int64) back to (n, sz) shape uint8 of 0 1 values'''
    sz = torch.iinfo(packed.dtype).bits
    bitmask = (1 << torch.arange(sz - 1, -1, -1, device=packed.device, dtype=packed.dtype))  # Shape: (num_bits,)
    unpacked = ((packed.unsqueeze(-1) & bitmask) != 0).view(*packed.shape[:-1], -1)  # Shape: (N, num_bits), bool tensor
    return unpacked[..., :clamp_sz]  # Convert to uint8 (0/1 values)

# t1 = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0], dtype=torch.uint8)
# t1 = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.uint8)
# t1 = torch.tensor([True, False, True, False], dtype=torch.bool)
# res = pack_bits(t1, dtype=torch.int8)
# t2 = unpack_bits(res, clamp_sz=t1.shape[-1])
# assert torch.equal(t1, t2), "Packing and unpacking failed, tensors are not equal."
# pass
        
class InteractionIndex(BinIndex):
    ''' Maps semantics to binary vector based on dynamically computed epsilons and given target. 
        One dim is one test and 0 means we far from passing the test, 1 - close.
        leaf (one interaction vector bin) splits when it has many semantics
    '''
    def __init__(self, target: torch.Tensor, pack_dtype = torch.int64, **kwargs):
        super().__init__(**kwargs)
        self.epsilons: torch.Tensor = torch.zeros((self.vectors.shape[-1], ), dtype=self.vectors.dtype, device=self.vectors.device)
        sz = torch.iinfo(pack_dtype).bits
        self.int_dims = math.ceil(self.vectors.shape[-1] / sz)
        # self.interactions = torch.zeros((self.capacity, int_dims), dtype=pack_dtype, device=self.vectors.device)
        self.target = target
        self.pack_dtype = pack_dtype
        self.best_interactions = self.get_bin_index(torch.zeros((self.vectors.shape[-1], ), dtype=self.vectors.dtype, device=self.vectors.device), is_distance=True)
        pass
        # self.iid_to_vids: dict[int, list[int]] = {} # interaction id to vector ids
        # self.vid_to_iid: dict[int, int] = {} # vector id to interaction id
    
    def get_bin_index(self, vectors: torch.Tensor, is_distance = False) -> list[tuple]:
        if is_distance:
            assert torch.all(vectors >= 0)
            distances = vectors 
        else:
            distances = torch.abs(vectors - self.target)
        interactions = (distances <= self.epsilons)
        ints = pack_bits(interactions, dtype = self.pack_dtype)
        del interactions, distances
        return ints    
    
    def on_rebuild(self, trigger_bin_id: tuple):
        # vectors = self.get_vectors(self.bins[trigger_bin_id])
        # distances = torch.abs(vectors - self.target)
        # new_epsilons = distances.mean(dim=0)
        # zero = torch.tensor(0, dtype=new_epsilons.dtype, device=new_epsilons.device)
        # self.epsilons = torch.where(torch.isclose(new_epsilons, zero, atol=self.atol, rtol=self.rtol), self.epsilons, new_epsilons)
        # del vectors, distances

        vectors = self.get_vectors(self.bins[trigger_bin_id]) # get all vectors in the bin
        distances = torch.abs(vectors - self.target)
        medians = distances.median(dim = 0).values
        self.epsilons[:] = medians
        # balances = (distances - medians).sum(dim=0)
        # balances.abs_()
        # sort_ids = torch.argsort(balances)
        # approx_num_dims = math.floor(math.log2(distances.shape[0] / self.max_bin_size)) + 1
        # selected_dims = sort_ids[:approx_num_dims] # take only first approx_num_dims dimensions
        # self.epsilons[selected_dims] = medians[selected_dims]

    def get_bins_range(self, qrange: torch.Tensor):
        ''' qrange is 1d distance tensor (dims), distance from target '''
        distant_bin = self.get_bin_index(qrange, is_distance=True)        
        all_bin_ids_list = [bin_id for bin_id in self.bins.keys()]
        all_bin_ids = torch.tensor(all_bin_ids_list, dtype = torch.int64, device=qrange.device)
        not_equal_bits = (all_bin_ids ^ distant_bin) | (all_bin_ids ^ self.best_interactions)
        not_equal_bits.bitwise_not_() # invert bit
        found_ids, = torch.where(torch.all(not_equal_bits == 0, dim=-1))
        selected_bin_ids = all_bin_ids[found_ids]
        selected_bin_ids_list = [tuple(bin_id) for bin_id in selected_bin_ids.tolist()]
        del not_equal_bits, all_bin_ids, found_ids
        new_qrange = torch.stack((self.target - qrange, self.target + qrange), dim=0) # (2, dims)
        return selected_bin_ids_list, new_qrange
    
