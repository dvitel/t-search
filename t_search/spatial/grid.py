
import math
import torch
from .bin import BinIndex


class GridIndex(BinIndex):
    ''' Grid-based spatial index for approximate NN searc.
        Splits space onto bins of fixed size. Works only with points.
        Rebuild scales down the grid to satisfy max bin size in number of points.
    '''
    
    def __init__(self, epsilons: float | torch.Tensor = 1e-3, **kwargs):
        super().__init__(**kwargs)
        if not torch.is_tensor(epsilons):
            self.epsilons = torch.full((self.vectors.shape[-1], ), epsilons, dtype=self.vectors.dtype, device=self.vectors.device)
        else:
            self.epsilons = epsilons

    def get_bin_index(self, vectors: torch.Tensor) -> torch.Tensor:
        ''' Get bin index for a given vector '''
        return torch.floor(vectors // self.epsilons).to(dtype=torch.int64)
    
    # def get_closest_bin_indices(self, vectors: torch.Tensor, obj_ids: torch.Tensor | None = None) -> torch.Tensor:
    #     pass
    
    # def on_rebuild(self, trigger_bin_id: tuple):
    #     start_r_time = time.time()
    #     selection = self.get_vectors(self.bins[trigger_bin_id]) # get all vectors in the bin
    #     # balances = (selection - medians).sum(dim=0)
    #     # balances.abs_()
    #     # sort_ids = torch.argsort(balances)
    #     approx_num_dims = math.floor(math.log2(selection.shape[0] / self.max_bin_size)) + 1
    #     # approx_num_dims = max(1, approx_num_dims)
    #     selected_dims = torch.randint(selection.shape[-1], (approx_num_dims,), device=self.epsilons.device, dtype=torch.int64)
    #     medians = selection[:,selected_dims].median(dim = 0).values

    #     # selected_dims = sort_ids[:approx_num_dims] # take only first approx_num_dims dimensions
    #     bin_id_tensor = torch.tensor([trigger_bin_id[i] for i in selected_dims], dtype=self.epsilons.dtype, device=self.epsilons.device)
    #     bin_start = self.epsilons[selected_dims] * bin_id_tensor
    #     self.epsilons[selected_dims] = medians - bin_start
    #     # zero = torch.tensor(0, dtype=new_epsilons.dtype, device=new_epsilons.device)
    #     # self.epsilons = torch.where(torch.isclose(new_epsilons, zero, atol=self.atol, rtol=self.rtol),
    #     #                            self.epsilons, new_epsilons)
    #     end_r_time = time.time()    
    #     print("\tRebuild: ", end_r_time - start_r_time)
    #     pass

    def on_rebuild(self, trigger_bin_id: tuple):
        # start_r_time = time.time()
        selection = self.get_vectors(self.bins[trigger_bin_id]) # get all vectors in the bin
        medians = selection.median(dim = 0).values
        balances = (selection - medians).sum(dim=0)
        balances.abs_()
        sort_ids = torch.argsort(balances)
        approx_num_dims = math.floor(math.log2(selection.shape[0] / self.max_bin_size)) + 1
        selected_dims = sort_ids[:approx_num_dims] # take only first approx_num_dims dimensions
        bin_id_tensor = torch.tensor([trigger_bin_id[i] for i in selected_dims], dtype=self.epsilons.dtype, device=self.epsilons.device)
        bin_start = self.epsilons[selected_dims] * bin_id_tensor
        self.epsilons[selected_dims] = medians[selected_dims] - bin_start
        # zero = torch.tensor(0, dtype=new_epsilons.dtype, device=new_epsilons.device)
        # self.epsilons = torch.where(torch.isclose(new_epsilons, zero, atol=self.atol, rtol=self.rtol),
        #                            self.epsilons, new_epsilons)
        # end_r_time = time.time()    
        # print("\tRebuild: ", end_r_time - start_r_time)        
        pass    