import argparse
import t_search.datasets as datasets

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--list-ds', action='store_true', help='Output datasets and exit')

def main():
    args = parser.parse_args()
    if args.list_ds:
        # iterate all objects in this module 
        for name in dir(datasets):
            ds = getattr(datasets, name)
            if isinstance(ds, datasets.Benchmark):
                print(f"Benchmark: {ds.name}")
                free_vars, target = ds.sample_set("train", device="cpu", dtype=torch.float32, sorted=True,
                                                    max_dim_size=1000)
                print(f"  - free_vars_dims: {free_vars.shape}")
                print(f"  - target: {target.shape}")
        pass    

if __name__ == "__main__":
    main()