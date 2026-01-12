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
                free_vars, target = ds.sample_set("train", device="cuda", dtype=torch.float32, sorted=True,
                                                    max_dim_size=1000)
                valid_target = torch.isfinite(target).all()
                print(f"  - free_vars_dims: {free_vars.shape}")
                print(f"  - target: {target.shape}. Valid: {valid_target}")
                free_vars_test, target_test = ds.sample_set("test", device="cuda", dtype=torch.float32, sorted=True,
                                                    max_dim_size=1000)
                valid_target_test = torch.isfinite(target_test).all()
                print(f"  - [test] free_vars_dims: {free_vars_test.shape}")
                print(f"  - [test] target: {target_test.shape}. Valid: {valid_target_test}")
        pass    

if __name__ == "__main__":
    main()