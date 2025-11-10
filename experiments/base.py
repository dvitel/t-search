
''' Utils to run GPSolver from command line and config files and store results '''

from typing import TYPE_CHECKING
import torch
import t_search.datasets as datasets
import numpy as np

if TYPE_CHECKING:
    from t_search import GPSolver

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'bfloat16': torch.bfloat16
}

def run(pipeline, *, dataset:str, config: str, output="koza-{}.json", device='cuda', dtype='float16', seed=42):

    dtype = dtype_mapping[dtype]

    import json
    with open(config, 'r') as f:
        kwargs = json.load(f)

    if not hasattr(datasets, dataset):
        raise ValueError(f"Unknown dataset {dataset}")
    
    rnd = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)

    ds: datasets.Benchmark = getattr(datasets, dataset)
    free_vars, target = ds.sample_set("train", device=device, dtype=dtype, generator=torch_gen, sorted=True)

    solver: GPSolver = pipeline(
        rnd=rnd, 
        torch_gen=torch_gen, 
        device=device,
        dtype=dtype, 
        **kwargs
    )
    
    solver.fit(free_vars, target)

    output = output.format(dataset)
    solver.save_metrics(output)

def run_args(pipeline):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='One of datasets from t_search.datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to config json file with specified constraints')
    parser.add_argument('--output', type=str, required=True, help='Output metrics file path pattern, use {} for dataset name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument() 

    args = parser.parse_args()
    
    run(pipeline,
        dataset=args.dataset, 
        config=args.config, 
        output=args.output, 
        device=args.device, 
        dtype=args.dtype, 
        seed=args.seed
    )
