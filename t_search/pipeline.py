
''' Running the GPSolver from command line/config files and store the results '''

import os
from typing import Any
import torch
import t_search.datasets as datasets
import numpy as np

import t_search.operators as operators_module
# import t_search.syntax as syntax_module
import t_search.spatial as spatial_module
from .solver import GPSolver

# only these modules are used for injection context building
injection_context_modules = [operators_module, spatial_module]


dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'bfloat16': torch.bfloat16
}

def register_services(service_cfgs: list[dict], injection_context: dict[str, Any]) -> list[Any]:
    services = []
    for cfg in service_cfgs:
        service_type = cfg.get('type', 'MISSING_TYPE')
        service_params = cfg.get('params', {})
        service_params = replace_refs(service_params, injection_context)
        service_module = next((m for m in injection_context_modules if hasattr(m, service_type)), None)
        if service_module is None:
            raise ValueError(f"Unknown service type '{service_type}'")
        service = getattr(service_module, service_type)(**service_params)
        context_name = cfg.get('name', None) or getattr(service, 'name', None)
        if context_name is not None:
            if context_name in injection_context:
                raise ValueError(f"Service with name '{context_name}' is already registered in the context. Check config.")
            injection_context[context_name] = service   
        services.append(service) 
    return services

def replace_refs(params: Any, injection_context: dict) -> Any:
    ''' Replaces !ref in json with specific named object in the context recursivelly 
        !ref has only one parameter - the name in the context: {"!ref": "name_in_context"}
    '''
    if isinstance(params, dict):
        if "!ref" in params:
            ref_name = params['!ref']
            if ref_name not in injection_context:
                raise ValueError(f"Reference '{ref_name}' not found in injection context")
            return injection_context[ref_name]
        new_params = {k: replace_refs(v, injection_context) for k, v in params.items() }
        return new_params
    if isinstance(params, list):
        new_params = [replace_refs(p, injection_context) for p in params ]
        return new_params
    return params

def read_config(config_path: str) -> dict:
    ''' Recursivelly reads configs. 
        Checks for !base on the root level and reads parent first and then overrides with child.
    '''
    import json
    visited = set()
    configs = []
    config_full_path = os.path.abspath(config_path)
    while True:
        if config_full_path in visited:
            raise ValueError(f"Cyclic config reference detected for {config_full_path}")
        with open(config_full_path, 'r') as f:
            child_config = json.load(f)
        visited.add(config_full_path)
        configs.append(child_config)
        if "!base" in child_config:
            base_path = child_config.pop("!base")
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(config_full_path), base_path)
            config_full_path = os.path.abspath(base_path)
        else:
            break            
    config = {}
    for cfg in reversed(configs):
        config.update(cfg)
    return config

def config_pipeline(*, dataset:str, config: str, output="koza-{}.json", device='cuda', 
                    dtype='float16', seed=42, debug: bool = False):

    dtype = dtype_mapping[dtype]

    kwargs = read_config(config)

    if not hasattr(datasets, dataset):
        raise ValueError(f"Unknown dataset {dataset}")
    
    rnd = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)

    ds: datasets.Benchmark = getattr(datasets, dataset)
    free_vars, target = ds.sample_set("train", device=device, dtype=dtype, generator=torch_gen, sorted=True)

    if 'pipeline' not in kwargs:
        raise ValueError("Config is missing required GP pipeline. Check _full.json for example.")
    pipeline: dict = kwargs.pop('pipeline')
    if 'init' not in pipeline or 'operators' not in pipeline or not isinstance(pipeline['operators'], list):
        raise ValueError("Pipeline config must contain 'init' and 'operators' fields. Check _full.json for example.")
    
    injection_context = {}

    if "context" in pipeline and isinstance(pipeline['context'], list):
        register_services(pipeline['context'], injection_context)

    listeners = []
    if "listeners" in pipeline and isinstance(pipeline['listeners'], list):
        listeners = register_services(pipeline['listeners'], injection_context)

    init, = register_services([pipeline['init']], injection_context)

    operators = register_services(pipeline['operators'], injection_context)

    solver = GPSolver(
        init=init,
        operators=operators,
        listeners=listeners,
        rnd=rnd,
        torch_gen=torch_gen,
        device=device,
        dtype=dtype,
        debug=debug,
        **kwargs
    )

    solver.fit(free_vars, target)

    output = output.format(dataset)
    solver.save_metrics(output)

def args_pipeline():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='One of datasets from t_search.datasets')
    parser.add_argument('--config', type=str, required=True, help='Path to config json file with specified constraints')
    parser.add_argument('--output', type=str, required=True, help='Output metrics file path pattern, use {} for dataset name')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='float16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    # parser.add_argument() 

    args = parser.parse_args()
    
    config_pipeline(
        dataset=args.dataset, 
        config=args.config, 
        output=args.output, 
        device=args.device, 
        dtype=args.dtype, 
        seed=args.seed,
        debug=args.debug
    )