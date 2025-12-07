
''' Running the GPSolver from command line/config files and store the results '''

from collections import deque
import inspect
import os
from typing import Any, Callable, Set, Type
import torch
import t_search.datasets as datasets
import numpy as np

import t_search.operators as operators_module
# import t_search.syntax as syntax_module
import t_search.spatial as spatial_module
import t_search.evaluators as evaluators_module
from .solver import GPSolver

# only these modules are used for injection context building
injection_context_modules = [operators_module, spatial_module, evaluators_module]


dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'bfloat16': torch.bfloat16
}

def get_method_params(cls: Type, method_name: str = "__init__") -> dict[str, bool]:
    """Get all __init__ parameter names from cls and its bases."""
    params = {}
    
    # Walk the Method Resolution Order
    for base in inspect.getmro(cls):
        if base is object:  # Skip object
            continue
        
        try:
            sig = inspect.signature(getattr(base, method_name))
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    continue
                has_default = param.default is not inspect.Parameter.empty
                params[param_name] = has_default
        except (ValueError, TypeError):
            # Some built-in types don't have inspectable signatures
            continue
    
    return params

def register_services(service_cfgs: dict, injection_context: dict[str, Any], 
                      service_builder: Callable) -> dict[str, Any]:
    services = {}
    visited_ids = set()
    service_q = deque(list(service_cfgs.items()))
    while len(service_q) > 0:
        service_name, cfg = service_q.popleft()
        fixed_cfg = replace_refs(cfg, injection_context) 
        if fixed_cfg is None: # should delay 
            if service_name in visited_ids:
                raise ValueError(f"Cyclic service reference detected for {service_name}")
            else:
                visited_ids.add(service_name)
                service_q.append((service_name, cfg))
                continue
        visited_ids.add(service_name)
        if type(fixed_cfg) is not dict: # injected service already 
            services[service_name] = fixed_cfg
            injection_context[service_name] = fixed_cfg
            continue
        if "type" not in fixed_cfg:
            raise ValueError(f"Service definition for '{service_name}' is missing required 'type' field.")
        module_name = fixed_cfg.get('module', None)
        if module_name is not None:
            service_module = __import__(module_name, fromlist=[fixed_cfg['type']])
        else:
            service_type = fixed_cfg['type']
            service_module = next((m for m in injection_context_modules if hasattr(m, service_type)), None)
            if service_module is None:
                raise ValueError(f"Unknown service type '{service_type}'")
        service = service_builder(service_name, getattr(service_module, fixed_cfg['type']), fixed_cfg.get('params', {}))
        services[service_name] = service
        injection_context[service_name] = service
    return services

def replace_refs(params: Any, injection_context: dict) -> Any | None:
    ''' Replaces !ref in json with specific named object in the context recursivelly 
        !ref has only one parameter - the name in the context: {"!ref": "name_in_context"}
    '''
    if isinstance(params, dict):
        if "!ref" in params:
            ref_name = params['!ref']
            if ref_name not in injection_context:
                return None # cannot resolve now, should delay
            return injection_context[ref_name]
        new_params = {}
        for k, v in params.items():
            new_v = replace_refs(v, injection_context)
            if new_v is None:
                return None
            new_params[k] = new_v
        return new_params
    if isinstance(params, list):
        new_params = []
        for p in params:
            new_p = replace_refs(p, injection_context)
            if new_p is None:
                return None
            new_params.append(new_p)
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
    service_definitions = {}
    for cfg in reversed(configs):
        if 'service_definitions' in cfg:
            for service_name, service_cfg in cfg['service_definitions'].items():
                if service_name in service_definitions:
                    service_params = service_cfg.pop('params', {})
                    existing_params = service_definitions[service_name].get('params', {})
                    existing_params.update(service_params)
                    service_definitions[service_name]['params'] = existing_params
                    service_definitions[service_name].update(service_cfg)
                else:
                    service_definitions[service_name] = service_cfg
        config.update(cfg)
    config['service_definitions'] = service_definitions
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

    service_definitions = kwargs.get('service_definitions', {})
    init_service_name = kwargs.get('init_service_name', '')
    operator_service_names = kwargs.get('operator_service_names', [])
    evaluator_service_name = kwargs.get('evaluator_service_name', '')
    syntax_service_name = kwargs.get('syntax_service_name', '')

    solver = GPSolver(
        service_definitions=service_definitions,
        init_service_name=init_service_name,
        operator_service_names=operator_service_names,
        evaluator_service_name=evaluator_service_name,
        syntax_service_name=syntax_service_name,
        device=device,
        dtype=dtype,
        rnd = rnd,
        torch_gen = torch_gen,
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