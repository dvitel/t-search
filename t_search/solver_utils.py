
''' Running the GPSolver from command line/config files and store the results '''

from collections import deque
import inspect
import os
from typing import Any, Callable, Type

from pyparsing import Optional

import t_search.operators as operators_module
import t_search.syntax.syntax as syntax_module
import t_search.spatial as spatial_module
import t_search.evaluators as evaluators_module

# only these modules are used for injection context building
injection_context_modules = [operators_module, spatial_module, syntax_module, evaluators_module ]

def get_method_params(cls: Type, method_name: str = "__init__") -> dict[str, tuple[bool, Any]]:
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
                params[param_name] = (has_default, param.default)
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