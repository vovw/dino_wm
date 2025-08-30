"""
Utility functions to replace Hydra functionality.
"""
import importlib
from typing import Any, Dict


def instantiate(config: Any, **kwargs) -> Any:
    """
    Instantiate an object from a config.
    Replaces hydra.utils.instantiate functionality.
    
    Args:
        config: Configuration object with _target_ attribute
        **kwargs: Additional keyword arguments
        
    Returns:
        Instantiated object
    """
    if hasattr(config, '_target_'):
        target = config._target_
    elif isinstance(config, dict) and '_target_' in config:
        target = config['_target_']
    else:
        raise ValueError("Config must have _target_ attribute")
    
    # Parse the target string
    module_name, class_name = target.rsplit('.', 1)
    
    # Import the module and get the class
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    
    # Get config parameters
    if hasattr(config, '__dict__'):
        params = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    elif isinstance(config, dict):
        params = {k: v for k, v in config.items() if not k.startswith('_')}
    else:
        params = {}
    
    # Merge with additional kwargs
    params.update(kwargs)
    
    # Instantiate and return
    return cls(**params)


def call(config: Any, **kwargs) -> Any:
    """
    Call a function from a config.
    Replaces hydra.utils.call functionality.
    
    Args:
        config: Configuration object with _target_ attribute
        **kwargs: Additional keyword arguments
        
    Returns:
        Result of function call
    """
    if hasattr(config, '_target_'):
        target = config._target_
    elif isinstance(config, dict) and '_target_' in config:
        target = config['_target_']
    else:
        raise ValueError("Config must have _target_ attribute")
    
    # Parse the target string
    module_name, func_name = target.rsplit('.', 1)
    
    # Import the module and get the function
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    
    # Get config parameters
    if hasattr(config, '__dict__'):
        params = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    elif isinstance(config, dict):
        params = {k: v for k, v in config.items() if not k.startswith('_')}
    else:
        params = {}
    
    # Merge with additional kwargs
    params.update(kwargs)
    
    # Call and return
    return func(**params)


def config_to_dict(config: Any) -> Dict[str, Any]:
    """
    Convert a config object to a dictionary.
    Replaces OmegaConf.to_container functionality.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary representation of config
    """
    if hasattr(config, '__dict__'):
        result = {}
        for k, v in config.__dict__.items():
            if hasattr(v, '__dict__'):
                result[k] = config_to_dict(v)
            elif isinstance(v, list):
                result[k] = [config_to_dict(item) if hasattr(item, '__dict__') else item for item in v]
            else:
                result[k] = v
        return result
    elif isinstance(config, dict):
        return {k: config_to_dict(v) if hasattr(v, '__dict__') else v for k, v in config.items()}
    else:
        return config
