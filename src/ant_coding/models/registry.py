"""
Registry for managing and instantiating model providers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from ant_coding.core.config import ModelConfig, load_model_config
from ant_coding.models.provider import ModelProvider

class ModelNotFoundError(Exception):
    """Exception raised when a model is not found in the registry."""
    pass

class ModelRegistry:
    """
    Registry for model configurations.
    Loads configs from YAML files and creates ModelProvider instances.
    """
    
    def __init__(self):
        self._configs: Dict[str, ModelConfig] = {}
        
    def load_from_yaml(self, directory: Union[str, Path]):
        """
        Load all YAML model configs from a directory.
        
        Args:
            directory: Path to the directory containing model YAML files.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return
            
        for yaml_file in dir_path.glob("*.yaml"):
            try:
                config = load_model_config(yaml_file)
                self.register(config)
            except Exception:
                # Skip invalid configs during bulk load
                continue
                
    def register(self, config: ModelConfig):
        """Register a model configuration."""
        self._configs[config.name] = config
        
    def get(self, name: str, token_budget: Optional[int] = None) -> ModelProvider:
        """
        Get a new ModelProvider instance for the given model name.
        
        Args:
            name: The name of the model.
            token_budget: Optional token budget for the provider.
            
        Returns:
            A fresh ModelProvider instance.
            
        Raises:
            ModelNotFoundError: If the model name is not registered.
        """
        if name not in self._configs:
            raise ModelNotFoundError(f"Model not found in registry: {name}")
            
        return ModelProvider(self._configs[name], token_budget=token_budget)
        
    def list_available(self) -> List[str]:
        """Return a list of available model names."""
        return sorted(list(self._configs.keys()))
