"""
Unified model provider interface using LiteLLM.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import litellm
from litellm import acompletion, model_cost
from ant_coding.core.config import ModelConfig, get_env

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Exception raised for model-related errors."""
    pass

class ModelProvider:
    """
    Provider for LLM completions using LiteLLM.
    Handles unified interface, retries, and token/cost tracking.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = get_env(config.api_key_env)
        
        # Usage tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        
    async def complete(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified completion interface.
        
        Args:
            messages: List of message dictionaries.
            max_tokens: Override config max_tokens.
            temperature: Override config temperature.
            **kwargs: Additional LiteLLM arguments.
            
        Returns:
            The raw response from LiteLLM.
            
        Raises:
            ModelError: If the call fails after retries.
        """
        params = {
            "model": self.config.litellm_model,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "api_key": self.api_key,
            **kwargs
        }
        
        retries = 3
        delay = 1
        
        for attempt in range(retries):
            try:
                response = await acompletion(**params)
                self._update_usage(response)
                return response
            except Exception as e:
                # LiteLLM raises various exceptions for transient errors
                # (RateLimitError, ServiceUnavailableError, etc.)
                if attempt < retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {retries} attempts failed for model {self.config.name}")
                    raise ModelError(f"Model call failed: {str(e)}") from e
                    
    def _update_usage(self, response: Any):
        """Update token and cost usage from response."""
        try:
            usage = response.usage
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            
            # Calculate cost
            # cost = model_cost.get(self.config.litellm_model, {}).get("cost_per_token", 0) * usage.total_tokens
            # LiteLLM provides completion_cost helper
            try:
                cost = litellm.completion_cost(completion_response=response)
                if cost:
                    self.total_cost += float(cost)
            except:
                # Fallback or ignore if cost calculation fails
                pass
        except AttributeError:
            logger.warning("Response missing usage information")

    def get_usage(self) -> Dict[str, Any]:
        """Return current usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "total_cost_usd": self.total_cost
        }

    def reset_usage(self):
        """Reset usage statistics to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
