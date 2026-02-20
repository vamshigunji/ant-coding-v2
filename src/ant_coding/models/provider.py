"""
Unified model provider interface using LiteLLM.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, List, Dict, Any, Optional

import litellm
from litellm import acompletion

from ant_coding.core.config import ModelConfig, get_env

if TYPE_CHECKING:
    from ant_coding.observability.event_logger import EventLogger

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Exception raised for model-related errors."""
    pass


class TokenBudgetExceeded(ModelError):
    """Exception raised when the token budget is exceeded."""
    def __init__(self, current_tokens: int, budget_limit: int, last_call_tokens: int = 0):
        self.current_tokens = current_tokens
        self.budget_limit = budget_limit
        self.last_call_tokens = last_call_tokens
        super().__init__(
            f"Token budget exceeded: {current_tokens} > {budget_limit} "
            f"(last call used {last_call_tokens} tokens)"
        )


class ModelProvider:
    """
    Provider for LLM completions using LiteLLM.
    Handles unified interface, retries, and token/cost tracking.
    """

    def __init__(
        self,
        config: ModelConfig,
        token_budget: Optional[int] = None,
        event_logger: Optional["EventLogger"] = None,
        experiment_id: str = "",
        task_id: str = "",
    ):
        self.config = config
        self.api_key = get_env(config.api_key_env)
        self.token_budget = token_budget
        self._event_logger = event_logger
        self._experiment_id = experiment_id
        self._task_id = task_id

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
            TokenBudgetExceeded: If the budget is already exceeded or hit.
            ModelError: If the call fails after retries.
        """
        if self.token_budget is not None and self.get_total_tokens() >= self.token_budget:
            raise TokenBudgetExceeded(self.get_total_tokens(), self.token_budget)

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
                call_start = time.time()
                response = await acompletion(**params)
                call_duration_ms = (time.time() - call_start) * 1000
                self._update_usage(response)
                self._log_llm_call(response, call_duration_ms, kwargs.get("agent_id"))
                return response
            except TokenBudgetExceeded:
                # Don't retry budget exceeded
                raise
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
            try:
                cost = litellm.completion_cost(completion_response=response)
                if cost:
                    self.total_cost += float(cost)
            except Exception:
                pass

            # Budget enforcement after update
            if self.token_budget is not None and self.get_total_tokens() > self.token_budget:
                raise TokenBudgetExceeded(
                    self.get_total_tokens(), 
                    self.token_budget, 
                    usage.total_tokens
                )
        except AttributeError:
            logger.warning("Response missing usage information")

    def get_total_tokens(self) -> int:
        """Return cumulative token count."""
        return self.prompt_tokens + self.completion_tokens

    def get_usage(self) -> Dict[str, Any]:
        """Return current usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.get_total_tokens(),
            "total_cost_usd": self.total_cost
        }

    def reset_usage(self):
        """Reset usage statistics to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

    def set_context(self, task_id: str, experiment_id: str) -> None:
        """Set the current task/experiment context for event logging."""
        self._task_id = task_id
        self._experiment_id = experiment_id

    def _log_llm_call(
        self, response: Any, duration_ms: float, agent_id: Optional[str] = None
    ) -> None:
        """Log an LLM_CALL event if event_logger is configured."""
        if self._event_logger is None:
            return

        from ant_coding.observability.event_logger import Event, EventType

        try:
            usage = response.usage
            payload = {
                "model": self.config.litellm_model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "duration_ms": round(duration_ms, 1),
            }
            # Add cost if available
            try:
                cost = litellm.completion_cost(completion_response=response)
                if cost:
                    payload["cost_usd"] = float(cost)
            except Exception:
                pass
        except AttributeError:
            payload = {"model": self.config.litellm_model, "duration_ms": round(duration_ms, 1)}

        self._event_logger.log(Event(
            type=EventType.LLM_CALL,
            task_id=self._task_id,
            experiment_id=self._experiment_id,
            agent_id=agent_id,
            payload=payload,
        ))
