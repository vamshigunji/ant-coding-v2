import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from ant_coding.core.config import ModelConfig
from ant_coding.models.provider import ModelProvider, ModelError, TokenBudgetExceeded
from ant_coding.models.registry import ModelRegistry

@pytest.fixture
def mock_model_config():
    return ModelConfig(
        name="test-model",
        litellm_model="anthropic/claude-test",
        api_key_env="TEST_API_KEY",
        max_tokens=1000,
        temperature=0.0
    )

@pytest.mark.asyncio
async def test_model_provider_complete_success(mock_model_config, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "fake-key")
    
    # Mock LiteLLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    
    with patch("ant_coding.models.provider.acompletion", new_callable=AsyncMock) as mock_acompletion, \
         patch("litellm.completion_cost", return_value=0.01):
        mock_acompletion.return_value = mock_response
        
        provider = ModelProvider(mock_model_config)
        response = await provider.complete(messages=[{"role": "user", "content": "hello"}])
        
        assert response.choices[0].message.content == "Test response"
        usage = provider.get_usage()
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_cost_usd"] == 0.01

@pytest.mark.asyncio
async def test_model_provider_retry_logic(mock_model_config, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "fake-key")
    
    with patch("ant_coding.models.provider.acompletion", new_callable=AsyncMock) as mock_acompletion:
        # Fail twice, then succeed
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        
        mock_acompletion.side_effect = [
            Exception("Transient error"),
            Exception("Transient error"),
            mock_response
        ]
        
        provider = ModelProvider(mock_model_config)
        # Use small sleep to speed up test
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await provider.complete(messages=[])
            
        assert mock_acompletion.call_count == 3

@pytest.mark.asyncio
async def test_model_provider_all_retries_fail(mock_model_config, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "fake-key")
    
    with patch("ant_coding.models.provider.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = Exception("Permanent failure")
        
        provider = ModelProvider(mock_model_config)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ModelError) as excinfo:
                await provider.complete(messages=[])
        
        assert "Permanent failure" in str(excinfo.value)
        assert mock_acompletion.call_count == 3

def test_model_provider_reset_usage(mock_model_config, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "fake-key")
    provider = ModelProvider(mock_model_config)
    provider.prompt_tokens = 100
    provider.completion_tokens = 50
    provider.total_cost = 1.5
    
    provider.reset_usage()
    usage = provider.get_usage()
    assert usage["total_tokens"] == 0
    assert usage["total_cost_usd"] == 0.0

@pytest.mark.asyncio
async def test_model_provider_token_budget_enforcement(mock_model_config, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "fake-key")
    
    # Budget of 50 tokens
    provider = ModelProvider(mock_model_config, token_budget=50)
    
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 30
    mock_response.usage.completion_tokens = 30
    mock_response.usage.total_tokens = 60
    
    with patch("ant_coding.models.provider.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        
        # This call should exceed the budget
        with pytest.raises(TokenBudgetExceeded) as excinfo:
            await provider.complete(messages=[])
            
        assert excinfo.value.current_tokens == 60
        assert excinfo.value.budget_limit == 50
        assert excinfo.value.last_call_tokens == 60

@pytest.mark.asyncio
async def test_model_provider_pre_call_budget_check(mock_model_config, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "fake-key")
    provider = ModelProvider(mock_model_config, token_budget=50)
    provider.prompt_tokens = 60 # Already exceeded
    
    with pytest.raises(TokenBudgetExceeded) as excinfo:
        await provider.complete(messages=[])
    assert excinfo.value.current_tokens == 60

def test_model_registry_load_and_get(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    
    registry = ModelRegistry()
    registry.load_from_yaml("configs/models/")
    
    available = registry.list_available()
    assert "claude-sonnet" in available
    assert "gpt-4o" in available
    assert "gemini-flash" in available
    
    provider = registry.get("claude-sonnet")
    assert isinstance(provider, ModelProvider)
    assert provider.config.name == "claude-sonnet"
    
    # Verify fresh instance
    provider2 = registry.get("claude-sonnet")
    assert provider is not provider2

@pytest.mark.asyncio
async def test_model_provider_real_call_gemini(monkeypatch):
    """
    Optional real integration test if GOOGLE_API_KEY is present.
    Ensures that our wrapping of LiteLLM actually works with the provider.
    """
    import os
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "AIzaxxx":
        pytest.skip("GOOGLE_API_KEY not set")

    config = ModelConfig(
        name="gemini-test",
        litellm_model="gemini/gemini-2.0-flash",
        api_key_env="GOOGLE_API_KEY"
    )
    provider = ModelProvider(config)
    
    messages = [{"role": "user", "content": "Say 'hello world' and nothing else."}]
    response = await provider.complete(messages=messages)
    
    content = response.choices[0].message.content
    assert "hello" in content.lower()
    assert provider.get_usage()["total_tokens"] > 0

@pytest.mark.asyncio
async def test_model_provider_malformed_response(mock_model_config, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "fake-key")
    
    with patch("ant_coding.models.provider.acompletion", new_callable=AsyncMock) as mock_acompletion:
        # Return something that doesn't have .usage
        mock_acompletion.return_value = {"not": "a real response object"}
        
        provider = ModelProvider(mock_model_config)
        # Should still return the response but log warning (internal usage update fails gracefully)
        response = await provider.complete(messages=[])
        assert response == {"not": "a real response object"}
        assert provider.get_usage()["total_tokens"] == 0
