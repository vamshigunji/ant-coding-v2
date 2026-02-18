import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from ant_coding.core.config import ModelConfig
from ant_coding.models.provider import ModelProvider, ModelError

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
