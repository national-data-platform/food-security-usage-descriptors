"""
Unit tests for LLMService (Ollama/LM Studio abstraction).
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from pub_analysis_agent.services.llm_service import LLMService, LLMModelConfig, PromptTemplate

@pytest.fixture
def model_config():
    return LLMModelConfig(
        name="qwen2:72b",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=128,
        timeout=5.0,
        headers={"Authorization": "Bearer test"}
    )

@pytest.fixture
def llm_service(model_config):
    service = LLMService(model_config)
    yield service
    asyncio.get_event_loop().run_until_complete(service.close())

@pytest.mark.asyncio
async def test_generate_success(llm_service):
    mock_response = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
    }
    mock_post = AsyncMock()
    mock_post.return_value.json = lambda: mock_response
    mock_post.return_value.raise_for_status.return_value = None
    with patch.object(llm_service._client, "post", new=mock_post):
        result = await llm_service.generate("Say hi!")
        assert result["choices"][0]["message"]["content"] == "Hello!"

@pytest.mark.asyncio
async def test_generate_http_error(llm_service):
    with patch.object(llm_service._client, "post", new=AsyncMock()) as mock_post:
        mock_post.side_effect = Exception("Connection error")
        with pytest.raises(Exception):
            await llm_service.generate("fail test")

@pytest.mark.asyncio
async def test_generate_batch(llm_service):
    responses = [
        {
            "id": "1",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "A"}, "finish_reason": "stop"}
            ],
        },
        {
            "id": "2",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "B"}, "finish_reason": "stop"}
            ],
        },
    ]
    async def fake_generate(prompt, **kwargs):
        return responses.pop(0)
    llm_service.generate = AsyncMock(side_effect=fake_generate)
    prompts = ["Prompt 1", "Prompt 2"]
    results = await llm_service.generate_batch(prompts, concurrency=2)
    assert len(results) == 2
    assert results[0]["choices"][0]["message"]["content"] == "A"
    assert results[1]["choices"][0]["message"]["content"] == "B"


def test_prompt_template_render():
    template = PromptTemplate(
        name="greet",
        template="Hello, {name}!",
        variables=["name"]
    )
    rendered = template.render(name="Alice")
    assert rendered == "Hello, Alice!"
    with pytest.raises(ValueError):
        template.render() 