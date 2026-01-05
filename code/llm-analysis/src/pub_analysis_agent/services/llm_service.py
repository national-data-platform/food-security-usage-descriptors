"""
LLM Service Abstraction Layer for Ollama/LM Studio.

This module provides an asynchronous abstraction for communicating with local LLM servers
such as Ollama and LM Studio, supporting configurable models, prompt templates, logging,
retry mechanisms, and batch processing.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
from http import HTTPStatus

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LLMModelConfig(BaseModel):
    """Configuration for an LLM model."""
    name: str = Field(..., description="Model name (e.g., 'gpt-oss:120b')")
    base_url: str = Field(..., description="Base URL for the LLM API endpoint")
    temperature: float = Field(0.7, description="Sampling temperature")
    max_tokens: int = Field(4000, description="Maximum tokens in response")
    top_p: Optional[float] = Field(None, description="Top-p sampling parameter")
    timeout: float = Field(500.0, description="Request timeout in seconds")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict, description="Custom HTTP headers")


class PromptTemplate(BaseModel):
    """Prompt template with versioning and rendering support."""
    name: str
    version: str = "1.0"
    template: str
    description: Optional[str] = None
    variables: List[str] = Field(default_factory=list)

    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable for prompt: {e}")


class LLMService:
    """Abstraction layer for local LLM communication (Ollama/LM Studio)."""

    def __init__(
        self,
        model_config: LLMModelConfig,
        prompt_templates: Optional[List[PromptTemplate]] = None,
        max_retries: int = 3,
        rate_limit: Optional[int] = None,
    ) -> None:
        """
        Initialize the LLMService.

        Args:
            model_config: LLM model configuration
            prompt_templates: List of available prompt templates
            max_retries: Maximum number of retries for failed requests
            rate_limit: Maximum concurrent requests (None for unlimited)
        """
        self.model_config = model_config
        self.prompt_templates = {t.name: t for t in (prompt_templates or [])}
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self._semaphore = asyncio.Semaphore(rate_limit) if rate_limit else None
        self._client = httpx.AsyncClient(timeout=model_config.timeout)

    async def close(self) -> None:
        """Close the HTTP client session."""
        await self._client.aclose()

    def add_prompt_template(self, template: PromptTemplate) -> None:
        """Add or update a prompt template."""
        self.prompt_templates[template.name] = template

    def get_prompt_template(self, name: str) -> PromptTemplate:
        """Retrieve a prompt template by name."""
        if name not in self.prompt_templates:
            raise ValueError(f"Prompt template '{name}' not found.")
        return self.prompt_templates[name]

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        log: bool = True,
    ) -> Dict[str, Any]:
        """
        Send a prompt to the LLM and return the response.

        Args:
            prompt: The prompt string to send
            model: Optional model name override
            parameters: Additional generation parameters (temperature, max_tokens, etc)
            log: Whether to log the request/response
        Returns:
            Parsed LLM response as dict
        Raises:
            httpx.HTTPError: If the HTTP request fails
            ValueError: If the response is invalid
        """
        # TODO: verify if lmstudio has /v1 prefix
        url = self.model_config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model or self.model_config.name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
        }
        if self.model_config.top_p is not None:
            payload["top_p"] = self.model_config.top_p
        if parameters:
            payload.update(parameters)
        headers = self.model_config.headers or {}
        try:
            if log:
                logger.info(f"[LLMService] Sending prompt to {url}: {payload}")
            response = await self._client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if log:
                logger.info(f"[LLMService] Response: {data}")
            return data
        except httpx.HTTPError as e:
            logger.error(f"[LLMService] HTTP error: {e}")
            logger.error(f"[LLMService] Response: {response.text}")
            raise
        except Exception as e:
            logger.error(f"[LLMService] Unexpected error: {e}")
            logger.error(f"[LLMService] Response: {response.text}")
            raise ValueError(f"Failed to get LLM response: {e}")

    async def generate_response(
        self,
        prompt_template: PromptTemplate,
        variables: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        log: bool = True,
    ) -> str:
        """
        Generate response using a prompt template with variables.

        Args:
            prompt_template: Prompt template to use
            variables: Variables to substitute in the template
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            log: Whether to log the request/response
        Returns:
            Generated text response
        """
        # Render the prompt template
        prompt = prompt_template.render(**variables)
        
        # Prepare parameters
        parameters = {}
        if temperature is not None:
            parameters["temperature"] = temperature
        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens
        
        # Generate response
        response_data = await self.generate(prompt, parameters=parameters, log=log)

        # Extract text from response (chat/completions format)
        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            if content:
                return content
        raise ValueError("Invalid response format from LLM")

    async def generate_batch(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        concurrency: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Send a batch of prompts to the LLM with concurrency control.

        Args:
            prompts: List of prompt strings
            model: Optional model name override
            parameters: Additional generation parameters
            concurrency: Max concurrent requests
        Returns:
            List of parsed LLM responses
        Raises:
            httpx.HTTPError, ValueError
        """
        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

        async def worker(idx: int, prompt: str) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.generate(
                        prompt, model=model, parameters=parameters, log=True
                    )
                except Exception as e:
                    logger.error(f"[LLMService] Batch error for prompt {idx}: {e}")
                    results[idx] = {"error": str(e)}

        tasks = [worker(i, p) for i, p in enumerate(prompts)]
        await asyncio.gather(*tasks)
        return results

    # Helper methods for logging, retry, validation, etc. will be implemented in the next steps. 