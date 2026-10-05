"""Provider factory for constructing LangChain chat models.

Decouples the lip-reading application from specific LLM vendors (OpenAI,
Anthropic, Ollama, Google, or local Mock) and instantiates configured
BaseChatModel instances via LangChain.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.genai.config import GenAIConfig

logger = logging.getLogger("lipread.genai.providers")


# ---------------------------------------------------------------------------
# Deterministic Mock Provider (Zero-Cost / Offline / Testing)
# ---------------------------------------------------------------------------


class MockChatModel(BaseChatModel):
    """Deterministic offline mock model for testing and demonstration.

    Applies rule-based viseme corrections (e.g. 'lay freen' -> 'lay green')
    without requiring external network calls, API keys, or paid services.
    """

    custom_response: Optional[str] = None
    provider_name: str = "mock"

    @property
    def _llm_type(self) -> str:
        return "lipread-mock-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.custom_response is not None:
            text = self.custom_response
        else:
            # Extract the raw transcript from the user prompt
            user_content = ""
            for msg in reversed(messages):
                if msg.type in ("human", "user"):
                    user_content = str(msg.content)
                    break

            # Extract content between quotes if present
            raw = user_content
            if '"' in user_content:
                parts = user_content.split('"')
                if len(parts) >= 2:
                    raw = parts[1]

            # Apply standard sample corrections for known GRID viseme ambiguities
            corrected = raw
            replacements = [
                ("lay freen", "lay green"),
                ("pin blue", "bin blue"),
                ("pleaze", "please"),
                ("bin red in", "bin red at"),
            ]
            for src, tgt in replacements:
                if src in corrected:
                    corrected = corrected.replace(src, tgt)

            text = corrected

        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------


def create_chat_model(config: GenAIConfig) -> BaseChatModel:
    """Instantiates a LangChain BaseChatModel based on the provided configuration.

    Args:
        config: The resolved GenAIConfig.

    Returns:
        Configured BaseChatModel instance.

    Raises:
        ValueError: If provider is unrecognized or required credentials are missing.
    """
    provider = config.provider.lower().strip()

    if provider in ("mock", "fake", "test", "none"):
        logger.info("Initializing offline MockChatModel for GenAI post-processing.")
        return MockChatModel()

    if provider == "openai":
        if not config.api_key:
            raise ValueError(
                "OpenAI provider selected but no API key was provided. "
                "Set LIPREAD_LLM_API_KEY or OPENAI_API_KEY environment variable."
            )
        try:
            # Try langchain_openai first, fall back to langchain_community
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                from langchain_community.chat_models import ChatOpenAI

            kwargs: dict[str, Any] = {
                "model_name": config.model,
                "openai_api_key": config.api_key,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
            }
            if config.api_base:
                kwargs["openai_api_base"] = config.api_base

            return ChatOpenAI(**kwargs)
        except Exception as exc:
            raise ValueError(f"Failed to initialize OpenAI chat model: {exc}") from exc

    if provider == "anthropic":
        if not config.api_key:
            raise ValueError("Anthropic provider selected but no API key was found.")
        try:
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                from langchain_community.chat_models import ChatAnthropic

            return ChatAnthropic(
                model=config.model,
                anthropic_api_key=config.api_key,
                temperature=config.temperature,
                max_tokens_to_sample=config.max_tokens,
                default_request_timeout=config.timeout,
            )
        except Exception as exc:
            raise ValueError(f"Failed to initialize Anthropic chat model: {exc}") from exc

    if provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama

            base_url = config.api_base or "http://localhost:11434"
            return ChatOllama(
                model=config.model,
                base_url=base_url,
                temperature=config.temperature,
            )
        except Exception as exc:
            raise ValueError(f"Failed to initialize Ollama chat model: {exc}") from exc

    raise ValueError(
        f"Unsupported LLM provider '{provider}'. "
        "Supported providers: 'mock', 'openai', 'anthropic', 'ollama'."
    )
