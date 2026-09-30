"""Unused reference adapter. Not registered or called by the application.

Settings must be explicitly injected; this module loads no environment config.
"""
import logging
import time
from openai import OpenAI, APITimeoutError
import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


# Adapter: translate our TextGenerator interface to the OpenAI-compatible SDK.
class LLMService:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.client = client if client is not None else OpenAI(
            base_url=self.settings.MODEL_BASE_URL,
            api_key=self.settings.MODEL_API_KEY,
            timeout=self.settings.MODEL_TIMEOUT_SECONDS,
            max_retries=self.settings.MODEL_MAX_RETRIES,
        )

    def stream(self, prompt: str, enable_thinking: bool | None = None,
               max_tokens: int | None = None):
        started = time.perf_counter()
        # Cover both initial request and mid-stream idle timeouts, preserving partial output.
        try:
            yield from self._stream(prompt, enable_thinking, max_tokens)
        except (APITimeoutError, httpx.TimeoutException) as exc:
            # Log only the exception class, never exception text or request headers.
            cause = exc.__cause__ if exc.__cause__ is not None else exc
            logger.warning("LLM request timed out after %.2f seconds (transport=%s)",
                           time.perf_counter() - started, type(cause).__name__)
            raise HTTPException(504, "LLM did not respond within the configured network timeout after any configured retries. Try again.") from None

    def _stream(self, prompt: str, enable_thinking: bool | None = None,
               max_tokens: int | None = None):
        if not prompt.strip():
            return
        start = time.perf_counter()
        logger.info("LLM generation request starting (model=%s; thinking=%s)",
                    self.settings.MODEL_NAME,
                    self.settings.MODEL_ENABLE_THINKING if enable_thinking is None else enable_thinking)
        response = self.client.chat.completions.create(
            model=self.settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.MODEL_TEMPERATURE,
            top_p=self.settings.MODEL_TOP_P,
            max_tokens=(self.settings.MODEL_MAX_TOKENS if max_tokens is None
                        else max(1, min(max_tokens, self.settings.MODEL_MAX_TOKENS))),
            extra_body={"chat_template_kwargs": {
                "enable_thinking": (self.settings.MODEL_ENABLE_THINKING
                    if enable_thinking is None else enable_thinking),
            }},
            stream=True,
        )
        logger.info("LLM response headers received in %.2f seconds", time.perf_counter() - start)
        first_token = None
        finish_reason = None
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                if choice.delta.content:
                    if first_token is None:
                        first_token = time.perf_counter() - start
                        logger.info("LLM first response token in %.2f seconds", first_token)
                    yield choice.delta.content
            if finish_reason == "length":
                yield "\n\n[Response reached its length limit. Ask a narrower follow-up or request the remaining details in another message.]"
        finally:
            response.close()
            logger.info("LLM request completed in %.2f seconds (first token: %s; finish: %s)",
                        time.perf_counter() - start,
                        f"{first_token:.2f}s" if first_token is not None else "none", finish_reason)

    def generate(self, prompt: str):
        return "".join(self.stream(prompt)).strip() or None
