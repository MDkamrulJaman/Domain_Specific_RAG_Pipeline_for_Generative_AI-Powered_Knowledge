import logging
import time
from typing import Optional

from openai import OpenAI

from app.core.config import Settings

# Setup logging
logger = logging.getLogger(__name__)


class LLMService:
    """Service class for an OpenAI-compatible chat-completion endpoint."""

    def __init__(self):
        settings = Settings()

        self.base_url = settings.MODEL_BASE_URL
        self.model_name = settings.MODEL_NAME
        self.timeout = settings.MODEL_TIMEOUT_SECONDS
        self.api_key = settings.MODEL_API_KEY
        self.temperature = settings.MODEL_TEMPERATURE
        self.top_p = settings.MODEL_TOP_P
        self.max_tokens = settings.MODEL_MAX_TOKENS

        try:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            logger.info(
                "LLMService successfully initialized with OpenAI-compatible "
                "provider targeting %s [%s]",
                self.base_url,
                self.model_name,
            )
        except Exception as e:
            logger.critical("Failed to initialize OpenAI client: %s", str(e))
            raise RuntimeError("Could not initialize LLMService client.") from e

    def generate(self, prompt: str) -> Optional[str]:
        """
        Sends a prompt to the OpenAI-compatible endpoint and returns its text.
        """
        if not prompt or not prompt.strip():
            logger.warning("Received an empty or invalid prompt.")
            return None

        try:
            logger.debug("Invoking model '%s'", self.model_name)
            start_time = time.perf_counter()

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature= self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": 16384,
                },
                stream=True,
            )

            content_parts = []
            reasoning_length = 0
            for chunk in response:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_length += len(reasoning)

                if delta.content is not None:
                    content_parts.append(delta.content)

            if reasoning_length:
                logger.debug("Received %d reasoning characters.", reasoning_length)

            content = "".join(content_parts).strip()
            elapsed_time = time.perf_counter() - start_time
            logger.info(
                "Model '%s' completed in %.2f seconds.",
                self.model_name,
                elapsed_time,
            )
            if not content:
                logger.error("The model returned an empty response.")
                return None

            return content

        except ConnectionError as ce:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}. Ensure Ollama is running. Error: {str(ce)}")
            return None
            
        except Exception as e:
            # Catch-all for timeouts, missing local models (e.g. if gemma3 needs to be pulled), or parsing glitches
            logger.error(f"An error occurred during LLM text generation with {self.model_name}: {str(e)}")
            return None
        


