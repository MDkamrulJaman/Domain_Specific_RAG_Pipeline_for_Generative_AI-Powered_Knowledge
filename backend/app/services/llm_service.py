import logging
import time
from openai import OpenAI
from app.core.config import Settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.settings = Settings()
        self.client = OpenAI(
            base_url=self.settings.MODEL_BASE_URL,
            api_key=self.settings.MODEL_API_KEY,
            timeout=self.settings.MODEL_TIMEOUT_SECONDS,
        )

    def stream(self, prompt: str, enable_thinking: bool | None = None):
        if not prompt.strip():
            return
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.MODEL_TEMPERATURE,
            top_p=self.settings.MODEL_TOP_P,
            max_tokens=self.settings.MODEL_MAX_TOKENS,
            extra_body={"chat_template_kwargs": {
                "enable_thinking": (self.settings.MODEL_ENABLE_THINKING
                    if enable_thinking is None else enable_thinking),
            }},
            stream=True,
        )
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
                        logger.info("NVIDIA first response token in %.2f seconds", first_token)
                    yield choice.delta.content
            if finish_reason == "length":
                yield "\n\n[Response reached its length limit. Ask a narrower follow-up, or increase MODEL_MAX_TOKENS for longer answers.]"
        finally:
            response.close()
            logger.info("LLM request completed in %.2f seconds (first token: %s; finish: %s)",
                        time.perf_counter() - start,
                        f"{first_token:.2f}s" if first_token is not None else "none", finish_reason)

    def generate(self, prompt: str):
        return "".join(self.stream(prompt)).strip() or None
