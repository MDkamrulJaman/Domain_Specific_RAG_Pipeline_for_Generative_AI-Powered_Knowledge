import logging
from typing import Optional

from langchain_ollama import ChatOllama

from app.core.config import Settings

# Setup logging
logger = logging.getLogger(__name__)


class LLMService:
    """Service class to interface with a local Ollama instance using LangChain."""

    def __init__(self):
        settings = Settings()

        self.base_url = settings.OLLAMA_BASE_URL
        self.model_name = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT_SECONDS

        try:
            # Initialize native LangChain Ollama Chat model
            self.client = ChatOllama(
                base_url=self.base_url,
                model=self.model_name,
                timeout=self.timeout,
                temperature=0.0  # Kept deterministic for predictable frontend pipeline outputs
            )
            logger.info(f"LLMService successfully initialized with LangChain targeting {self.base_url} [{self.model_name}]")
        except Exception as e:
            logger.critical(f"Failed to initialize LangChain ChatOllama client: {str(e)}")
            raise RuntimeError("Could not initialize LLMService client.") from e

    def generate(self, prompt: str) -> Optional[str]:
        """
        Sends a prompt to the local Ollama Gemma3 instance via LangChain 
        and extracts the text response safely for the frontend.
        """
        if not prompt or not prompt.strip():
            logger.warning("Received an empty or invalid prompt.")
            return None

        try:
            logger.debug(f"Invoking LangChain model '{self.model_name}'")
            
            # Simple, direct invoke using LangChain
            response = self.client.invoke(prompt)

            # Defensive validation against empty content objects
            if not response or not response.content:
                logger.error("Ollama returned an empty response or invalid content structure.")
                return None
                
            return str(response.content).strip()

        except ConnectionError as ce:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}. Ensure Ollama is running. Error: {str(ce)}")
            return None
            
        except Exception as e:
            # Catch-all for timeouts, missing local models (e.g. if gemma3 needs to be pulled), or parsing glitches
            logger.error(f"An error occurred during LLM text generation with {self.model_name}: {str(e)}")
            return None
        



# def main():
#     print("--- Initializing LLMService ---")
#     loader = LLMService()
    
#     # Define a test prompt for Gemma 3
#     test_prompt = "Say hello and tell me my name."
    
#     print(f"\n--- Sending Prompt to {loader.model_name} ---")
#     print(f"Prompt: '{test_prompt}'\n")
    
#     # Execute the generation (Passing the required prompt argument)
#     response = loader.generate(test_prompt)
    
#     print("--- Response Received ---")
#     if response:
#         print(response)
#     else:
#         print("Generation failed. Check the console logs above for connection or timeout issues.")


# if __name__ == "__main__":
#     main()

