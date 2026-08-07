import logging
import sys
from pathlib import Path

# 1. CRITICAL FIX: Append path BEFORE importing custom app modules
# root_dir = Path(__file__).resolve().parent.parent.parent
# if str(root_dir) not in sys.path:
#     sys.path.append(str(root_dir))

# Now these imports will resolve properly without raising ModuleNotFoundError
from app.pipeline.retrieval_service import VectorStore
from app.services.llm_service import LLMService

# 2. Setup logging infrastructure so terminal output works out-of-the-box
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        try:
            self.vectorstore = VectorStore("data/vectorstore")
            self.llm = LLMService()
        except Exception as e:
            logger.critical(f"RAG component initialization failed: {str(e)}")
            raise

    def ask(self, query: str, top_k: int = 5) -> str:
        """
        Inverts the flow: First calls the LLM directly, then attempts to search 
        the vector store. If relevant vector store results are found, it follows up 
        with a context-infused prompt.
        """
        if not query or not query.strip():
            return "Please provide a valid query."

        try:
            # 1. Always call LLM first (or get initial baseline response)
            initial_response = self.llm.generate(query)

            # 2. Check vector store for matching query chunks
            if not self.vectorstore:
                logger.info("Vector store instance not found. Returning initial LLM response.")
                return initial_response if initial_response else "The local model failed to return a response structure."

            docs = self.vectorstore.search(query, top_k)
            
            # 3. If no vector chunks exist, stick to the initial LLM response
            if not docs:
                logger.info(f"No vector store matches found for query: {query}. Returning direct LLM response.")
                return initial_response if initial_response else "The local model failed to return a response structure."

            # 4. If query HAS vector storage matches, build the RAG context prompt and call LLM again
            logger.info(f"Vector store matches found for query: {query}. Generating context-enhanced response.")
            context = "\n".join([d.get("text", d.get("content", "")) for d in docs if d])

            prompt = f"""
            Answer the user's question based strictly on the provided context.

            Context:
            {context}

            Question:
            {query}
            
            Answer:
            """

            rag_response = self.llm.generate(prompt)
            return rag_response if rag_response else "The local model failed to return a response structure."

        except Exception as e:
            logger.error(f"Error executing pipeline logic: {str(e)}")
            return "An internal error occurred while executing the search pipeline."