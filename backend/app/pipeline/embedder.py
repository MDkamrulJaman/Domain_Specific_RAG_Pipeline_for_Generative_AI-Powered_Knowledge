import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
import logging


# from loader import UniversalDocumentLoader
# from chunker import DocumentSplitter


# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer model safely.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            # In production, you'd use a proper logger here
            print(f"CRITICAL: Failed to load embedding model '{model_name}': {e}")
            self.model = None

    def _ensure_model_loaded(self):
        """Internal helper to verify the model exists before encoding."""
        if self.model is None:
            raise RuntimeError(
                "Embedding model is not initialized. Check your internet connection or model name."
            )

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """
        Extracts text from LangChain-style chunks and returns a float32 NumPy array of vectors.
        """
        self._ensure_model_loaded()
        
        # Extract page content from the chunk objects
        texts = [c.page_content for c in chunks]
        
        embeddings = self.model.encode(texts)
        return np.array(embeddings).astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Converts a single string query into a 2D float32 NumPy vector array.
        """
        self._ensure_model_loaded()
        
        embedding = self.model.encode(query)
        # Keeps the 2D array structure matching your original code: shape (1, dimensions)
        return np.array([embedding]).astype("float32")
    




   
# # Local Development Execution test
# def main():
#     logger.info("Testing document loading and chunking functionality...")
    
#     # 1. Path for test PDF
#     pdf_path = "data/raw"

#     try:
#         # 2. First, load the PDF file into documents
#         logger.info(f"Attempting to load: {pdf_path}")
#         loader_instance = UniversalDocumentLoader(str(pdf_path))
#         docs = loader_instance.load_all_documents()
        
#         if not docs:
#             logger.error("Loader returned an empty document list.")
#             return

#         logger.info(f"Successfully loaded document content. Parsing chunks next...")

#         # 3. Pass those loaded documents into your DocumentSplitter chunking method
#         chunks_loader = DocumentSplitter()
#         chunks = chunks_loader.chunk_documents(docs)
        
#         # 4. Log out the results
#         if chunks:
#             logger.info(f"Success! Created {len(chunks)} text chunks.")
#             logger.info("--- Sample Chunk 1 Preview ---")
#             # Print out the first chunk text so you can visually verify the split quality
#             logger.info(chunks[0] if isinstance(chunks[0], str) else getattr(chunks[0], 'page_content', chunks[0]))
#         else:
#             logger.warning("Chunking returned 0 chunks. Check your splitting logic strategy.")

#         # 3. Pass those loaded documents into your DocumentSplitter chunking method
#         embeddings_instance = EmbeddingService()
#         embeddings= embeddings_instance.embed_chunks(chunks)
        
#         # 4. Log out the results
#         if embeddings.any():  
#             logger.info(f"Success! Created {embeddings.shape} .")
#         else:
#             logger.error(f"Embedding generation failed for {str(e)}")

#     except Exception as e:
#         logger.error(f"embedding execution test failed: {str(e)}", exc_info=True)

    
# if __name__ == "__main__":
#     main()