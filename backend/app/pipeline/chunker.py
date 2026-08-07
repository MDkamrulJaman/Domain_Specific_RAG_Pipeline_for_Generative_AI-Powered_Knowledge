from typing import List
# import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from loader import UniversalDocumentLoader


# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

class DocumentSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the text splitter service with configurable chunk parameters.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_documents(self, docs: List) -> List:
        """
        Splits a list of LangChain documents into smaller chunks.
        """
        return self.splitter.split_documents(docs)
    




    
# # Local Development Execution test
# def main():
#     logger.info("Testing document loading and chunking functionality...")
    
#     # 1. Path for test PDF
#     pdf_path = "data/raw"

#     try:
#         # 2. First, parse/load the PDF file into documents
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

#     except Exception as e:
#         logger.error(f"Loader/Chunker execution test failed: {str(e)}", exc_info=True)


# if __name__ == "__main__":
#     main()