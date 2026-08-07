
from pathlib import Path
import logging
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader

# Professional logging setup
logger = logging.getLogger(__name__)



class UniversalDocumentLoader:
    def __init__(self, target_directory: str ):
        """
        Initializes the loader with a target directory path.
        """
        if not target_directory:
            logger.error("Initialization failed: target_directory path is missing or empty.")
            raise ValueError("A valid target directory path must be provided.")
            
        self.target_path = Path(target_directory)

    def load_all_documents(self) -> List[Any]:
        """
        Scans the target directory, detects file types (.pdf)
        and extracts LangChain Document objects safely
        """
        docs = []

        if not self.target_path.exists():
            logger.error(f"Directory '{self.target_path}' does not exist.")
            return docs

        if not self.target_path.is_dir():
            logger.error(f"Path '{self.target_path}' is a file, but a directory is required.")
            return docs

        # Iterate through all files in the directory
        for file in self.target_path.iterdir():
            # Skip sub-directories and hidden files
            if file.is_dir() or file.name.startswith('.'):
                continue

            try:
                suffix = file.suffix.lower()
                
                if suffix == ".pdf":
                    loader = PyPDFLoader(str(file))
                    docs.extend(loader.load())
                    logger.info(f"Successfully loaded PDF: {file.name}")

                else:
                    logger.warning(f"Skipping unsupported file type: {file.name}")

            except Exception as e:
                # Log the error with stack trace details, but don't crash the loop
                logger.error(f"Error loading file {file.name}: {str(e)}", exc_info=True)

        return docs




# # Local Development Execution test
# def main():
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#     logger.info("Testing UniversalDocumentLoader class functionality...")
    
#     # Example local test run:
#     test_dir = "data/raw"

#     try:
#         loader = UniversalDocumentLoader(target_directory=test_dir)
#         documents = loader.load_all_documents()
#         logger.info(f"Test complete. Total LangChain documents loaded: {len(documents)}")
#     except Exception as e:
#         logger.error(f"Loader execution test failed: {e}")


# if __name__ == "__main__":
#     main()