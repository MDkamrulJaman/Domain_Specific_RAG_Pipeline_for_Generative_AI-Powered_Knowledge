
from pathlib import Path
import logging
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

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
                docs.extend(self.load_file(file))

            except Exception as e:
                # Log the error with stack trace details, but don't crash the loop
                logger.error(f"Error loading file {file.name}: {str(e)}", exc_info=True)

        return docs

    def load_file(self, file_path: str | Path) -> List[Any]:
        """Load one supported file so an upload cannot re-index older files."""
        file = Path(file_path)
        suffix = file.suffix.lower()

        if suffix == ".pdf":
            return PyPDFLoader(str(file)).load()
        if suffix == ".txt":
            return [
                Document(
                    page_content=file.read_text(encoding="utf-8", errors="replace"),
                    metadata={"source": str(file)},
                )
            ]
        raise ValueError(f"Unsupported file type: {suffix}")


