
from io import BytesIO
from pathlib import Path
import logging
from typing import List, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Professional logging setup
logger = logging.getLogger(__name__)


class UniversalDocumentLoader:
    def __init__(self, target_directory: str | None = None):
        """
        Initializes the loader with an optional target directory path.
        For in-memory uploads, a directory is not required.
        """
        self.target_path = Path(target_directory) if target_directory else None

    def load_all_documents(self) -> List[Any]:
        """
        Scans the target directory, detects file types (.pdf)
        and extracts LangChain Document objects safely
        """
        docs = []

        if not self.target_path:
            logger.error("Directory-less loader cannot scan files from disk.")
            return docs

        if not self.target_path.exists():
            logger.error(f"Directory '{self.target_path}' does not exist.")
            return docs

        if not self.target_path.is_dir():
            logger.error(f"Path '{self.target_path}' is a file, but a directory is required.")
            return docs

        for file in self.target_path.iterdir():
            if file.is_dir() or file.name.startswith('.'):
                continue

            try:
                docs.extend(self.load_file(file))
            except Exception as e:
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

    def load_bytes(self, filename: str, file_bytes: bytes) -> List[Any]:
        """Load a file directly from bytes without saving it to disk."""
        if not filename:
            raise ValueError("A valid filename is required to parse uploaded content.")

        suffix = Path(filename).suffix.lower()

        if suffix == ".txt":
            text = file_bytes.decode("utf-8", errors="replace")
            return [
                Document(
                    page_content=text,
                    metadata={"source": filename},
                )
            ]

        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text:
                    pages.append(text)

            return [
                Document(
                    page_content="\n".join(pages),
                    metadata={"source": filename},
                )
            ]

        raise ValueError(f"Unsupported file type: {suffix}")


