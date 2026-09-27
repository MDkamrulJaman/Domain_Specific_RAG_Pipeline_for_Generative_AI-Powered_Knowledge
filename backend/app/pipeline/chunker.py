from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:
    """Split documents while preserving source metadata."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )

    def chunk_documents(self, docs):
        return self.splitter.split_documents(docs)
