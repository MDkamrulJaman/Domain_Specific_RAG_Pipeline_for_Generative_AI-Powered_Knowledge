"""Provider-independent document indexing workflow with injected capabilities."""
from collections.abc import Callable
from app.services.contracts import (
    ChunkEmbedder, DocumentChunker, DocumentLoader, Progress, VectorWriter, ignore_progress,
)


class IndexingService:
    def __init__(self, writer: VectorWriter, loader: DocumentLoader,
                 chunker: DocumentChunker, embedder: Callable[[], ChunkEmbedder]):
        self.writer = writer
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder

    def index(self, filename: str, content: bytes, progress: Progress = ignore_progress):
        progress(0.15, "Extracting document text")
        documents = self.loader.load_bytes(filename, content)
        if not documents:
            raise ValueError("Document loader returned empty data.")
        progress(0.3, "Splitting document into chunks")
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            raise ValueError("Splitting resulted in 0 chunks.")
        embeddings = None
        if self.writer.integrated_embedding is True:
            progress(0.5, "Pinecone will embed the document text")
        else:
            progress(0.5, "Generating Hugging Face embeddings")
            embeddings = self.embedder().embed_chunks(chunks)
            if embeddings.size == 0:
                raise ValueError("Embedding service returned no embeddings.")
        progress(0.8, "Saving to the Pinecone index")
        self.writer.add(embeddings, chunks)
        self.writer.save()
        return {"status": "success", "storage": "shared_pinecone",
                "message": f"Successfully indexed '{filename}'", "chunks_created": len(chunks)}
