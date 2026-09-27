"""Pure validation of Pinecone index descriptions; no client construction."""
from collections.abc import Mapping
from fastapi import HTTPException
from pinecone.models.indexes.schema import DenseVectorField, SemanticTextField, SparseVectorField


def _read_field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def validate_index_dimension(index_info, index_name: str, expected: int):
    """Read v10 schema fields without triggering deprecated IndexModel accessors."""
    schema = _read_field(index_info, "schema")
    if schema is not None:
        fields = _read_field(schema, "fields", {}) or {}
        dense = []
        kinds = []
        for name, field in fields.items():
            if isinstance(field, DenseVectorField):
                kind = "dense_vector"
            elif isinstance(field, SemanticTextField):
                kind = "semantic_text"
            elif isinstance(field, SparseVectorField):
                kind = "sparse_vector"
            else:
                kind = _read_field(field, "type", type(field).__name__)
            kinds.append(f"{name}: {kind}")
            if kind == "dense_vector":
                dense.append((name, _read_field(field, "dimension")))
        if len(dense) != 1:
            problem = "no dense-vector field" if not dense else "multiple dense-vector fields"
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Pinecone index '{index_name}' has {problem} "
                    f"(schema: {', '.join(kinds) or 'empty'}). "
                    f"This shared Hugging Face pipeline requires an index with one dense-vector field "
                    f"of dimension {expected}. Select or create a compatible index and set "
                    "PINECONE_INDEX_NAME to its name, then upload your documents there. "
                    "A semantic_text field uses Pinecone-managed embeddings and cannot replace "
                    "the Hugging Face vector field. Your existing index has not been modified."
                ),
            )
        dimension = dense[0][1]
    else:
        # Compatibility with older API responses that actually contain a top-level dimension.
        dimension = _read_field(index_info, "dimension")
    if dimension != expected:
        actual = dimension if dimension is not None else "unknown"
        raise HTTPException(
            status_code=503,
            detail=(f"Pinecone index '{index_name}' has dimension {actual}; "
                    f"PINECONE_DIMENSION is {expected}. Use a dense index whose dimension "
                    "matches your Hugging Face embedding model, then re-upload documents if changing indexes."),
        )
    return dimension


