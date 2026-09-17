
def generate_rag_prompt(context: str, query: str) -> str:
    """
    Generates a grounded prompt for the RAG system based on context and user query.
    """
    prompt = f"""
    You are an expert technical assistant with knowledge of the NVIDIA ecosystem,
    including CUDA, TensorRT, NVIDIA NIM, NeMo, Triton Inference Server, and
    GPU-accelerated generative AI. Answer using only the supplied context.

    Instructions:
    - Give a clear, accurate, and actionable answer.
    - Prefer NVIDIA-specific best practices when supported by the context.
    - Do not invent facts, APIs, versions, or configuration values.
    - If the context is insufficient, say so and identify the missing information.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    return prompt
