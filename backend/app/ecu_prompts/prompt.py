

def generate_rag_prompt(context: str, query: str) -> str:
    """
    Generates a structured prompt for the RAG system based on context and user query.
    """
    prompt = f"""
    Answer based on context:

    Context:
    {context}

    Question:
    {query}
    """
    return prompt
