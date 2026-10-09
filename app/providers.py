try:
    from .config import settings
except ImportError:
    from config import settings


def _local_embeddings(texts: list[str], dimensions: int = 128) -> list[list[float]]:
    """Small deterministic fallback for demos and offline smoke tests."""
    import hashlib
    import math
    vectors = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


def client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before starting the API") from exc
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def embed(texts: list[str]) -> list[list[float]]:
    if not settings.openai_api_key:
        return _local_embeddings(texts)
    response = client().embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def complete(system: str, user: str) -> str:
    if not settings.openai_api_key:
        # Extractive fallback keeps the demo grounded without inventing content.
        import re
        blocks = re.findall(r"\[(\d+)\]\s+(.*?)(?=\n\n|$)", user, flags=re.S)
        question = user.split("QUESTION:", 1)[-1].strip().lower()
        if not blocks:
            return "I don't know based on the indexed documents."
        query_terms = set(re.findall(r"\w+", question))
        ranked = sorted(blocks, key=lambda item: len(query_terms & set(re.findall(r"\w+", item[1].lower()))), reverse=True)
        number, text = ranked[0]
        sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0]
        return f"{sentence} [{number}]"
    response = client().chat.completions.create(model=settings.generation_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0)
    return response.choices[0].message.content or ""


def rewrite_query(question: str) -> str:
    if not settings.openai_api_key:
        return question
    return complete(
        "Rewrite search queries for internal documentation retrieval. Return only the rewritten query.",
        f"Rewrite this question into a concise search query preserving exact identifiers and error codes:\n{question}",
    ).strip() or question
