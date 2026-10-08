from .config import settings


def client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before starting the API") from exc
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def embed(texts: list[str]) -> list[list[float]]:
    response = client().embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def complete(system: str, user: str) -> str:
    response = client().chat.completions.create(model=settings.generation_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0)
    return response.choices[0].message.content or ""
