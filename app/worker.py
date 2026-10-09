import os
import json
os.environ["CLOUDFLARE_WORKER"] = "1"

from fastapi import Request
from fastapi.responses import Response
from workers import asgi

from main import app, documents, retriever


@app.middleware("http")
async def cloudflare_index_storage(request: Request, call_next):
    """Hydrate per-isolate memory from KV and persist successful ingestion."""
    env = request.scope.get("env")
    kv = getattr(env, "RAG_INDEX", None) if env is not None else None
    if kv is not None:
        stored = await kv.get("index")
        if stored:
            retriever.load_payload(json.loads(stored))
            documents.clear()
            for chunk in retriever.chunks:
                documents[chunk.source] = documents.get(chunk.source, 0) + 1
    response = await call_next(request)
    if kv is not None and request.url.path == "/v1/ingest" and response.status_code < 300:
        await kv.put("index", json.dumps(retriever.payload()))
    return response


@app.get("/{path:path}")
async def static_assets(path: str, request: Request):
    env = request.scope["env"]
    asset_path = path or "index.html"
    response = await env.ASSETS.fetch(f"https://assets.local/{asset_path}")
    return Response(content=await response.bytes(), status_code=response.status, headers=response.headers)


Default = asgi.entrypoint(app)
