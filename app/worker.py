import os
os.environ["CLOUDFLARE_WORKER"] = "1"

from fastapi import Request
from fastapi.responses import Response
from workers import asgi

from main import app


@app.get("/{path:path}")
async def static_assets(path: str, request: Request):
    env = request.scope["env"]
    asset_path = path or "index.html"
    response = await env.ASSETS.fetch(f"https://assets.local/{asset_path}")
    return Response(content=await response.bytes(), status_code=response.status, headers=response.headers)


Default = asgi.entrypoint(app)
