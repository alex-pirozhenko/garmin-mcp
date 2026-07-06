import os
import subprocess
import sys

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

app = FastAPI()

PORT = 8000
PROXY_PORT = 8001
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"

mcp_proxy_proc = None
proxy_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def startup_event():
    global mcp_proxy_proc, proxy_client
    env = dict(os.environ)
    env["GARMIN_MCP_TRANSPORT"] = "streamable-http"
    env["GARMIN_MCP_HOST"] = "127.0.0.1"
    env["GARMIN_MCP_PORT"] = str(PROXY_PORT)
    cmd = ["garmin-mcp"]
    print(f"Starting garmin-mcp: {' '.join(cmd)}")
    mcp_proxy_proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    # A single long-lived client, not one `async with`-scoped per request: a
    # per-request `async with AsyncClient()` closes (and kills the socket)
    # the instant the route handler *returns* a StreamingResponse, before
    # Starlette actually drains resp.aiter_raw() — which raised
    # httpx.ReadError on every SSE/streamable-http response from upstream.
    proxy_client = httpx.AsyncClient(base_url=PROXY_URL, timeout=120.0)


@app.on_event("shutdown")
async def shutdown_event():
    global mcp_proxy_proc, proxy_client
    if proxy_client:
        await proxy_client.aclose()
    if mcp_proxy_proc:
        print("Terminating garmin-mcp...")
        mcp_proxy_proc.terminate()
        mcp_proxy_proc.wait()


@app.get("/health")
async def health():
    global mcp_proxy_proc
    if mcp_proxy_proc is None or mcp_proxy_proc.poll() is not None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "garmin-mcp is not running"},
        )
    return {"status": "ok", "service": "garmin-mcp-shim"}


@app.get("/.well-known/oauth-protected-resource")
async def protected_resource_metadata():
    resource_url = os.environ.get(
        "GARMIN_MCP_RESOURCE_URL", "https://garmin-mcp.pirozhenko.me"
    )
    auth_server_url = os.environ.get(
        "GARMIN_MCP_AUTH_SERVER_URL", "https://auth.pirozhenko.me"
    )
    return {
        "resource": resource_url,
        "authorization_servers": [auth_server_url],
    }


@app.api_route("/mcp-unauthorized", methods=["GET", "POST", "DELETE"])
async def mcp_unauthorized():
    resource_url = os.environ.get(
        "GARMIN_MCP_RESOURCE_URL", "https://garmin-mcp.pirozhenko.me"
    )
    return JSONResponse(
        status_code=401,
        headers={
            "WWW-Authenticate": (
                'Bearer error="invalid_token", '
                f'resource_metadata="{resource_url}/.well-known/oauth-protected-resource"'
            )
        },
        content={
            "error": "unauthorized",
            "error_description": "Bearer token required",
        },
    )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS"],
)
async def reverse_proxy(request: Request, path: str):
    client = proxy_client
    url = f"/{path}"
    if request.query_params:
        url += f"?{request.query_params}"

    req_headers = dict(request.headers)
    req_headers.pop("host", None)

    req_body = await request.body()

    proxy_req = client.build_request(
        method=request.method,
        url=url,
        headers=req_headers,
        content=req_body,
    )

    resp = await client.send(proxy_req, stream=True)

    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        # background=resp.aclose closes *this response's* stream once fully
        # sent — the shared client itself stays open across requests.
        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=dict(resp.headers),
            background=BackgroundTask(resp.aclose),
        )
    else:
        # Must use aread() (async), not read() (sync) — this response was
        # obtained via AsyncClient.send(..., stream=True).
        await resp.aread()
        await resp.aclose()
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
