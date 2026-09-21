"""Uddhar FastAPI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import allocate, events, graph, inventory, report, upload
from core import db
from core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("uddhar")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    db.connect()
    if await db.ping():
        await db.init_indexes()
    else:
        log.warning("mongodb unreachable at %s - API will serve but writes will fail",
                    settings.mongodb_uri)
    log.info("device=%s sat_backend=%s ground_backend=%s",
             settings.resolved_device(), settings.sat_backend, settings.ground_backend)
    yield
    await db.close()


app = FastAPI(
    title="Uddhar",
    version="0.1.0",
    description=(
        "Multi-modal disaster damage classification and web-grounded "
        "resource allocation."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events.router)
app.include_router(upload.router)
app.include_router(inventory.router)
app.include_router(graph.router)
app.include_router(allocate.router)
app.include_router(report.router)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """A malformed object id is the caller's mistake, not a server fault."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness plus the two things that silently break everything: the DB
    connection and whether we actually got a GPU."""
    settings = get_settings()
    cuda = False
    gpu_name = None
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "status": "ok",
        "version": "0.1.0",
        "mongodb": await db.ping(),
        "cuda": cuda,
        "gpu": gpu_name,
        "device": settings.resolved_device(),
        "sat_backend": settings.sat_backend,
        "ground_backend": settings.ground_backend,
    }
