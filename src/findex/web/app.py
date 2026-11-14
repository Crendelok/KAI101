# pyright: reportUnusedFunction=false
# (route handlers are registered by their decorators; nothing calls them by name)
"""FastAPI app: the index is loaded once (lifespan) and injected with ``Depends``.

INDEX_PATH=index.fidx uvicorn findex.web.app:create_app --factory
findex serve --index index.fidx          # same thing, via the CLI
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from findex import __version__
from findex.index import Index
from findex.query import QueryError
from findex.scoring import get_scorer
from findex.search import search, snippet_for
from findex.store import load
from findex.web.models import (
    DocumentResponse,
    ErrorResponse,
    Health,
    Mode,
    SearchHit,
    SearchResponse,
    StatsResponse,
)
from findex.web.settings import Settings, get_settings

log = logging.getLogger("findex.web")
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")


# --- dependencies ------------------------------------------------------------------
# All trivial, so all `async def`: a *sync* dependency costs a threadpool round-trip.


async def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_index_optional(request: Request) -> Index | None:
    return request.app.state.index


async def get_index(index: Annotated[Index | None, Depends(get_index_optional)]) -> Index:
    if index is None:
        raise HTTPException(status_code=503, detail="index not loaded")
    return index


async def get_semantic(request: Request) -> Any:
    return getattr(request.app.state, "semantic", None)


IndexDep = Annotated[Index, Depends(get_index)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SemanticDep = Annotated[Any, Depends(get_semantic)]
QParam = Annotated[str, Query(min_length=1, max_length=500, description="Query text.")]
KParam = Annotated[int, Query(ge=1, le=50, description="Results per page.")]
PageParam = Annotated[int, Query(ge=1, le=1000, description="1-based page number.")]
ScorerParam = Annotated[Literal["bm25", "tfidf"], Query(description="Ranking function.")]
ModeParam = Annotated[Mode, Query(description="keyword | semantic | hybrid")]


# --- the search itself (shared by the API and the HTML UI) ---------------------------


def run_search(
    index: Index,
    q: str,
    *,
    k: int,
    page: int,
    scorer: Literal["bm25", "tfidf"],
    mode: Mode = "keyword",
    semantic: Any = None,
) -> SearchResponse:
    t0 = time.perf_counter()
    offset = (page - 1) * k
    if mode != "keyword":
        if semantic is None:
            raise HTTPException(
                status_code=400, detail=f"{mode} search is not enabled on this server"
            )
        from findex.semantic import search_mode

        results = search_mode(index, semantic, q, mode=mode, k=k, offset=offset, scorer=scorer)
    else:
        try:
            results = search(index, q, get_scorer(scorer), k, offset=offset)
        except QueryError as exc:
            raise HTTPException(status_code=400, detail=f"bad query: {exc}") from exc
    hits: list[SearchHit] = []
    for r in results:
        snip = snippet_for(index, r.doc_id, results.terms)
        hits.append(
            SearchHit(
                doc_id=r.doc_id,
                title=r.title,
                score=round(r.score, 4),
                snippet=snip.text,
                snippet_html=snip.html(),
            )
        )
    return SearchResponse(
        query=q,
        mode=mode,
        scorer=scorer,
        total=results.total,
        page=page,
        k=k,
        pages=max(1, math.ceil(results.total / k)),
        took_ms=round((time.perf_counter() - t0) * 1000, 2),
        results=hits,
    )


# --- app factory ------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        s = settings or get_settings()
        app.state.settings = s
        app.state.started = time.time()
        app.state.index = None
        app.state.semantic = None
        try:
            # Loading touches the disk: do it off the event loop, once, before serving.
            app.state.index = await run_in_threadpool(load, s.index_path, lazy=True)
            log.info("loaded %r from %s", app.state.index, s.index_path)
        except (FileNotFoundError, ValueError) as exc:
            log.error("index not loaded (%s): /health will report 503", exc)
        emb = s.embeddings_path
        if app.state.index is not None and emb is None:
            guess = s.index_path.with_suffix(".emb.npy")
            emb = guess if guess.exists() else None
        if app.state.index is not None and emb is not None:
            try:
                from findex.semantic import SemanticIndex

                app.state.semantic = await run_in_threadpool(SemanticIndex.load, emb)
                log.info("semantic search enabled (%s)", emb)
            except (ImportError, OSError, ValueError) as exc:
                log.warning("semantic search disabled: %s", exc)
        yield
        if app.state.index is not None:
            app.state.index.close()

    app = FastAPI(
        title="findex",
        version=__version__,
        description="Full-text search over a corpus you choose: BM25, phrases, boolean queries.",
        lifespan=lifespan,
        responses={503: {"model": ErrorResponse}},
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("[%s] unhandled error on %s %s", rid, request.method, request.url.path)
            response = JSONResponse({"detail": "internal server error", "request_id": rid}, 500)
        response.headers["x-request-id"] = rid
        log.info(
            "[%s] %s %s -> %d (%.1f ms)",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - t0) * 1000,
        )
        return response

    # ---- API ----
    s = settings or get_settings()
    if s.search_async:
        # Lab 7 experiment: CPU-bound work in `async def` runs ON the event loop.
        @app.get("/search", response_model=SearchResponse, tags=["api"])
        async def search_api_async(
            index: IndexDep,
            semantic: SemanticDep,
            q: QParam,
            k: KParam = 10,
            page: PageParam = 1,
            scorer: ScorerParam = "bm25",
            mode: ModeParam = "keyword",
        ) -> SearchResponse:
            return run_search(index, q, k=k, page=page, scorer=scorer, mode=mode, semantic=semantic)
    else:
        # `def`: FastAPI runs it in a worker thread, so the event loop stays free.
        @app.get("/search", response_model=SearchResponse, tags=["api"])
        def search_api(
            index: IndexDep,
            semantic: SemanticDep,
            q: QParam,
            k: KParam = 10,
            page: PageParam = 1,
            scorer: ScorerParam = "bm25",
            mode: ModeParam = "keyword",
        ) -> SearchResponse:
            return run_search(index, q, k=k, page=page, scorer=scorer, mode=mode, semantic=semantic)

    @app.get(
        "/docs/{doc_id}",
        response_model=DocumentResponse,
        tags=["api"],
        responses={404: {"model": ErrorResponse}},
    )
    def document(index: IndexDep, doc_id: int) -> DocumentResponse:
        if not 0 <= doc_id < index.num_docs:
            raise HTTPException(status_code=404, detail=f"no document {doc_id}")
        meta = index.doc_meta[doc_id]
        return DocumentResponse(
            doc_id=doc_id,
            title=meta.title,
            path=meta.path,
            length=index.doc_length(doc_id),
            text=index.text(doc_id) or "",
        )

    @app.get("/stats", response_model=StatsResponse, tags=["api"])
    def stats(
        request: Request, index: IndexDep, settings: SettingsDep, semantic: SemanticDep
    ) -> StatsResponse:
        path = settings.index_path
        return StatsResponse(
            version=__version__,
            documents=index.num_docs,
            terms=len(index),
            total_tokens=index.total_tokens,
            avg_doc_length=round(index.avg_doc_length, 1),
            positions=index.has_positions,
            index_bytes=path.stat().st_size if path.exists() else 0,
            semantic=semantic is not None,
            uptime_s=round(time.time() - request.app.state.started, 1),
        )

    @app.get(
        "/health", response_model=Health, tags=["ops"], responses={503: {"model": ErrorResponse}}
    )
    async def health(index: Annotated[Index | None, Depends(get_index_optional)]) -> Health:
        # `async def`: no blocking work, so it runs on the event loop and never waits for a
        # threadpool slot — the liveness probe stays responsive while searches run.
        if index is None:
            raise HTTPException(status_code=503, detail="index not loaded")
        return Health(status="ok")

    # ---- HTML UI ----
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        settings: SettingsDep,
        index: Annotated[Index | None, Depends(get_index_optional)],
        semantic: SemanticDep,
        q: str = "",
        page: Annotated[int, Query(ge=1, le=1000)] = 1,
        scorer: Literal["bm25", "tfidf"] = "bm25",
        mode: Mode = "keyword",
    ) -> HTMLResponse:
        ctx: dict[str, Any] = {
            "q": q,
            "page": page,
            "scorer": scorer,
            "mode": mode,
            "title": settings.site_title,
            "res": None,
            "error": None,
            "index": index,
            "semantic": semantic is not None,
        }
        if q.strip() and index is not None:
            try:
                ctx["res"] = run_search(
                    index,
                    q,
                    k=settings.default_k,
                    page=page,
                    scorer=scorer,
                    mode=mode,
                    semantic=semantic,
                )
            except HTTPException as exc:
                ctx["error"] = exc.detail
        return templates.TemplateResponse(request, "index.html", ctx)

    @app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def doc_page(
        request: Request,
        index: IndexDep,
        settings: SettingsDep,
        doc_id: int,
        q: str = "",
    ) -> HTMLResponse:
        if not 0 <= doc_id < index.num_docs:
            raise HTTPException(status_code=404, detail=f"no document {doc_id}")
        from findex.snippets import Snippet
        from findex.tokenize import find_tokens, tokenize

        text = index.text(doc_id) or ""
        terms = set(tokenize(q))
        spans = tuple((s, e) for tok, s, e in find_tokens(text) if tok in terms)
        body = Snippet(text, spans).html()
        return templates.TemplateResponse(
            request,
            "doc.html",
            {
                "title": settings.site_title,
                "doc_title": index.title(doc_id),
                "body": body,
                "q": q,
                "meta": index.doc_meta[doc_id],
                "length": index.doc_length(doc_id),
            },
        )

    return app


def app() -> FastAPI:  # for `uvicorn findex.web.app:app --factory`
    return create_app()
