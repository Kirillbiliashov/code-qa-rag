import zipfile

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from api.models import AskRequest, AskResponse, QuotaResponse, UploadResponse
from db.database import Database
from ingestion.vector_indexer import VectorIndexer
from services.ingestion_service import IngestionService
from services.qa_service import QAService
from services.rate_limiter import RateLimiter

router = APIRouter()


@router.get("/")
async def index(request: Request):
    return FileResponse(request.app.state.index_file)


@router.get("/qa/{repo_id}")
async def qa_page(request: Request, repo_id: str):
    database: Database = request.app.state.container.database
    if not database.get_repo(repo_id):
        raise HTTPException(status_code=404, detail="Repository not found")
    return FileResponse(request.app.state.qa_file)


@router.post("/api/upload", response_model=UploadResponse)
async def upload_zip(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip archives are accepted")

    contents = await file.read()
    try:
        chunks, fingerprint = IngestionService.extract_chunks_from_zip(contents)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File is not a valid zip archive")

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No Python files found in the archive. Please upload a repository that contains .py files.",
        )

    database: Database = request.app.state.container.database

    existing = database.get_repo_by_fingerprint(fingerprint)
    if existing is not None:
        return UploadResponse(
            repo_id=str(existing["_id"]),
            chunks=existing.get("chunks_count", len(chunks)),
            files_processed=len({c.file_path for c in chunks}),
        )

    repo_id = database.create_repo(
        name=file.filename,
        size=len(contents),
        chunks_count=len(chunks),
        fingerprint=fingerprint,
    )
    database.insert_chunks(repo_id, chunks)

    vector_indexer: VectorIndexer = request.app.state.vector_indexer
    vector_indexer.index_chunks(chunks, repo_id)

    files_processed = len({c.file_path for c in chunks})
    return UploadResponse(
        repo_id=repo_id,
        chunks=len(chunks),
        files_processed=files_processed,
    )


@router.get("/api/quota", response_model=QuotaResponse)
async def quota(request: Request):
    rate_limiter: RateLimiter = request.app.state.rate_limiter
    ip = request.client.host if request.client else "unknown"
    return _quota_payload(rate_limiter.state(ip))


@router.post("/api/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    database: Database = request.app.state.container.database
    if not database.get_repo(body.repo_id):
        raise HTTPException(status_code=404, detail="Repository not found")

    rate_limiter: RateLimiter = request.app.state.rate_limiter
    ip = request.client.host if request.client else "unknown"
    allowed, blocking_state = rate_limiter.check(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily query limit reached ({blocking_state.quota} per "
                f"{rate_limiter.window.total_seconds() / 3600:.0f}h). "
                f"Resets at {blocking_state.quota_reset.isoformat()}."
            ),
        )

    qa_service: QAService = request.app.state.qa_service
    answer = await qa_service.answer(body.repo_id, question)

    rate_limiter.record(ip)
    return AskResponse(answer=answer, quota=_quota_payload(rate_limiter.state(ip)))


def _quota_payload(state) -> QuotaResponse:
    return QuotaResponse(
        queries_count=state.queries_count,
        quota=state.quota,
        quota_reset=state.quota_reset,
    )
