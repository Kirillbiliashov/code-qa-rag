import zipfile

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from api.models import AskRequest, AskResponse, UploadResponse
from db.database import Database
from ingestion.vector_indexer import VectorIndexer
from services.ingestion_service import IngestionService
from services.qa_service import QAService

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
        chunks = IngestionService.extract_chunks_from_zip(contents)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File is not a valid zip archive")

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No Python files found in the archive. Please upload a repository that contains .py files.",
        )

    database: Database = request.app.state.container.database
    repo_id = database.create_repo(
        name=file.filename,
        size=len(contents),
        chunks_count=len(chunks),
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


@router.post("/api/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    database: Database = request.app.state.container.database
    if not database.get_repo(body.repo_id):
        raise HTTPException(status_code=404, detail="Repository not found")

    qa_service: QAService = request.app.state.qa_service
    answer = qa_service.answer(body.repo_id, question)
    return AskResponse(answer=answer)
