import zipfile

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from api.models import AskRequest, AskResponse, UploadResponse
from ingestion.chunks_store import ChunksStore
from ingestion.vector_indexer import VectorIndexer
from services.ingestion_service import IngestionService
from services.qa_service import QAService

router = APIRouter()


@router.get("/")
async def index(request: Request):
    return FileResponse(request.app.state.index_file)


@router.post("/api/upload", response_model=UploadResponse)
async def upload_zip(request: Request, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip archives are accepted")

    contents = await file.read()
    try:
        chunks = IngestionService.extract_chunks_from_zip(contents)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="File is not a valid zip archive")

    IngestionService.persist_chunks(chunks)

    vector_indexer: VectorIndexer = request.app.state.vector_indexer
    vector_indexer.reindex_chunks(chunks)

    chunks_store: ChunksStore = request.app.state.chunks_store
    chunks_store.set(chunks)

    files_processed = len({c.file_path for c in chunks})
    return UploadResponse(chunks=len(chunks), files_processed=files_processed)


@router.post("/api/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    chunks_store: ChunksStore = request.app.state.chunks_store
    if chunks_store.count() == 0:
        raise HTTPException(status_code=400, detail="Upload a repo .zip first")

    qa_service: QAService = request.app.state.qa_service
    answer = qa_service.answer(question)
    return AskResponse(answer=answer)
