from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.routes import router
from config.container import Container
from ingestion.vector_indexer import VectorIndexer
from services.qa_service import QAService


BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing container...")
    container = Container(recreate_collection=True)
    app.state.container = container
    app.state.qa_service = QAService(container)
    app.state.vector_indexer = VectorIndexer(
        embedding_model=container.embedding_model,
        qdrant_client=container.qdrant_client,
    )
    app.state.chunks_count = 0
    app.state.index_file = INDEX_FILE
    print("Container ready. Server is up.")
    yield


app = FastAPI(title="Code QA RAG", lifespan=lifespan)
app.include_router(router)
