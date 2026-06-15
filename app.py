from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config.container import Container
from config.settings import COLLECTION_NAME, DAILY_QUERY_QUOTA, QUOTA_WINDOW_HOURS
from inference.answer_generator import AnswerGenerator
from ingestion.vector_indexer import VectorIndexer
from retrieval.retriever import Retriever
from services.qa_service import QAService
from services.rate_limiter import RateLimiter


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"
QA_FILE = STATIC_DIR / "qa.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing container...")
    container = Container(recreate_collection=False)

    retriever = Retriever(
        llm=container.llm,
        embedding_model=container.embedding_model,
        sparse_embedding_model=container.sparse_embedding_model,
        qdrant_client=container.qdrant_client,
        collection_name=COLLECTION_NAME,
    )
    answer_generator = AnswerGenerator(llm=container.llm)

    app.state.container = container
    app.state.vector_indexer = VectorIndexer(
        embedding_model=container.embedding_model,
        sparse_embedding_model=container.sparse_embedding_model,
        qdrant_client=container.qdrant_client,
    )
    app.state.qa_service = QAService(
        retriever=retriever,
        database=container.database,
        answer_generator=answer_generator,
    )
    app.state.rate_limiter = RateLimiter(
        database=container.database,
        quota=DAILY_QUERY_QUOTA,
        window_hours=QUOTA_WINDOW_HOURS,
    )
    app.state.index_file = INDEX_FILE
    app.state.qa_file = QA_FILE
    print("Container ready. Server is up.")
    try:
        yield
    finally:
        container.close()


app = FastAPI(title="Code QA RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router)
