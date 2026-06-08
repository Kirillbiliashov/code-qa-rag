from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.routes import router
from config.container import Container
from config.settings import COLLECTION_NAME
from inference.answer_generator import AnswerGenerator
from ingestion.chunks_store import ChunksStore
from ingestion.vector_indexer import VectorIndexer
from retrieval.retriever import Retriever
from services.qa_service import QAService


BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing container...")
    container = Container(recreate_collection=True)

    chunks_store = ChunksStore()
    retriever = Retriever(
        llm=container.llm,
        embedding_model=container.embedding_model,
        qdrant_client=container.qdrant_client,
        collection_name=COLLECTION_NAME,
    )
    answer_generator = AnswerGenerator(llm=container.llm)

    app.state.container = container
    app.state.chunks_store = chunks_store
    app.state.vector_indexer = VectorIndexer(
        embedding_model=container.embedding_model,
        qdrant_client=container.qdrant_client,
    )
    app.state.qa_service = QAService(
        retriever=retriever,
        chunks_store=chunks_store,
        answer_generator=answer_generator,
    )
    app.state.index_file = INDEX_FILE
    print("Container ready. Server is up.")
    yield


app = FastAPI(title="Code QA RAG", lifespan=lifespan)
app.include_router(router)
