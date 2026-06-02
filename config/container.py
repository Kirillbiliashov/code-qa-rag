from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from config.settings import (
    COLLECTION_NAME,
    LLM_CONTEXT_WINDOW,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    MODEL_NAME,
    QDRANT_URL,
    VECTOR_SIZE,
)


class Container:
    def __init__(self, recreate_collection: bool = False):
        print(f"Initializing embedding model: {MODEL_NAME}")
        self.embedding_model: SentenceTransformer = SentenceTransformer(MODEL_NAME)

        print(f"Connecting to Qdrant at {QDRANT_URL}")
        self.qdrant_client: QdrantClient = QdrantClient(url=QDRANT_URL)
        self._init_collection(recreate=recreate_collection)

        print(f"Initializing LLM: {LLM_MODEL}")
        self.llm: LLM = Ollama(
            model=LLM_MODEL,
            request_timeout=LLM_REQUEST_TIMEOUT,
            temperature=LLM_TEMPERATURE,
            context_window=LLM_CONTEXT_WINDOW,
            verbose=True,
        )

    def _init_collection(self, recreate: bool):
        exists = self.qdrant_client.collection_exists(COLLECTION_NAME)
        if exists and recreate:
            print(f"Deleting existing collection: {COLLECTION_NAME}")
            self.qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            exists = False
        if not exists:
            print(f"Creating collection: {COLLECTION_NAME}")
            self.qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
