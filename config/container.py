from llama_index.core.llms import LLM
from llama_index.llms.deepseek import DeepSeek
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from config.settings import (
    COLLECTION_NAME,
    LLM_API_KEY,
    LLM_CONTEXT_WINDOW,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    MODEL_NAME,
    MONGO_DB_NAME,
    MONGO_URL,
    QDRANT_URL,
    VECTOR_SIZE,
)
from db.database import Database


class Container:
    def __init__(self, recreate_collection: bool = False):
        print(f"Initializing embedding model: {MODEL_NAME}")
        self.embedding_model: SentenceTransformer = SentenceTransformer(MODEL_NAME)
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        print(f"Connecting to Qdrant at {QDRANT_URL}")
        self.qdrant_client: QdrantClient = QdrantClient(url=QDRANT_URL)
        self._init_collection(recreate=recreate_collection)
        print(f"Connecting to MongoDB at {MONGO_URL}/{MONGO_DB_NAME}")
        self.database: Database = Database(MONGO_URL, MONGO_DB_NAME)

        print(f"Initializing LLM: {LLM_MODEL}")
        self.llm: LLM = DeepSeek(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=LLM_API_KEY,
            verbose=True,
        )

    def close(self) -> None:
        self.database.close()

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
                vectors_config={
                    "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
                    },
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                }
            )
