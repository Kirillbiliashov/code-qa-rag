import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from config.settings import COLLECTION_NAME
from ingestion.chunks import FileChunk


class VectorIndexer:
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        qdrant_client: QdrantClient,
        batch_size: int = 64,
    ):
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        self.batch_size = batch_size

    def index_chunks(self, chunks: list[FileChunk], repo_id: str) -> int:
        if not chunks:
            return 0

        chunk_dicts = [c.to_json_dict() for c in chunks]
        texts = [d["retrieval_text"] for d in chunk_dicts]

        points: list[PointStruct] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            embeddings = self.embedding_model.encode(
                batch_texts, show_progress_bar=False, normalize_embeddings=True
            )
            for j, emb in enumerate(embeddings):
                d = chunk_dicts[i + j]
                vector = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            **d["metadata"],
                            "semantic_id": d["id"],
                            "repo_id": repo_id,
                        },
                    )
                )

        self.qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points,
        )
        return len(points)
