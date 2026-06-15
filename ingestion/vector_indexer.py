import uuid

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from sentence_transformers import SentenceTransformer

from config.settings import COLLECTION_NAME
from ingestion.chunks import FileChunk


class VectorIndexer:
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        sparse_embedding_model: SparseTextEmbedding,
        qdrant_client: QdrantClient,
        batch_size: int = 64,
    ):
        self.embedding_model = embedding_model
        self.sparse_embedding_model = sparse_embedding_model
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

            dense_embeddings = self.embedding_model.encode(
                batch_texts, show_progress_bar=False, normalize_embeddings=True
            )
            sparse_embeddings = list(self.sparse_embedding_model.embed(batch_texts))

            if len(sparse_embeddings) != len(batch_texts):
                raise ValueError(
                    f"sparse embedder produced {len(sparse_embeddings)} vectors "
                    f"for {len(batch_texts)} input texts"
                )
            if len(dense_embeddings) != len(batch_texts):
                raise ValueError(
                    f"dense embedder produced {len(dense_embeddings)} vectors "
                    f"for {len(batch_texts)} input texts"
                )

            for j, (dense, sparse) in enumerate(zip(dense_embeddings, sparse_embeddings)):
                d = chunk_dicts[i + j]
                dense_vector = dense.tolist()

                indices = sparse.indices.tolist()
                values = sparse.values.tolist()
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense_vector,
                            "bm25": SparseVector(indices=indices, values=values),
                        },
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
