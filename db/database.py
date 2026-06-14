from datetime import datetime, timezone

from bson import ObjectId
from pymongo import MongoClient

from ingestion.chunks import FileChunk


class Database:
    def __init__(self, url: str, db_name: str):
        self._client: MongoClient = MongoClient(url)
        self._db = self._client[db_name]
        self.repos = self._db["repos"]
        self.code_chunks = self._db["code_chunks"]
        self._ensure_indexes()

    def close(self) -> None:
        self._client.close()

    def _ensure_indexes(self) -> None:
        self.code_chunks.create_index([("repo_id", 1), ("chunk_id", 1)], unique=True)
        self.code_chunks.create_index([("repo_id", 1)])
        self.repos.create_index(
            [("fingerprint", 1)],
            unique=True,
            partialFilterExpression={"fingerprint": {"$type": "string"}},
        )

    def create_repo(
        self, name: str, size: int, chunks_count: int, fingerprint: str
    ) -> str:
        result = self.repos.insert_one({
            "name": name,
            "size": size,
            "chunks_count": chunks_count,
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc),
        })
        return str(result.inserted_id)

    def get_repo(self, repo_id: str) -> dict | None:
        obj_id = _to_object_id(repo_id)
        if obj_id is None:
            return None
        return self.repos.find_one({"_id": obj_id})

    def get_repo_by_fingerprint(self, fingerprint: str) -> dict | None:
        return self.repos.find_one({"fingerprint": fingerprint})

    def insert_chunks(self, repo_id: str, chunks: list[FileChunk]) -> int:
        if not chunks:
            return 0
        repo_obj_id = ObjectId(repo_id)
        docs = []
        for chunk in chunks:
            d = chunk.to_json_dict()
            docs.append({
                "repo_id": repo_obj_id,
                "chunk_id": d["id"],
                "file_path": chunk.file_path,
                "type": chunk.type,
                "code": getattr(chunk, "code", None) or chunk.to_embedding_text(),
                "metadata": d["metadata"],
            })
        self.code_chunks.insert_many(docs)
        return len(docs)

    def get_chunks(self, repo_id: str, chunk_ids: list[str]) -> list[dict]:
        repo_obj_id = _to_object_id(repo_id)
        if repo_obj_id is None or not chunk_ids:
            return []
        return list(self.code_chunks.find({
            "repo_id": repo_obj_id,
            "chunk_id": {"$in": chunk_ids},
        }))


def _to_object_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except Exception:
        return None
