from ingestion.chunks import FileChunk


class ChunksStore:
    def __init__(self):
        self._chunks: dict[str, FileChunk] = {}

    def set(self, chunks: list[FileChunk]) -> None:
        self._chunks = {c.id: c for c in chunks}

    def get(self, chunk_id: str) -> FileChunk | None:
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        return len(self._chunks)
