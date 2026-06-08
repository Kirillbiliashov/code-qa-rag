import ast
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

from config.settings import CHUNKS_FILE
from ingestion.chunks import FileChunk
from ingestion.extractor import SemanticExtractor
from ingestion.repo_scanner import RepositoryScanner


class IngestionService:
    @staticmethod
    def extract_chunks_from_zip(zip_bytes: bytes) -> list[FileChunk]:
        chunks: list[FileChunk] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(tmpdir)

            repo_root = IngestionService._detect_repo_root(tmpdir)
            files = RepositoryScanner.scan(repo_root)

            for rel_path in files:
                full_path = os.path.join(repo_root, rel_path)
                try:
                    source = Path(full_path).read_text(encoding="utf-8")
                    tree = ast.parse(source)
                except (SyntaxError, UnicodeDecodeError, ValueError):
                    continue
                extractor = SemanticExtractor(source, rel_path)
                extractor.visit(tree)
                chunks.extend(extractor.chunks)
        return chunks

    @staticmethod
    def persist_chunks(chunks: list[FileChunk]) -> None:
        with open(CHUNKS_FILE, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_json_dict()) + "\n")

    @staticmethod
    def _detect_repo_root(extract_dir: str) -> str:
        entries = [e for e in os.listdir(extract_dir) if not e.startswith(".")]
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            return os.path.join(extract_dir, entries[0])
        return extract_dir
