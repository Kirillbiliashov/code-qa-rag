from inference.answer_generator import AnswerGenerator
from ingestion.chunks_store import ChunksStore
from retrieval.retriever import Retriever


class QAService:
    def __init__(
        self,
        retriever: Retriever,
        chunks_store: ChunksStore,
        answer_generator: AnswerGenerator,
    ):
        self.retriever = retriever
        self.chunks_store = chunks_store
        self.answer_generator = answer_generator

    def answer(self, question: str) -> str:
        points = self.retriever.retrieve(question, top_k=1)
        if not points:
            return ""

        chunk_id = points[0].payload.get("semantic_id")
        chunk = self.chunks_store.get(chunk_id) if chunk_id else None
        if chunk is None:
            return ""

        code = getattr(chunk, "code", None) or chunk.to_embedding_text()
        return self.answer_generator.generate(
            query=question,
            file_path=chunk.file_path,
            code=code,
        )
