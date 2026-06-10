from db.database import Database
from inference.answer_generator import AnswerGenerator
from retrieval.retriever import Retriever


class QAService:
    def __init__(
        self,
        retriever: Retriever,
        database: Database,
        answer_generator: AnswerGenerator,
    ):
        self.retriever = retriever
        self.database = database
        self.answer_generator = answer_generator

    def answer(self, repo_id: str, question: str) -> str:
        points = self.retriever.retrieve(question, repo_id=repo_id, top_k=1)
        if not points:
            return ""

        chunk_id = points[0].payload.get("semantic_id")
        if not chunk_id:
            return ""

        doc = self.database.get_chunk(repo_id, chunk_id)
        if doc is None:
            return ""

        code = doc.get("code") or doc.get("retrieval_text", "")
        return self.answer_generator.generate(
            query=question,
            file_path=doc["file_path"],
            code=code,
        )
