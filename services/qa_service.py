import asyncio

from db.database import Database
from inference.answer_generator import AnswerGenerator
from retrieval.retriever import Retriever


TOP_K = 3


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

    async def answer(self, repo_id: str, question: str) -> str:
        points = await self.retriever.retrieve(question, repo_id=repo_id, top_k=TOP_K)
        if not points:
            return ""

        chunk_ids = [
            p.payload.get("semantic_id")
            for p in points
            if p.payload.get("semantic_id")
        ]
        docs = self.database.get_chunks(repo_id, chunk_ids)
        if not docs:
            return ""

        partial_responses = await asyncio.gather(*(
            self.answer_generator.generate(
                query=question,
                file_path=doc["file_path"],
                code=doc["code"],
            )
            for doc in docs
        ))
        partial_responses = [p for p in partial_responses if p.strip()]
        if not partial_responses:
            return ""

        return await self.answer_generator.reduce(query=question, answers=partial_responses)
