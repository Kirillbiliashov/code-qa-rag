from llama_index.core.llms import LLM

from qa.prompts import QA_TEMPLATE, REDUCE_TEMPLATE


class AnswerGenerator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, query: str, file_path: str, code: str) -> str:
        messages = QA_TEMPLATE.format_messages(
            query=query, file_path=file_path, code=code
        )
        response = self.llm.chat(messages)
        return response.message.content or ""

    def reduce(self, query: str, answers: list[str]) -> str:
        joined = ("\n" + "-" * 30 + "\n").join(answers)
        messages = REDUCE_TEMPLATE.format_messages(query=query, answers=joined)
        response = self.llm.chat(messages)
        return response.message.content or ""
