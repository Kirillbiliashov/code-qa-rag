class QAService:
    def __init__(self, container):
        self.container = container

    def answer(self, question: str) -> str:
        # TODO: wire up retrieval + LLM call using self.container
        return "ok"
