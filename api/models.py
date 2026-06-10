from pydantic import BaseModel


class UploadResponse(BaseModel):
    repo_id: str
    chunks: int
    files_processed: int


class AskRequest(BaseModel):
    repo_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
