from pydantic import BaseModel


class UploadResponse(BaseModel):
    chunks: int
    files_processed: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
