from datetime import datetime

from pydantic import BaseModel


class UploadResponse(BaseModel):
    repo_id: str
    chunks: int
    files_processed: int


class AskRequest(BaseModel):
    repo_id: str
    question: str


class QuotaResponse(BaseModel):
    queries_count: int
    quota: int
    quota_reset: datetime


class AskResponse(BaseModel):
    answer: str
    quota: QuotaResponse
