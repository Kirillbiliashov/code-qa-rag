import os

from dotenv import load_dotenv
load_dotenv()


MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
EMBEDDINGS_FILE = "evaluation/chunks_embeddings.jsonl"
QA_DATASET_FILE = "evaluation/qa-dataset.jsonl"
CHUNKS_FILE = "chunks.jsonl"
COLLECTION_NAME = "code-vectors"
VECTOR_SIZE = 768

QDRANT_URL = "http://192.168.106.2:6333"

MONGO_URL = os.getenv("MONGO_DB_CONN_STRING")
MONGO_DB_NAME = "code-qa-rag"

LLM_MODEL = "deepseek-v4-flash"
LLM_REQUEST_TIMEOUT = 300.0
LLM_TEMPERATURE = 0.1
LLM_CONTEXT_WINDOW = 1024


LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DAILY_QUERY_QUOTA = 5
QUOTA_WINDOW_HOURS = 24