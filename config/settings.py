MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
EMBEDDINGS_FILE = "evaluation/chunks_embeddings.jsonl"
CHUNKS_FILE = "chunks.jsonl"
COLLECTION_NAME = "code-vectors"
VECTOR_SIZE = 768

QDRANT_URL = "http://192.168.106.2:6333"

LLM_MODEL = "phi3"
LLM_REQUEST_TIMEOUT = 300.0
LLM_TEMPERATURE = 0.1
LLM_CONTEXT_WINDOW = 1099
