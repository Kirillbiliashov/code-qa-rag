from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


CLASSIFY_USER_INPUT_PROMPT = PromptTemplate("""
You will be given user question related to the python codebase.
Your task is to classify this question accoriding to its scope.
There are 3 scopes, each subsequent scope is more specific and narrower than the previous one:
'module' - user asks question about a whole file, interactions between different functions and classes in the file, or general question about the file;
'class' - user asks question about a specific class and its methods, interactions between methods, or general question about the class;
'function' - the question is specifically about a single function or method, its implementation details, or its interactions with other functions or classes.

In the response, return ONLY one of the selected scopes.

Question:
'{question}'
""")


VALID_CHUNK_TYPES = {"module", "class", "function"}


class Retriever:
    def __init__(
        self,
        llm: LLM,
        embedding_model: SentenceTransformer,
        qdrant_client: QdrantClient,
        collection_name: str,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

    def _classify_query(self, query: str) -> str | None:
        prompt = CLASSIFY_USER_INPUT_PROMPT.format(question=query)
        response = self.llm.complete(prompt)
        chunk_type = response.text.strip().lower()
        if chunk_type not in VALID_CHUNK_TYPES:
            return None
        return chunk_type

    def retrieve(self, query: str, top_k: int = 5):
        chunk_type = self._classify_query(query)
        print(f"query type: {chunk_type}")

        query_vector = self.embedding_model.encode(
            query, show_progress_bar=False, normalize_embeddings=True
        )

        query_filter = None
        if chunk_type is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=chunk_type),
                    )
                ]
            )

        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return response.points
