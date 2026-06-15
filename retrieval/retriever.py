from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Fusion, FusionQuery, MatchValue, Prefetch, SparseVector
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding


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
        sparse_embedding_model: SparseTextEmbedding,
        qdrant_client: QdrantClient,
        collection_name: str,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

    async def _classify_query(self, query: str) -> str | None:
        prompt = CLASSIFY_USER_INPUT_PROMPT.format(question=query)
        response = await self.llm.acomplete(prompt)
        chunk_type = response.text.strip().lower()
        if chunk_type not in VALID_CHUNK_TYPES:
            return None
        return chunk_type

    async def retrieve(self, query: str, repo_id: str | None = None, top_k: int = 5):
        chunk_type = await self._classify_query(query)
        print(f"query type: {chunk_type}")


        must_conditions: list[FieldCondition] = []
        if chunk_type is not None:
            must_conditions.append(
                FieldCondition(key="type", match=MatchValue(value=chunk_type))
            )
        if repo_id is not None:
            must_conditions.append(
                FieldCondition(key="repo_id", match=MatchValue(value=repo_id))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None
        
        sparse_emb = next(self.sparse_embedding_model.embed(query))
        sparse_query = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )
        dense_query = self.embedding_model.encode(
            query, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
        prefetch = [
            Prefetch(
                query=dense_query,
                using="dense",
                limit=10,
            ),
            Prefetch(
                query=sparse_query,
                using="bm25",
                limit=10,
            ),
        ]

        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=FusionQuery(fusion=Fusion.RRF),
            prefetch=prefetch,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return response.points
