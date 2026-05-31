#!/usr/bin/env python
"""
Evaluation pipeline for RAG QA dataset.
- Load embeddings from chunks_embeddings.jsonl
- Create Qdrant collection and upsert embeddings
- Query top 10 for each question
- Calculate Recall@5, Recall@10, MRR
- Print final metrics
"""

import json
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def load_embeddings(embeddings_file: str) -> list[dict]:
    """Load embeddings from JSONL file."""
    embeddings = []
    with open(embeddings_file, 'r') as f:
        for line in f:
            embeddings.append(json.loads(line))
    print(f"Loaded {len(embeddings)} embeddings from {embeddings_file}")
    return embeddings


def load_qa_dataset(qa_file: str) -> list[dict]:
    """Load QA dataset from JSONL file."""
    qa_pairs = []
    with open(qa_file, 'r') as f:
        for line in f:
            qa_pairs.append(json.loads(line))
    print(f"Loaded {len(qa_pairs)} QA pairs from {qa_file}")
    return qa_pairs


def setup_qdrant(embeddings: list[dict], collection_name: str = "test-qa", url: str = "http://192.168.106.2:6333") -> QdrantClient:
    """Initialize Qdrant client, recreate collection, and upsert embeddings."""
    client = QdrantClient(url=url)
    
    # Get vector size from first embedding
    vector_size = len(embeddings[0]["embedding"])
    
    # Delete collection if exists
    if client.collection_exists(collection_name):
        print(f"Deleting existing collection '{collection_name}'")
        client.delete_collection(collection_name=collection_name)
    
    # Create new collection
    print(f"Creating collection '{collection_name}' with vector size {vector_size}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    
    # Upsert embeddings as points
    points = []
    for emb in embeddings:
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb["embedding"],
                payload={**emb["metadata"], "semantic_id": emb["id"]},
            )
        )
    
    print(f"Upserting {len(points)} points into '{collection_name}'")
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points,
    )
    print(f"Upsert operation: {operation_info}")
    
    return client


def calculate_recall(relevant_set: set, retrieved_ids: list, k: int) -> float:
    """Calculate Recall@k: how many relevant items in top-k results."""
    if not relevant_set:
        return 0.0
    top_k = retrieved_ids[:k]
    matches = len(set(top_k) & relevant_set)
    return matches / len(relevant_set)


def calculate_mrr(relevant_set: set, retrieved_ids: list) -> float:
    """Calculate Mean Reciprocal Rank: 1 / rank of first relevant result."""
    if not relevant_set:
        return 0.0
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_set:
            return 1.0 / rank
    return 0.0


def success_at_k(relevant_set: set, retrieved_ids: list, k: int) -> int:
    """Return 1 if at least one relevant entity is in top-k retrieved_ids, else 0."""
    if not relevant_set:
        return 0
    return int(len(set(retrieved_ids[:k]) & relevant_set) > 0)


def evaluate_qa_dataset(client: QdrantClient, qa_pairs: list[dict], model: SentenceTransformer, collection_name: str = "test-qa"):
    """Evaluate QA dataset: query, retrieve, and compute metrics."""
    
    recalls_5 = []
    recalls_10 = []
    mrrs = []
    success5 = []
    success10 = []
    
    print(f"\nEvaluating {len(qa_pairs)} QA pairs...")
    print("-" * 80)
    
    for idx, qa_pair in enumerate(qa_pairs, start=1):
        question = qa_pair["question"]
        relevant_entities = set(qa_pair["relevant_entities"])
        
        # Encode question
        query_vector = model.encode(question, show_progress_bar=False, normalize_embeddings=True)
        
        # Query top 10
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=10,
            with_payload=True,
        )
        
        # Extract semantic IDs from results
        retrieved_ids = [point.payload.get("semantic_id") for point in response.points]
        
        # Calculate metrics
        recall_5 = calculate_recall(relevant_entities, retrieved_ids, k=5)
        recall_10 = calculate_recall(relevant_entities, retrieved_ids, k=10)
        mrr = calculate_mrr(relevant_entities, retrieved_ids)
        succ5 = success_at_k(relevant_entities, retrieved_ids, k=5)
        succ10 = success_at_k(relevant_entities, retrieved_ids, k=10)
        
        recalls_5.append(recall_5)
        recalls_10.append(recall_10)
        mrrs.append(mrr)
        success5.append(succ5)
        success10.append(succ10)
        
        # Print per-question metrics
        print(f"Q{idx}: {question[:60]}...")
        print(f"  Relevant: {relevant_entities}")
        print(f"  Retrieved (top 10): {retrieved_ids}")
        print(f"  Recall@5={recall_5:.3f}, Recall@10={recall_10:.3f}, MRR={mrr:.3f}, Success@5={succ5}, Success@10={succ10}")
        print()
    
    # Calculate averages
    avg_recall_5 = sum(recalls_5) / len(recalls_5) if recalls_5 else 0.0
    avg_recall_10 = sum(recalls_10) / len(recalls_10) if recalls_10 else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
    avg_success5 = sum(success5) / len(success5) if success5 else 0.0
    avg_success10 = sum(success10) / len(success10) if success10 else 0.0
    
    return {
        "avg_recall_5": avg_recall_5,
        "avg_recall_10": avg_recall_10,
        "avg_mrr": avg_mrr,
        "avg_success_5": avg_success5,
        "avg_success_10": avg_success10,
        "total_questions": len(qa_pairs),
    }


def main():
    # Paths
    eval_dir = Path(__file__).parent
    embeddings_file = eval_dir / "chunks_embeddings.jsonl"
    qa_file = eval_dir / "qa-dataset.jsonl"
    
    # Load data
    embeddings = load_embeddings(str(embeddings_file))
    qa_pairs = load_qa_dataset(str(qa_file))
    
    # Setup Qdrant
    client = setup_qdrant(embeddings, collection_name="test-qa")
    
    # Load model
    model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
    print(f"Loaded model: flax-sentence-embeddings/st-codesearch-distilroberta-base")
    
    # Evaluate
    results = evaluate_qa_dataset(client, qa_pairs, model, collection_name="test-qa")
    
    # Print final results
    print("\n" + "=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Questions: {results['total_questions']}")
    print(f"Avg Recall@5:  {results['avg_recall_5']:.4f}")
    print(f"Avg Recall@10: {results['avg_recall_10']:.4f}")
    print(f"Avg MRR:       {results['avg_mrr']:.4f}")
    print(f"Avg Success@5:  {results['avg_success_5']:.4f}")
    print(f"Avg Success@10: {results['avg_success_10']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
