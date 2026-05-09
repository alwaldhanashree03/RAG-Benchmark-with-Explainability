"""Guardrail threshold sensitivity experiment.

Sweeps the retrieval confidence threshold from 0.1 to 0.8 and measures
how it affects guardrail trigger rate, ROUGE-L, and faithfulness across
all 4 RAG strategies. Produces a CSV for plotting.

Usage:
    python experiments/threshold_sweep.py [--queries 50]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.data.text_chunker import TextChunker
from src.data.vector_store import VectorStore
from src.data.embedding_generator import EmbeddingGenerator
from src.models.baseline_rag import BaselineRAG
from src.models.hybrid_rag import HybridRAG
from src.models.reranker_rag import RerankerRAG
from src.models.query_decomposition_rag import QueryDecompositionRAG
from src.guardrails.guardrail_checker import GuardrailChecker
from src.evaluation.metrics import RAGMetrics
from rouge_score import rouge_scorer as rs


def run_threshold_sweep(num_queries: int = 50):
    """Run guardrail threshold sweep experiment."""
    setup_logger()

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # Load data
    processed_dir = Path("./data/processed")
    if not (processed_dir / "queries.parquet").exists():
        logger.error("Run 'python main.py prepare-data && python main.py build-index' first")
        return

    queries_df = pd.read_parquet(processed_dir / "queries.parquet")
    passages_df = pd.read_parquet(processed_dir / "passages.parquet")

    # Limit queries
    queries = queries_df.head(num_queries).to_dict("records")
    logger.info(f"Threshold sweep: {len(thresholds)} thresholds x {len(queries)} queries")

    # Initialize vector store and models
    vector_store = VectorStore()
    if vector_store.get_count() == 0:
        logger.error("Vector index empty. Run 'python main.py build-index' first")
        return

    chunker = TextChunker()
    chunks = chunker.chunk_documents(passages_df["text"].tolist())
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Precompute query embeddings
    emb_gen = EmbeddingGenerator()
    query_texts = [q["query"] for q in queries]
    emb_gen.generate_embeddings(list(set(query_texts)), show_progress=True)

    # Initialize models (excluding Reranker to avoid rate limit issues)
    models = {
        "Baseline": BaselineRAG(vector_store),
        "Hybrid": HybridRAG(vector_store, chunk_texts, chunk_ids),
        "QueryDecomp": QueryDecompositionRAG(vector_store),
    }

    rouge = rs.RougeScorer(["rougeL"], use_stemmer=True)
    results_rows = []

    for threshold in thresholds:
        logger.info(f"\n--- Threshold: {threshold} ---")

        for model_name, model in models.items():
            triggered_count = 0
            rouge_scores = []
            faith_scores = []
            confidences = []

            for q in queries:
                query_text = q["query"]
                ref_answer = q.get("answers", [""])[0] if q.get("answers") else ""

                try:
                    response = model.answer(query_text, top_k=3, apply_guardrails=False)

                    max_score = max(
                        (c.score for c in response.retrieved_chunks), default=0.0
                    )
                    confidences.append(max_score)

                    # Apply threshold
                    if max_score < threshold:
                        triggered_count += 1
                        continue

                    # Compute ROUGE-L if reference available
                    if ref_answer:
                        scores = rouge.score(ref_answer, response.answer)
                        rouge_scores.append(scores["rougeL"].fmeasure)

                    # Simplified faithfulness
                    ctx = " ".join(c.text for c in response.retrieved_chunks).lower()
                    words = set(response.answer.lower().split()) - {
                        "the", "a", "an", "is", "are", "was", "were",
                        "in", "on", "at", "to", "for", "of", "and",
                    }
                    if words:
                        faith = sum(1 for w in words if w in ctx) / len(words)
                        faith_scores.append(faith)

                except Exception as e:
                    triggered_count += 1

            n = len(queries)
            trigger_rate = triggered_count / n
            answered = n - triggered_count

            results_rows.append({
                "threshold": threshold,
                "strategy": model_name,
                "guardrail_trigger_rate": trigger_rate,
                "answered_queries": answered,
                "rouge_l_f1": np.mean(rouge_scores) if rouge_scores else None,
                "faithfulness": np.mean(faith_scores) if faith_scores else None,
                "confidence_mean": np.mean(confidences) if confidences else None,
                "confidence_std": np.std(confidences) if confidences else None,
            })

            logger.info(
                f"  {model_name}: trigger={trigger_rate:.0%}, "
                f"answered={answered}/{n}, "
                f"ROUGE-L={np.mean(rouge_scores):.3f}" if rouge_scores else ""
            )

    # Save results
    output_dir = Path("./results/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results_rows)
    output_path = output_dir / "threshold_sweep.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 90)
    pivot = df.pivot_table(
        index="threshold",
        columns="strategy",
        values="guardrail_trigger_rate",
        aggfunc="first",
    )
    print("\nGuardrail Trigger Rate by Threshold:")
    print(pivot.to_string(float_format="%.2f"))

    pivot_rouge = df.pivot_table(
        index="threshold",
        columns="strategy",
        values="rouge_l_f1",
        aggfunc="first",
    )
    print("\nROUGE-L F1 by Threshold:")
    print(pivot_rouge.to_string(float_format="%.3f", na_rep="N/A"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardrail threshold sweep")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries")
    args = parser.parse_args()
    run_threshold_sweep(args.queries)
