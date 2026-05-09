"""Failure case analysis: find examples of correct refusals, incorrect refusals,
and unfaithful answers.

Produces a markdown report with concrete query examples that demonstrate
guardrail behavior.

Usage:
    python experiments/failure_analysis.py [--queries 30]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.data.vector_store import VectorStore
from src.data.embedding_generator import EmbeddingGenerator
from src.models.hybrid_rag import HybridRAG
from src.data.text_chunker import TextChunker
from rouge_score import rouge_scorer as rs


def analyze_failures(num_queries: int = 30):
    """Analyze guardrail behavior: correct refusals, wrong refusals, unfaithful answers."""
    setup_logger()

    processed_dir = Path("./data/processed")
    queries_df = pd.read_parquet(processed_dir / "queries.parquet")
    passages_df = pd.read_parquet(processed_dir / "passages.parquet")

    queries = queries_df.head(num_queries).to_dict("records")

    vector_store = VectorStore()
    chunker = TextChunker()
    chunks = chunker.chunk_documents(passages_df["text"].tolist())
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Precompute embeddings
    emb_gen = EmbeddingGenerator()
    emb_gen.generate_embeddings(list(set(q["query"] for q in queries)), show_progress=False)

    model = HybridRAG(vector_store, chunk_texts, chunk_ids)
    rouge = rs.RougeScorer(["rougeL"], use_stemmer=True)

    cases = {
        "correct_refusal": [],      # Low confidence + no relevant content
        "incorrect_refusal": [],     # Low confidence but answer would be good
        "faithful_answer": [],       # Answered well
        "unfaithful_answer": [],     # Answered but poorly grounded
    }

    threshold = 0.3  # Current tuned threshold

    for q in queries:
        query_text = q["query"]
        ref_answer = q.get("answers", [""])[0] if q.get("answers") else ""

        try:
            response = model.answer(query_text, top_k=3, apply_guardrails=False)
            max_score = max((c.score for c in response.retrieved_chunks), default=0.0)
            triggered = max_score < threshold

            # Compute quality metrics
            rouge_f1 = 0.0
            faith = 0.0
            if ref_answer:
                scores = rouge.score(ref_answer, response.answer)
                rouge_f1 = scores["rougeL"].fmeasure

            ctx = " ".join(c.text for c in response.retrieved_chunks).lower()
            words = set(response.answer.lower().split()) - {
                "the", "a", "an", "is", "are", "was", "were",
                "in", "on", "at", "to", "for", "of", "and", "that", "this",
            }
            if words:
                faith = sum(1 for w in words if w in ctx) / len(words)

            case = {
                "query": query_text,
                "reference": ref_answer,
                "answer": response.answer[:300],
                "confidence": max_score,
                "rouge_l_f1": rouge_f1,
                "faithfulness": faith,
                "triggered": triggered,
                "top_chunk": response.retrieved_chunks[0].text[:200] if response.retrieved_chunks else "",
                "top_score": response.retrieved_chunks[0].score if response.retrieved_chunks else 0,
            }

            if triggered and rouge_f1 < 0.1:
                cases["correct_refusal"].append(case)
            elif triggered and rouge_f1 >= 0.1:
                cases["incorrect_refusal"].append(case)
            elif not triggered and faith >= 0.4:
                cases["faithful_answer"].append(case)
            elif not triggered and faith < 0.4:
                cases["unfaithful_answer"].append(case)

        except Exception as e:
            pass

    # Generate markdown report
    output_dir = Path("./results/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Failure Case Analysis",
        "",
        f"Strategy: Hybrid (BM25 + Semantic) | Threshold: {threshold} | Queries: {num_queries}",
        "",
    ]

    for category, label, description in [
        ("correct_refusal", "Correct Refusals",
         "Guardrail correctly refused - low confidence AND answer would be poor"),
        ("incorrect_refusal", "Incorrect Refusals (False Positives)",
         "Guardrail refused but the system could have given a reasonable answer"),
        ("faithful_answer", "Faithful Answers",
         "Guardrail passed and the answer is well-grounded in context"),
        ("unfaithful_answer", "Unfaithful Answers (False Negatives)",
         "Guardrail passed but the answer is poorly grounded"),
    ]:
        items = cases[category][:3]  # Show top 3 per category
        report_lines.append(f"## {label}")
        report_lines.append(f"_{description}_")
        report_lines.append(f"Count: {len(cases[category])}/{num_queries}")
        report_lines.append("")

        for i, case in enumerate(items, 1):
            report_lines.append(f"### Example {i}")
            report_lines.append(f"**Query**: {case['query']}")
            report_lines.append(f"**Reference**: {case['reference']}")
            report_lines.append(f"**Generated**: {case['answer']}")
            report_lines.append(f"**Confidence**: {case['confidence']:.4f} | "
                              f"**ROUGE-L**: {case['rouge_l_f1']:.3f} | "
                              f"**Faithfulness**: {case['faithfulness']:.3f}")
            report_lines.append(f"**Top Retrieved Chunk** (score={case['top_score']:.4f}):")
            report_lines.append(f"> {case['top_chunk']}")
            report_lines.append("")

    # Summary
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("| Category | Count | Rate |")
    report_lines.append("|----------|-------|------|")
    total = num_queries
    for cat_key, cat_label in [
        ("correct_refusal", "Correct Refusals"),
        ("incorrect_refusal", "Incorrect Refusals"),
        ("faithful_answer", "Faithful Answers"),
        ("unfaithful_answer", "Unfaithful Answers"),
    ]:
        n = len(cases[cat_key])
        report_lines.append(f"| {cat_label} | {n} | {n/total:.0%} |")

    report_text = "\n".join(report_lines)

    report_path = output_dir / "failure_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=30)
    args = parser.parse_args()
    analyze_failures(args.queries)
