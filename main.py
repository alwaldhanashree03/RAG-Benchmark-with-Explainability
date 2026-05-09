"""Main entry point for RAG benchmark system.

Usage:
    python main.py prepare-data     # Download and prepare dataset
    python main.py build-index      # Build vector index
    python main.py benchmark        # Run full benchmark
    python main.py ui               # Launch Streamlit UI
"""

import argparse
from pathlib import Path

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.data.data_loader import DatasetLoader
from src.data.text_chunker import TextChunker
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.baseline_rag import BaselineRAG
from src.models.hybrid_rag import HybridRAG
from src.models.reranker_rag import RerankerRAG
from src.models.query_decomposition_rag import QueryDecompositionRAG
from src.evaluation.benchmark import RAGBenchmark
from src.utils.cost_tracker import get_cost_tracker

from loguru import logger


def prepare_data():
    """Download and prepare dataset."""
    logger.info("Preparing dataset...")
    
    loader = DatasetLoader()
    queries_df, passages_df = loader.load()
    
    # Save processed data
    loader.save_processed(queries_df, passages_df)
    
    logger.info(f"Dataset prepared: {len(queries_df)} queries, {len(passages_df)} passages")


def build_index():
    """Build vector index from processed data."""
    logger.info("Building vector index...")
    
    # Load data
    processed_dir = Path("./data/processed")
    if not (processed_dir / "passages.parquet").exists():
        logger.error("Processed data not found. Run 'prepare-data' first.")
        return
    
    import pandas as pd
    passages_df = pd.read_parquet(processed_dir / "passages.parquet")
    
    # Chunk documents
    chunker = TextChunker()
    chunks = chunker.chunk_documents(passages_df["text"].tolist())
    
    stats = chunker.get_stats(chunks)
    logger.info(f"Chunking stats: {stats}")
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator()
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    
    embeddings = embedding_gen.generate_embeddings(chunk_texts)
    
    # Build vector store
    vector_store = VectorStore()
    vector_store.reset()  # Clear existing data
    
    vector_store.add_documents(
        ids=chunk_ids,
        embeddings=embeddings,
        texts=chunk_texts,
    )
    
    logger.info(f"Vector index built: {vector_store.get_count()} documents")


def run_benchmark(query_limit: int = None):
    """Run full benchmark across all RAG configurations.

    Args:
        query_limit: If set, only benchmark on the first N queries.
                     Useful for quick debug runs (e.g. --limit 20).
    """
    logger.info("Starting benchmark...")

    # Load data
    processed_dir = Path("./data/processed")
    if not (processed_dir / "queries.parquet").exists():
        logger.error("Processed data not found. Run 'prepare-data' first.")
        return

    import pandas as pd
    queries_df = pd.read_parquet(processed_dir / "queries.parquet")
    passages_df = pd.read_parquet(processed_dir / "passages.parquet")

    # Initialize vector store
    vector_store = VectorStore()

    if vector_store.get_count() == 0:
        logger.error("Vector index not found. Run 'build-index' first.")
        return

    # Prepare chunks for hybrid search
    chunker = TextChunker()
    chunks = chunker.chunk_documents(passages_df["text"].tolist())
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Initialize RAG models
    config = get_config()
    rag_models = {}

    if config.get("rag_configs.baseline.enabled", True):
        rag_models["Baseline"] = BaselineRAG(vector_store)

    if config.get("rag_configs.hybrid.enabled", True):
        rag_models["Hybrid"] = HybridRAG(vector_store, chunk_texts, chunk_ids)

    if config.get("rag_configs.reranker.enabled", True):
        rag_models["Reranker"] = RerankerRAG(vector_store)

    if config.get("rag_configs.query_decomposition.enabled", True):
        rag_models["Query Decomposition"] = QueryDecompositionRAG(vector_store)

    logger.info(f"Initialized {len(rag_models)} RAG configurations")

    # Prepare queries for benchmark
    queries = queries_df.to_dict('records')

    # Apply query limit for debug runs
    if query_limit and query_limit < len(queries):
        logger.info(f"Debug mode: limiting to {query_limit} queries (out of {len(queries)})")
        queries = queries[:query_limit]

    # Run benchmark
    benchmark = RAGBenchmark()
    results = benchmark.run_benchmark(
        rag_configs=rag_models,
        queries=queries,
        apply_guardrails=True,
        output_dir=Path("./results"),
    )

    # Print summary
    logger.info("=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)

    for config_name, metrics in results["summary"]["aggregated_metrics"].items():
        logger.info(f"\n{config_name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")

    # Print cost summary
    cost_summary = results["cost_summary"]
    logger.info(f"\nTotal Cost: ${cost_summary['total_cost_usd']:.4f}")
    logger.info(f"Remaining Budget: ${cost_summary['remaining_usd']:.2f}")

    logger.info("\nResults saved to ./results/")


def launch_ui():
    """Launch Streamlit UI."""
    import subprocess
    import sys
    
    logger.info("Launching Streamlit UI...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/app.py"])


def main():
    """Main entry point."""
    setup_logger()
    
    parser = argparse.ArgumentParser(description="RAG Benchmark System")
    parser.add_argument(
        "command",
        choices=["prepare-data", "build-index", "benchmark", "ui", "quick-test"],
        help="Command to execute"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data for quick testing (prepare-data command)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries for benchmark (e.g. --limit 20 for debug)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "prepare-data":
            if args.sample:
                logger.info("Using sample data for quick testing...")
                from src.data.sample_data import SampleDataGenerator
                generator = SampleDataGenerator()
                generator.generate()
                logger.info("[SUCCESS] Sample data generated! Run 'build-index' next.")
            else:
                prepare_data()
        elif args.command == "build-index":
            build_index()
        elif args.command == "benchmark":
            run_benchmark(query_limit=args.limit)
        elif args.command == "ui":
            launch_ui()
        elif args.command == "quick-test":
            import subprocess
            import sys
            logger.info("Running quick test...")
            subprocess.run([sys.executable, "quick_test.py"])
    
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        raise


if __name__ == "__main__":
    main()
