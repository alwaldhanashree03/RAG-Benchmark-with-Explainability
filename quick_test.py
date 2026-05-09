"""Quick test script to verify RAG system with sample data.

Run this before full benchmark to verify installation and API keys work.
Cost: ~$0.01
"""

import sys
from pathlib import Path
from loguru import logger

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.data.sample_data import SampleDataGenerator
from src.data.text_chunker import TextChunker
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.baseline_rag import BaselineRAG
from src.guardrails.guardrail_checker import GuardrailChecker


def main():
    """Run quick test with sample data."""
    print("=" * 70)
    print("RAG System Quick Test")
    print("=" * 70)
    print()

    setup_logger()

    # Step 1: Generate sample data
    print("Step 1: Generating sample data...")
    generator = SampleDataGenerator()
    generator.generate()
    sample_data = generator.load_sample_data()
    passages = [p["text"] for p in sample_data["passages"]]
    print(f"  Loaded {len(sample_data['queries'])} queries and {len(passages)} passages\n")

    # Step 2: Chunk documents
    print("Step 2: Chunking documents...")
    chunker = TextChunker()
    chunks = chunker.chunk_documents(passages)
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    print(f"  Created {len(chunks)} chunks\n")

    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings (~$0.001)...")
    try:
        emb_gen = EmbeddingGenerator()
        embeddings = emb_gen.generate_embeddings(chunk_texts, show_progress=False)
        print(f"  Generated {len(embeddings)} embeddings\n")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Make sure OPENAI_API_KEY is set in .env file")
        return 1

    # Step 4: Create vector store
    print("Step 4: Creating vector store...")
    try:
        vector_store = VectorStore(collection_name="quick_test")
        vector_store.reset()
        vector_store.add_documents(ids=chunk_ids, embeddings=embeddings, texts=chunk_texts)
        print(f"  Vector store: {vector_store.get_count()} documents\n")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 1

    # Step 5: Test baseline RAG
    print("Step 5: Testing Baseline RAG...")
    try:
        rag = BaselineRAG(vector_store)
        test_query = sample_data["queries"][0]["query"]
        print(f"  Query: {test_query}")

        response = rag.answer(test_query, top_k=3, apply_guardrails=False)
        print(f"  Answer: {response.answer[:150]}...")
        print(f"  Confidence: {response.confidence_score:.3f}")
        print(f"  Chunks retrieved: {len(response.retrieved_chunks)}\n")
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 6: Test guardrails
    print("Step 6: Testing guardrails...")
    try:
        guardrails = GuardrailChecker()
        triggered, reason, details = guardrails.check_guardrails(
            query=test_query,
            chunks=response.retrieved_chunks,
            answer=response.answer,
        )
        print(f"  Triggered: {triggered}")
        print(f"  Confidence level: {details.get('confidence_level', 'unknown')}")
        print()
    except Exception as e:
        print(f"  WARNING: Guardrail check failed: {e}")
        print("  Continuing...\n")

    print("=" * 70)
    print("Quick test PASSED")
    print("=" * 70)
    print("\nNext steps:")
    print("  python main.py prepare-data     # Download MS MARCO")
    print("  python main.py build-index      # Build vector index")
    print("  python main.py benchmark        # Run benchmark")
    print("  python main.py ui               # Launch Streamlit UI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
