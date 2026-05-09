"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import RAGMetrics
from src.models.base_rag import RetrievedChunk


def test_precision_recall_at_k():
    """Test precision and recall calculation."""
    metrics = RAGMetrics()
    
    retrieved_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant_ids = ["doc2", "doc3", "doc6"]
    
    # Test precision@3
    precision_3, recall_3 = metrics._precision_recall_at_k(retrieved_ids, relevant_ids, 3)
    
    # 2 out of 3 retrieved are relevant
    assert precision_3 == pytest.approx(2/3, rel=1e-3)
    
    # 2 out of 3 relevant were retrieved
    assert recall_3 == pytest.approx(2/3, rel=1e-3)


def test_mrr():
    """Test Mean Reciprocal Rank calculation."""
    metrics = RAGMetrics()
    
    # First relevant at position 2 (rank 2)
    retrieved_ids = ["doc1", "doc2", "doc3"]
    relevant_ids = ["doc2"]
    
    mrr = metrics._mean_reciprocal_rank(retrieved_ids, relevant_ids)
    assert mrr == pytest.approx(1/2, rel=1e-3)
    
    # No relevant documents
    retrieved_ids = ["doc1", "doc2", "doc3"]
    relevant_ids = ["doc4"]
    
    mrr = metrics._mean_reciprocal_rank(retrieved_ids, relevant_ids)
    assert mrr == 0.0


def test_rouge_l():
    """Test ROUGE-L evaluation."""
    metrics = RAGMetrics()
    
    generated = "The quick brown fox jumps over the lazy dog"
    reference = "The quick brown fox jumps over a lazy dog"
    
    scores = metrics.evaluate_generation(generated, reference)
    
    assert "rouge_l_f1" in scores
    assert 0.0 <= scores["rouge_l_f1"] <= 1.0
