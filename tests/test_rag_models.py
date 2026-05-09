"""Comprehensive tests for RAG models.

Industry best practice: High test coverage for production systems.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.models.base_rag import BaseRAG, RetrievedChunk, RAGResponse
from src.models.baseline_rag import BaselineRAG
from src.models.hyde_rag import HyDERAG
from src.models.self_rag import SelfRAG
from src.models.multi_query_rag import MultiQueryRAG


class TestBaseRAG:
    """Test base RAG functionality."""
    
    def test_retrieved_chunk_creation(self):
        """Test RetrievedChunk dataclass."""
        chunk = RetrievedChunk(
            chunk_id="test_1",
            text="This is test text",
            score=0.85,
            metadata={"source": "test"},
            rank=1
        )
        
        assert chunk.chunk_id == "test_1"
        assert chunk.score == 0.85
        assert chunk.rank == 1
    
    def test_rag_response_creation(self):
        """Test RAGResponse dataclass."""
        chunks = [
            RetrievedChunk("c1", "text1", 0.9, {}, 1),
            RetrievedChunk("c2", "text2", 0.8, {}, 2),
        ]
        
        response = RAGResponse(
            query="test query",
            answer="test answer",
            retrieved_chunks=chunks,
            confidence_score=0.85,
            guardrail_triggered=False,
            guardrail_reason=None,
            metadata={"config": "test"}
        )
        
        assert response.query == "test query"
        assert len(response.retrieved_chunks) == 2
        assert response.confidence_score == 0.85
        assert not response.guardrail_triggered


class TestBaselineRAG:
    """Test baseline RAG implementation."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock()
        mock_store.search = Mock(return_value=(
            ["chunk_1", "chunk_2", "chunk_3"],
            [0.9, 0.8, 0.7],
            ["Text 1", "Text 2", "Text 3"],
            [{}, {}, {}]
        ))
        return mock_store
    
    @pytest.fixture
    def baseline_rag(self, mock_vector_store):
        """Create baseline RAG instance."""
        with patch('src.models.baseline_rag.EmbeddingGenerator'):
            with patch('src.models.baseline_rag.LLMClient'):
                rag = BaselineRAG(mock_vector_store)
                return rag
    
    def test_retrieve(self, baseline_rag, mock_vector_store):
        """Test retrieval functionality."""
        # Mock embedding generation
        baseline_rag.embedding_generator.generate_embedding = Mock(
            return_value=np.array([0.1] * 1536)
        )
        
        chunks = baseline_rag.retrieve("test query", top_k=3)
        
        assert len(chunks) == 3
        assert chunks[0].chunk_id == "chunk_1"
        assert chunks[0].score == 0.9
        assert chunks[0].rank == 1
        
        # Verify search was called
        mock_vector_store.search.assert_called_once()
    
    def test_generate(self, baseline_rag):
        """Test answer generation."""
        chunks = [
            RetrievedChunk("c1", "Context text 1", 0.9, {}, 1),
            RetrievedChunk("c2", "Context text 2", 0.8, {}, 2),
        ]
        
        baseline_rag.llm_client.generate_rag_answer = Mock(
            return_value="Generated answer"
        )
        
        answer = baseline_rag.generate("test query", chunks)
        
        assert answer == "Generated answer"
        baseline_rag.llm_client.generate_rag_answer.assert_called_once()


class TestHyDERAG:
    """Test HyDE RAG implementation."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock()
        mock_store.search = Mock(return_value=(
            ["chunk_1"],
            [0.85],
            ["Retrieved text"],
            [{}]
        ))
        return mock_store
    
    @pytest.fixture
    def hyde_rag(self, mock_vector_store):
        """Create HyDE RAG instance."""
        with patch('src.models.hyde_rag.EmbeddingGenerator'):
            with patch('src.models.hyde_rag.LLMClient'):
                rag = HyDERAG(mock_vector_store)
                return rag
    
    def test_hypothetical_document_generation(self, hyde_rag):
        """Test hypothetical document generation."""
        hyde_rag.llm_client.generate = Mock(
            return_value="This is a hypothetical document about AI."
        )
        
        hyde_doc = hyde_rag._generate_hypothetical_document("What is AI?")
        
        assert "hypothetical" in hyde_doc.lower() or "AI" in hyde_doc
        hyde_rag.llm_client.generate.assert_called_once()
    
    def test_hyde_retrieve(self, hyde_rag, mock_vector_store):
        """Test HyDE retrieval process."""
        # Mock hypothetical doc generation
        hyde_rag._generate_hypothetical_document = Mock(
            return_value="Hypothetical answer"
        )
        
        # Mock embedding
        hyde_rag.embedding_generator.generate_embedding = Mock(
            return_value=np.array([0.1] * 1536)
        )
        
        chunks = hyde_rag.retrieve("What is AI?", top_k=1)
        
        assert len(chunks) == 1
        hyde_rag._generate_hypothetical_document.assert_called_once()


class TestSelfRAG:
    """Test Self-RAG implementation."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock()
        mock_store.search = Mock(return_value=(
            ["chunk_1"],
            [0.9],
            ["High quality context"],
            [{}]
        ))
        return mock_store
    
    @pytest.fixture
    def self_rag(self, mock_vector_store):
        """Create Self-RAG instance."""
        with patch('src.models.self_rag.EmbeddingGenerator'):
            with patch('src.models.self_rag.LLMClient'):
                rag = SelfRAG(mock_vector_store, max_iterations=2)
                return rag
    
    def test_retrieval_decision(self, self_rag):
        """Test retrieval decision making."""
        from src.models.self_rag import RetrievalDecision
        
        # Mock LLM response for retrieval decision
        self_rag.llm_client.generate = Mock(return_value="RETRIEVE")
        
        decision = self_rag._assess_retrieval_need("What is the capital of France?")
        
        assert decision == RetrievalDecision.RETRIEVE
    
    def test_relevance_assessment(self, self_rag):
        """Test relevance assessment."""
        from src.models.self_rag import RelevanceScore
        
        chunks = [
            RetrievedChunk("c1", "Relevant text", 0.95, {}, 1),
        ]
        
        relevance, filtered = self_rag._assess_relevance("query", chunks)
        
        assert relevance == RelevanceScore.RELEVANT
        assert len(filtered) == 1


class TestMultiQueryRAG:
    """Test Multi-Query RAG implementation."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        mock_store = Mock()
        mock_store.search = Mock(return_value=(
            ["chunk_1", "chunk_2"],
            [0.9, 0.8],
            ["Text 1", "Text 2"],
            [{}, {}]
        ))
        return mock_store
    
    @pytest.fixture
    def multi_query_rag(self, mock_vector_store):
        """Create Multi-Query RAG instance."""
        with patch('src.models.multi_query_rag.EmbeddingGenerator'):
            with patch('src.models.multi_query_rag.LLMClient'):
                rag = MultiQueryRAG(mock_vector_store, num_queries=3)
                return rag
    
    def test_query_variations_generation(self, multi_query_rag):
        """Test query variation generation."""
        multi_query_rag.llm_client.generate = Mock(
            return_value="1. What is artificial intelligence?\n2. Define AI\n3. Explain AI concepts"
        )
        
        variations = multi_query_rag._generate_query_variations("What is AI?")
        
        assert len(variations) >= 1
        assert "What is AI?" in variations  # Original should be included
    
    def test_multi_query_retrieve(self, multi_query_rag):
        """Test multi-query retrieval."""
        # Mock query variations
        multi_query_rag._generate_query_variations = Mock(
            return_value=["What is AI?", "Define AI", "Explain AI"]
        )
        
        # Mock embedding
        multi_query_rag.embedding_generator.generate_embedding = Mock(
            return_value=np.array([0.1] * 1536)
        )
        
        chunks = multi_query_rag.retrieve("What is AI?", top_k=2)
        
        assert len(chunks) <= 2
        # Should have called search multiple times (once per variation)
        assert multi_query_rag.vector_store.search.call_count >= 1


class TestIntegration:
    """Integration tests for RAG pipeline."""
    
    def test_end_to_end_mock_pipeline(self):
        """Test complete RAG pipeline with mocks."""
        # Create mock components
        mock_store = Mock()
        mock_store.search = Mock(return_value=(
            ["c1"],
            [0.9],
            ["AI is artificial intelligence"],
            [{}]
        ))
        
        with patch('src.models.baseline_rag.EmbeddingGenerator'):
            with patch('src.models.baseline_rag.LLMClient'):
                rag = BaselineRAG(mock_store)
                
                # Mock methods
                rag.embedding_generator.generate_embedding = Mock(
                    return_value=np.array([0.1] * 1536)
                )
                rag.llm_client.generate_rag_answer = Mock(
                    return_value="AI stands for Artificial Intelligence."
                )
                
                # Run pipeline
                response = rag.answer("What is AI?", top_k=1)
                
                assert isinstance(response, RAGResponse)
                assert response.query == "What is AI?"
                assert response.answer == "AI stands for Artificial Intelligence."
                assert len(response.retrieved_chunks) == 1
                assert response.confidence_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
