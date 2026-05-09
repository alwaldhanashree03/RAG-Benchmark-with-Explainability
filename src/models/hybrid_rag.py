"""Hybrid RAG: BM25 + Semantic search combination.

Industry best practice: Combine lexical (BM25) and semantic search.
Reference: Hybrid search strategies in BEIR benchmark
"""

from typing import List

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src.models.base_rag import BaseRAG, RetrievedChunk
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.llm_client import LLMClient
from src.utils.config_loader import get_config


class HybridRAG(BaseRAG):
    """Hybrid RAG: Combine BM25 (lexical) and semantic search.
    
    Implementation:
    1. Retrieve candidates using BM25 (lexical matching)
    2. Retrieve candidates using semantic search
    3. Combine and rerank using weighted scores
    4. Generate answer from top-k results
    
    Best practice: Hybrid search often outperforms pure semantic search
    Reference: BEIR benchmark (Thakur et al., 2021)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        corpus_texts: List[str],
        corpus_ids: List[str],
    ):
        """Initialize hybrid RAG.
        
        Args:
            vector_store: Vector store with indexed documents
            corpus_texts: Full corpus texts for BM25
            corpus_ids: Corresponding corpus IDs
        """
        super().__init__("Hybrid Search (BM25 + Semantic)")
        
        self.config = get_config()
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()
        
        # BM25 configuration
        self.bm25_weight = self.config.get("rag_configs.hybrid.bm25_weight", 0.5)
        self.semantic_weight = self.config.get("rag_configs.hybrid.semantic_weight", 0.5)
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        
        logger.info(
            f"HybridRAG initialized: BM25 weight={self.bm25_weight}, "
            f"Semantic weight={self.semantic_weight}, Corpus size={len(corpus_texts)}"
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve chunks using hybrid search.
        
        Args:
            query: Query text
            top_k: Number of final chunks to return
            
        Returns:
            List of retrieved chunks with combined scores
            
        Best practice: Retrieve more candidates (10x) then rerank to top-k
        """
        # Retrieve more candidates for reranking
        candidate_k = self.config.get("rag_configs.hybrid.top_k", 10)
        
        # 1. BM25 retrieval
        bm25_scores = self._bm25_search(query, candidate_k)
        
        # 2. Semantic retrieval
        semantic_scores = self._semantic_search(query, candidate_k)
        
        # 3. Combine scores
        combined_scores = self._combine_scores(bm25_scores, semantic_scores)
        
        # 4. Get top-k
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        # Create RetrievedChunk objects
        chunks = []
        for rank, idx in enumerate(top_indices):
            chunk = RetrievedChunk(
                chunk_id=self.corpus_ids[idx],
                text=self.corpus_texts[idx],
                score=combined_scores[idx],
                metadata={
                    "bm25_score": bm25_scores[idx],
                    "semantic_score": semantic_scores[idx],
                },
                rank=rank + 1,
            )
            chunks.append(chunk)
        
        logger.debug(
            f"Hybrid retrieval: {len(chunks)} chunks with combined scores: "
            f"{[f'{c.score:.3f}' for c in chunks]}"
        )
        
        return chunks

    def _bm25_search(self, query: str, top_k: int) -> np.ndarray:
        """Perform BM25 search.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            Array of BM25 scores for entire corpus
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores

    def _semantic_search(self, query: str, top_k: int) -> np.ndarray:
        """Perform semantic search.
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            Array of semantic similarity scores for entire corpus
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store for more candidates
        ids, scores, _, _ = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=min(top_k, len(self.corpus_texts)),
        )
        
        # Create score array for entire corpus
        semantic_scores = np.zeros(len(self.corpus_texts))
        
        # Map returned scores to corpus indices
        for chunk_id, score in zip(ids, scores):
            try:
                idx = self.corpus_ids.index(chunk_id)
                semantic_scores[idx] = score
            except ValueError:
                continue
        
        return semantic_scores

    def _combine_scores(
        self,
        bm25_scores: np.ndarray,
        semantic_scores: np.ndarray,
    ) -> np.ndarray:
        """Combine BM25 and semantic scores using weighted sum.
        
        Args:
            bm25_scores: BM25 scores
            semantic_scores: Semantic similarity scores
            
        Returns:
            Combined scores
        """
        combined = (
            self.bm25_weight * bm25_scores +
            self.semantic_weight * semantic_scores
        )
        return combined

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Generate answer from query and retrieved chunks.
        
        Args:
            query: Query text
            chunks: Retrieved chunks
            
        Returns:
            Generated answer
        """
        chunk_texts = [chunk.text for chunk in chunks]
        answer = self.llm_client.generate_rag_answer(query, chunk_texts)
        return answer
