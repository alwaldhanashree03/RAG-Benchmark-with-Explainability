"""RAG with Query Decomposition.

Industry best practice: Decompose complex queries into sub-queries for better retrieval.
Reference: Query decomposition strategies in complex QA systems
"""

from typing import List

from loguru import logger

from src.models.base_rag import BaseRAG, RetrievedChunk
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.llm_client import LLMClient
from src.utils.config_loader import get_config


class QueryDecompositionRAG(BaseRAG):
    """RAG with query decomposition for complex questions.
    
    Implementation:
    1. Decompose complex query into simpler sub-queries using LLM
    2. Retrieve relevant chunks for each sub-query
    3. Aggregate and deduplicate retrieved chunks
    4. Generate answer from aggregated context
    
    Best practice: Query decomposition improves handling of multi-part questions
    Reference: Multi-hop QA systems, HotpotQA benchmark
    """

    def __init__(self, vector_store: VectorStore):
        """Initialize query decomposition RAG.
        
        Args:
            vector_store: Vector store with indexed documents
        """
        super().__init__("Query Decomposition")
        
        self.config = get_config()
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()
        
        # Configuration
        self.max_subqueries = self.config.get(
            "rag_configs.query_decomposition.max_subqueries", 3
        )
        self.top_k_per_subquery = self.config.get(
            "rag_configs.query_decomposition.top_k_per_subquery", 2
        )
        self.final_top_k = self.config.get(
            "rag_configs.query_decomposition.final_top_k", 3
        )
        
        logger.info(
            f"QueryDecompositionRAG initialized: max_subqueries={self.max_subqueries}, "
            f"top_k_per_sub={self.top_k_per_subquery}"
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve chunks using query decomposition.
        
        Args:
            query: Complex query text
            top_k: Number of final chunks to return
            
        Returns:
            List of retrieved chunks from all sub-queries
        """
        # 1. Decompose query into sub-queries
        subqueries = self._decompose_query(query)
        
        if not subqueries:
            logger.warning("Query decomposition failed, falling back to original query")
            subqueries = [query]
        
        logger.debug(f"Decomposed into {len(subqueries)} sub-queries: {subqueries}")
        
        # 2. Retrieve chunks for each sub-query
        all_chunks = {}  # Use dict to deduplicate by chunk_id
        
        for subquery in subqueries:
            query_embedding = self.embedding_generator.generate_embedding(subquery)
            
            ids, scores, documents, metadatas = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k_per_subquery,
            )
            
            # Add to collection with max score if duplicate
            for chunk_id, score, text, metadata in zip(ids, scores, documents, metadatas):
                if chunk_id not in all_chunks or score > all_chunks[chunk_id].score:
                    all_chunks[chunk_id] = RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        score=score,
                        metadata=metadata or {},
                        rank=0,  # Will be updated later
                    )
        
        # 3. Sort by score and take top-k
        sorted_chunks = sorted(
            all_chunks.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]
        
        # Update ranks
        for rank, chunk in enumerate(sorted_chunks):
            chunk.rank = rank + 1
        
        logger.debug(
            f"Query decomposition retrieval: {len(sorted_chunks)} unique chunks "
            f"from {len(subqueries)} sub-queries"
        )
        
        return sorted_chunks

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into simpler sub-queries.
        
        Args:
            query: Complex query
            
        Returns:
            List of sub-queries
            
        Best practice: Use LLM to intelligently decompose complex questions
        """
        system_prompt = (
            "You are a query decomposition expert. Break down complex questions into "
            "simpler sub-questions that can be answered independently. "
            f"Generate at most {self.max_subqueries} sub-questions."
        )
        
        user_prompt = f"""Decompose this question into simpler sub-questions:

Question: {query}

Rules:
1. Generate {self.max_subqueries} or fewer sub-questions
2. Each sub-question should be self-contained and answerable independently
3. Sub-questions should cover different aspects of the original question
4. Format: Return ONLY the sub-questions, one per line, numbered
5. If the question is already simple, return it as is

Sub-questions:"""
        
        try:
            response = self.llm_client.generate(user_prompt, system_prompt, max_tokens=200)
            
            # Parse sub-queries from response
            subqueries = []
            for line in response.strip().split('\n'):
                # Remove numbering and clean up
                line = line.strip()
                if not line:
                    continue
                
                # Remove leading numbers, dots, parentheses
                import re
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                if cleaned:
                    subqueries.append(cleaned)
            
            # Limit to max_subqueries
            subqueries = subqueries[:self.max_subqueries]
            
            return subqueries if subqueries else [query]
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]  # Fallback to original query

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Generate answer from original query and retrieved chunks.
        
        Args:
            query: Original complex query
            chunks: Retrieved chunks from all sub-queries
            
        Returns:
            Generated answer
        """
        chunk_texts = [chunk.text for chunk in chunks]
        answer = self.llm_client.generate_rag_answer(query, chunk_texts)
        return answer
