"""Multi-Query RAG: Generate multiple query variations for better retrieval.

Industry best practice: Query from multiple angles to improve recall.
Reference: MultiQueryRetriever - LangChain pattern
"""

from typing import List
import re
from loguru import logger

from src.models.base_rag import BaseRAG, RetrievedChunk
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.llm_client import LLMClient


class MultiQueryRAG(BaseRAG):
    """Multi-Query RAG: Generate multiple query variations.
    
    Implementation:
    1. Generate multiple variations of the original query
    2. Retrieve with each query variation
    3. Merge and deduplicate results
    4. Rerank by frequency and score
    5. Generate answer from top results
    
    Best practice: Multiple query perspectives improve retrieval recall.
    """

    def __init__(self, vector_store: VectorStore, num_queries: int = 3):
        """Initialize Multi-Query RAG.
        
        Args:
            vector_store: Vector store with indexed documents
            num_queries: Number of query variations to generate
        """
        super().__init__(f"Multi-Query (n={num_queries})")
        
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()
        self.num_queries = num_queries
        
        logger.info(f"MultiQueryRAG initialized with {num_queries} query variations")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve using multiple query variations.
        
        Args:
            query: Original query text
            top_k: Number of final chunks to return
            
        Returns:
            List of retrieved chunks with scores
        """
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        
        logger.debug(f"Query variations: {query_variations}")
        
        # Retrieve with each variation
        all_chunks = {}  # chunk_id -> (chunk, max_score, frequency)
        
        for variation in query_variations:
            # Embed query variation
            query_embedding = self.embedding_generator.generate_embedding(variation)
            
            # Retrieve
            ids, scores, documents, metadatas = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Retrieve more to allow for merging
            )
            
            # Add to collection
            for chunk_id, score, text, metadata in zip(ids, scores, documents, metadatas):
                if chunk_id in all_chunks:
                    # Update max score and increment frequency
                    existing_chunk, max_score, freq = all_chunks[chunk_id]
                    all_chunks[chunk_id] = (
                        existing_chunk,
                        max(max_score, score),
                        freq + 1
                    )
                else:
                    # Add new chunk
                    chunk = RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        score=score,
                        metadata=metadata or {},
                        rank=0,
                    )
                    all_chunks[chunk_id] = (chunk, score, 1)
        
        # Rerank by combined score (max_score * frequency)
        ranked_chunks = []
        for chunk_id, (chunk, max_score, freq) in all_chunks.items():
            # Combined score: favor chunks that appear in multiple queries
            combined_score = max_score * (1 + 0.1 * freq)
            chunk.score = combined_score
            ranked_chunks.append(chunk)
        
        # Sort by combined score and take top-k
        ranked_chunks.sort(key=lambda x: x.score, reverse=True)
        final_chunks = ranked_chunks[:top_k]
        
        # Update ranks
        for rank, chunk in enumerate(final_chunks):
            chunk.rank = rank + 1
        
        logger.debug(
            f"Multi-query retrieval: {len(final_chunks)} chunks from "
            f"{len(all_chunks)} unique, {len(query_variations)} queries"
        )
        
        return final_chunks

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate multiple variations of the query.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations (including original)
        """
        system_prompt = (
            "You are an expert at generating alternative phrasings of questions. "
            "Generate different ways to ask the same question that might retrieve "
            "different relevant documents."
        )
        
        user_prompt = f"""Generate {self.num_queries - 1} alternative versions of this question.
Each version should ask for the same information but use different words and phrasing.

Original question: {query}

Requirements:
1. Keep the same meaning and intent
2. Use different vocabulary and phrasing
3. Vary the specificity (more general or more specific)
4. Generate EXACTLY {self.num_queries - 1} alternatives

Format your response as a numbered list:
1. [First alternative]
2. [Second alternative]
...

Alternatives:"""

        try:
            response = self.llm_client.generate(user_prompt, system_prompt, max_tokens=200)
            
            # Parse alternatives
            alternatives = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Remove numbering
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                if cleaned:
                    alternatives.append(cleaned)
            
            # Limit to requested number
            alternatives = alternatives[:self.num_queries - 1]
            
            # Include original query
            all_queries = [query] + alternatives
            
            return all_queries
            
        except Exception as e:
            logger.error(f"Query variation generation failed: {e}")
            # Fallback to just original query
            return [query]

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Generate answer from query and retrieved chunks.
        
        Args:
            query: Original query text
            chunks: Retrieved chunks
            
        Returns:
            Generated answer
        """
        chunk_texts = [chunk.text for chunk in chunks]
        answer = self.llm_client.generate_rag_answer(query, chunk_texts)
        return answer


class FusionRAG(BaseRAG):
    """Fusion RAG: Reciprocal Rank Fusion for multi-query retrieval.
    
    Extension: Uses Reciprocal Rank Fusion (RRF) algorithm to merge
    results from multiple queries.
    
    Reference: Cormack et al. (2009) - Reciprocal Rank Fusion
    """

    def __init__(self, vector_store: VectorStore, num_queries: int = 3, k: int = 60):
        """Initialize Fusion RAG.
        
        Args:
            vector_store: Vector store with indexed documents
            num_queries: Number of query variations
            k: RRF constant (typically 60)
        """
        super().__init__(f"Fusion RAG (RRF, n={num_queries})")
        
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()
        self.num_queries = num_queries
        self.rrf_k = k
        
        logger.info(f"FusionRAG initialized: num_queries={num_queries}, k={k}")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve using Reciprocal Rank Fusion.
        
        Args:
            query: Original query text
            top_k: Number of final chunks to return
            
        Returns:
            List of retrieved chunks with RRF scores
        """
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        
        # Retrieve with each variation and store rankings
        all_rankings = []  # List of (chunk_id -> rank) dicts
        chunk_data = {}  # chunk_id -> RetrievedChunk
        
        for variation in query_variations:
            query_embedding = self.embedding_generator.generate_embedding(variation)
            
            ids, scores, documents, metadatas = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 3,
            )
            
            # Store ranking
            ranking = {}
            for rank, (chunk_id, score, text, metadata) in enumerate(
                zip(ids, scores, documents, metadatas)
            ):
                ranking[chunk_id] = rank + 1  # 1-indexed rank
                
                # Store chunk data
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = RetrievedChunk(
                        chunk_id=chunk_id,
                        text=text,
                        score=score,
                        metadata=metadata or {},
                        rank=0,
                    )
            
            all_rankings.append(ranking)
        
        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id in chunk_data:
            rrf_score = 0.0
            for ranking in all_rankings:
                if chunk_id in ranking:
                    rank = ranking[chunk_id]
                    rrf_score += 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk_id] = rrf_score
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Create final chunks with RRF scores
        final_chunks = []
        for rank, chunk_id in enumerate(sorted_ids[:top_k]):
            chunk = chunk_data[chunk_id]
            chunk.score = rrf_scores[chunk_id]
            chunk.rank = rank + 1
            final_chunks.append(chunk)
        
        logger.debug(f"RRF retrieval: {len(final_chunks)} chunks with scores")
        
        return final_chunks

    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations (same as MultiQueryRAG)."""
        system_prompt = (
            "You are an expert at generating alternative phrasings of questions."
        )
        
        user_prompt = f"""Generate {self.num_queries - 1} alternative versions of this question:

Original: {query}

Provide ONLY the alternatives, one per line, without numbering.

Alternatives:"""

        try:
            response = self.llm_client.generate(user_prompt, system_prompt, max_tokens=200)
            
            alternatives = [
                line.strip() 
                for line in response.strip().split('\n') 
                if line.strip()
            ][:self.num_queries - 1]
            
            return [query] + alternatives
            
        except Exception as e:
            logger.error(f"Query variation failed: {e}")
            return [query]

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Generate answer from query and retrieved chunks."""
        chunk_texts = [chunk.text for chunk in chunks]
        answer = self.llm_client.generate_rag_answer(query, chunk_texts)
        return answer
