"""RAG with Cohere Reranker.

Uses Cohere Rerank API with built-in rate limiting to respect Trial key
limits (10 calls/min). Automatically throttles calls and retries on
rate-limit errors with longer backoff.

Reference: Cohere Rerank API, BEIR benchmark
"""

import time
from collections import deque
from typing import List

import cohere
from loguru import logger

from src.models.base_rag import BaseRAG, RetrievedChunk
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.llm_client import LLMClient
from src.utils.config_loader import get_config
from src.utils.cost_tracker import get_cost_tracker


class RerankerRAG(BaseRAG):
    """RAG with Cohere reranking for improved retrieval quality.

    Includes a sliding-window rate limiter that respects Cohere Trial key
    limits (10 calls/min) by throttling before sending requests.

    Reference: Cohere Rerank achieves SOTA on BEIR benchmark (avg 0.524)
    https://docs.cohere.com/docs/rerank
    """

    # Cohere Trial key limit
    _COHERE_RPM = 10
    _COHERE_WINDOW = 60  # seconds

    def __init__(self, vector_store: VectorStore):
        """Initialize reranker RAG.

        Args:
            vector_store: Vector store with indexed documents
        """
        super().__init__("With Cohere Reranker")

        self.config = get_config()
        self.cost_tracker = get_cost_tracker()

        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()

        # Initialize Cohere client
        api_key = self.config.get_api_key("cohere")
        self.cohere_client = cohere.Client(api_key)

        # Reranker configuration
        self.reranker_model = self.config.get(
            "rag_configs.reranker.reranker_model",
            "rerank-english-v3.0"
        )
        self.initial_top_k = self.config.get("rag_configs.reranker.top_k", 10)
        self.final_top_k = self.config.get("rag_configs.reranker.reranker_top_k", 3)

        # Sliding-window rate limiter: track timestamps of recent API calls
        cohere_rpm = self.config.get("rate_limiting.cohere_rpm", self._COHERE_RPM)
        self._rate_limit = cohere_rpm
        self._call_timestamps: deque = deque()

        logger.info(
            f"RerankerRAG initialized: model={self.reranker_model}, "
            f"initial_k={self.initial_top_k}, final_k={self.final_top_k}, "
            f"rate_limit={self._rate_limit} calls/min"
        )

    def _wait_for_rate_limit(self) -> None:
        """Block until we have capacity under the Cohere rate limit."""
        now = time.time()

        # Remove timestamps older than the window
        while self._call_timestamps and self._call_timestamps[0] < now - self._COHERE_WINDOW:
            self._call_timestamps.popleft()

        if len(self._call_timestamps) >= self._rate_limit:
            # Must wait until the oldest call exits the window
            wait_until = self._call_timestamps[0] + self._COHERE_WINDOW
            wait_seconds = wait_until - now + 0.5  # small buffer
            if wait_seconds > 0:
                logger.info(f"Cohere rate limit reached, waiting {wait_seconds:.1f}s...")
                time.sleep(wait_seconds)

        self._call_timestamps.append(time.time())

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve chunks using semantic search + reranking.
        
        Args:
            query: Query text
            top_k: Number of final chunks to return
            
        Returns:
            List of reranked chunks
            
        Best practice: Retrieve more candidates (10x) for effective reranking
        """
        # 1. Initial retrieval with semantic search
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        ids, scores, documents, metadatas = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.initial_top_k,
        )
        
        if not documents:
            logger.warning("No documents retrieved for reranking")
            return []
        
        # 2. Rerank using Cohere
        reranked_results = self._rerank(query, documents, ids, metadatas)
        
        # 3. Return top-k reranked results
        chunks = reranked_results[:top_k]
        
        logger.debug(
            f"Reranked retrieval: {len(chunks)} chunks with scores: "
            f"{[f'{c.score:.3f}' for c in chunks]}"
        )
        
        return chunks

    def _rerank(
        self,
        query: str,
        documents: List[str],
        doc_ids: List[str],
        metadatas: List[dict],
    ) -> List[RetrievedChunk]:
        """Rerank documents using Cohere Rerank API with rate limiting.

        Proactively waits if we're near the rate limit, then retries with
        longer backoff on rate-limit errors (the retry wait is long enough
        to let the window expire).

        Args:
            query: Query text
            documents: Candidate documents
            doc_ids: Document IDs
            metadatas: Document metadata

        Returns:
            Reranked chunks with new scores
        """
        max_retries = self.config.get("rate_limiting.max_retries", 3)

        for attempt in range(max_retries):
            try:
                # Throttle to stay under the Cohere rate limit
                self._wait_for_rate_limit()

                # Call Cohere Rerank API
                response = self.cohere_client.rerank(
                    model=self.reranker_model,
                    query=query,
                    documents=documents,
                    top_n=min(self.final_top_k, len(documents)),
                )

                # Track cost (Cohere rerank is per search, not per token)
                self.cost_tracker.add_entry(
                    service="reranker",
                    operation="rerank",
                    tokens=1,  # 1 search
                    model=self.reranker_model,
                    metadata={"num_documents": len(documents)},
                )

                # Create RetrievedChunk objects from reranked results
                chunks = []
                for rank, result in enumerate(response.results):
                    original_idx = result.index
                    doc_text = documents[original_idx]

                    chunk = RetrievedChunk(
                        chunk_id=doc_ids[original_idx],
                        text=doc_text,
                        score=result.relevance_score,
                        metadata=metadatas[original_idx] or {},
                        rank=rank + 1,
                    )
                    chunks.append(chunk)

                return chunks

            except Exception as e:
                is_rate_limit = "rate" in str(e).lower() or "limit" in str(e).lower()

                if attempt < max_retries - 1:
                    # On rate-limit errors, wait a full minute to reset the window
                    wait_time = 62 if is_rate_limit else 2 ** attempt
                    logger.warning(
                        f"Reranking failed (attempt {attempt + 1}/{max_retries}): "
                        f"{'rate limited' if is_rate_limit else e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Reranking failed after {max_retries} attempts: {e}")
                    logger.warning("Falling back to semantic search results without reranking")
                    return self._create_fallback_chunks(documents, doc_ids, metadatas)

    def _create_fallback_chunks(
        self,
        documents: List[str],
        doc_ids: List[str],
        metadatas: List[dict],
    ) -> List[RetrievedChunk]:
        """Create chunks from original results when reranking fails.
        
        Args:
            documents: Document texts
            doc_ids: Document IDs
            metadatas: Document metadata
            
        Returns:
            List of chunks
        """
        chunks = []
        for rank, (doc_id, text, metadata) in enumerate(
            zip(doc_ids, documents, metadatas)
        ):
            chunk = RetrievedChunk(
                chunk_id=doc_id,
                text=text,
                score=1.0 - (rank * 0.1),  # Decreasing scores
                metadata=metadata or {},
                rank=rank + 1,
            )
            chunks.append(chunk)
        
        return chunks[:self.final_top_k]

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
