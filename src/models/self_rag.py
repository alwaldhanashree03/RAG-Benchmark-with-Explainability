"""Self-RAG: Self-reflective retrieval-augmented generation.

Industry best practice: Let the model decide when to retrieve and verify its outputs.
Reference: Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023)
"""

from typing import List, Tuple, Optional
from enum import Enum
import re
from loguru import logger

from src.models.base_rag import BaseRAG, RetrievedChunk
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.llm_client import LLMClient


class RetrievalDecision(Enum):
    """Decision on whether to retrieve."""
    RETRIEVE = "retrieve"
    NO_RETRIEVE = "no_retrieve"


class RelevanceScore(Enum):
    """Relevance assessment of retrieved documents."""
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"


class SupportScore(Enum):
    """Support assessment of generated answer."""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class UtilityScore(Enum):
    """Utility assessment of generated answer."""
    HIGHLY_USEFUL = "highly_useful"
    SOMEWHAT_USEFUL = "somewhat_useful"
    NOT_USEFUL = "not_useful"


class SelfRAG(BaseRAG):
    """Self-RAG: Self-reflective retrieval-augmented generation.
    
    Implementation:
    1. Decide if retrieval is needed (retrieval decision)
    2. If needed, retrieve and assess relevance
    3. Generate answer
    4. Assess if answer is supported by evidence (critique)
    5. Assess utility of answer
    6. Optionally re-retrieve if unsupported
    
    Key Innovation: Model reflects on its own retrieval and generation process.
    
    Reference:
    - Asai et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique
    - arXiv:2310.11511
    """

    def __init__(self, vector_store: VectorStore, max_iterations: int = 2):
        """Initialize Self-RAG.
        
        Args:
            vector_store: Vector store with indexed documents
            max_iterations: Maximum self-reflection iterations
        """
        super().__init__("Self-RAG (Self-Reflective)")
        
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()
        self.max_iterations = max_iterations
        
        logger.info(f"SelfRAG initialized with max_iterations={max_iterations}")

    def answer(
        self,
        query: str,
        top_k: int = 3,
        apply_guardrails: bool = True,
    ):
        """Complete Self-RAG pipeline with reflection.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            apply_guardrails: Whether to apply guardrails
            
        Returns:
            RAG response with self-reflection metadata
        """
        # Step 1: Decide if retrieval is needed
        retrieval_decision = self._assess_retrieval_need(query)
        
        metadata = {
            "config_name": self.config_name,
            "retrieval_decision": retrieval_decision.value,
            "iterations": 0,
        }
        
        if retrieval_decision == RetrievalDecision.NO_RETRIEVE:
            # Generate without retrieval
            answer = self._generate_without_retrieval(query)
            
            from src.models.base_rag import RAGResponse
            return RAGResponse(
                query=query,
                answer=answer,
                retrieved_chunks=[],
                confidence_score=0.5,
                guardrail_triggered=False,
                guardrail_reason=None,
                metadata=metadata,
            )
        
        # Step 2-6: Retrieve, generate, and reflect
        return self._retrieve_generate_reflect(query, top_k, metadata)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Standard retrieval (used by base class methods).
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks
        """
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        ids, scores, documents, metadatas = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        
        chunks = []
        for rank, (chunk_id, score, text, metadata) in enumerate(
            zip(ids, scores, documents, metadatas)
        ):
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                score=score,
                metadata=metadata or {},
                rank=rank + 1,
            )
            chunks.append(chunk)
        
        return chunks

    def _assess_retrieval_need(self, query: str) -> RetrievalDecision:
        """Assess if retrieval is needed for this query.
        
        Args:
            query: User query
            
        Returns:
            RetrievalDecision enum
        """
        prompt = f"""Determine if external knowledge retrieval is needed to answer this question.

Question: {query}

Consider:
- Does this require factual knowledge beyond general reasoning?
- Would retrieved documents help provide a better answer?
- Is this a simple conversational query that doesn't need retrieval?

Respond with ONLY one word: "RETRIEVE" or "NO_RETRIEVE"

Decision:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            response_upper = response.strip().upper()
            
            if "NO" in response_upper or "NOT" in response_upper:
                return RetrievalDecision.NO_RETRIEVE
            else:
                return RetrievalDecision.RETRIEVE
                
        except Exception as e:
            logger.error(f"Retrieval decision failed: {e}")
            # Default to retrieve
            return RetrievalDecision.RETRIEVE

    def _assess_relevance(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> Tuple[RelevanceScore, List[RetrievedChunk]]:
        """Assess relevance of retrieved documents.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            
        Returns:
            Tuple of (relevance score, filtered chunks)
        """
        if not chunks:
            return RelevanceScore.IRRELEVANT, []
        
        # Simple heuristic: check retrieval scores
        max_score = max(chunk.score for chunk in chunks)
        
        if max_score >= 0.8:
            relevance = RelevanceScore.RELEVANT
            filtered = chunks
        elif max_score >= 0.6:
            relevance = RelevanceScore.PARTIALLY_RELEVANT
            # Filter to only high-scoring chunks
            filtered = [c for c in chunks if c.score >= 0.6]
        else:
            relevance = RelevanceScore.IRRELEVANT
            filtered = []
        
        logger.debug(f"Relevance assessment: {relevance.value}, {len(filtered)} chunks kept")
        return relevance, filtered

    def _assess_support(self, answer: str, chunks: List[RetrievedChunk]) -> SupportScore:
        """Assess if answer is supported by retrieved evidence.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            SupportScore enum
        """
        if not chunks:
            return SupportScore.NOT_SUPPORTED
        
        context_text = "\n\n".join([chunk.text for chunk in chunks])
        
        prompt = f"""Assess if the answer is supported by the provided evidence.

Evidence:
{context_text}

Answer: {answer}

Is the answer fully supported, partially supported, or not supported by the evidence?

Respond with ONLY one of: "FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", or "NOT_SUPPORTED"

Assessment:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=20)
            response_upper = response.strip().upper()
            
            if "FULLY" in response_upper:
                return SupportScore.FULLY_SUPPORTED
            elif "PARTIALLY" in response_upper:
                return SupportScore.PARTIALLY_SUPPORTED
            else:
                return SupportScore.NOT_SUPPORTED
                
        except Exception as e:
            logger.error(f"Support assessment failed: {e}")
            return SupportScore.PARTIALLY_SUPPORTED

    def _assess_utility(self, query: str, answer: str) -> UtilityScore:
        """Assess utility/usefulness of the answer.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            UtilityScore enum
        """
        prompt = f"""Assess the utility of this answer to the question.

Question: {query}

Answer: {answer}

Is the answer highly useful, somewhat useful, or not useful?

Respond with ONLY one of: "HIGHLY_USEFUL", "SOMEWHAT_USEFUL", or "NOT_USEFUL"

Assessment:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=20)
            response_upper = response.strip().upper()
            
            if "HIGHLY" in response_upper:
                return UtilityScore.HIGHLY_USEFUL
            elif "SOMEWHAT" in response_upper:
                return UtilityScore.SOMEWHAT_USEFUL
            else:
                return UtilityScore.NOT_USEFUL
                
        except Exception as e:
            logger.error(f"Utility assessment failed: {e}")
            return UtilityScore.SOMEWHAT_USEFUL

    def _retrieve_generate_reflect(
        self,
        query: str,
        top_k: int,
        metadata: dict,
    ):
        """Retrieve, generate, and reflect iteratively.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            metadata: Metadata dict to update
            
        Returns:
            RAG response
        """
        from src.models.base_rag import RAGResponse
        
        best_answer = None
        best_chunks = []
        best_support = SupportScore.NOT_SUPPORTED
        
        for iteration in range(self.max_iterations):
            # Retrieve
            chunks = self.retrieve(query, top_k)
            
            # Assess relevance
            relevance, filtered_chunks = self._assess_relevance(query, chunks)
            
            if relevance == RelevanceScore.IRRELEVANT:
                logger.warning(f"Iteration {iteration + 1}: Retrieved docs irrelevant")
                if iteration == 0:
                    # First iteration failed, return without retrieval
                    answer = self._generate_without_retrieval(query)
                    metadata["iterations"] = iteration + 1
                    metadata["relevance"] = relevance.value
                    
                    return RAGResponse(
                        query=query,
                        answer=answer,
                        retrieved_chunks=[],
                        confidence_score=0.3,
                        guardrail_triggered=False,
                        guardrail_reason=None,
                        metadata=metadata,
                    )
                else:
                    # Use best from previous iterations
                    break
            
            # Generate answer
            answer = self.generate(query, filtered_chunks)
            
            # Assess support
            support = self._assess_support(answer, filtered_chunks)
            
            # Assess utility
            utility = self._assess_utility(query, answer)
            
            logger.debug(
                f"Iteration {iteration + 1}: relevance={relevance.value}, "
                f"support={support.value}, utility={utility.value}"
            )
            
            # Update metadata
            metadata[f"iteration_{iteration + 1}_relevance"] = relevance.value
            metadata[f"iteration_{iteration + 1}_support"] = support.value
            metadata[f"iteration_{iteration + 1}_utility"] = utility.value
            
            # Check if we should stop
            if support == SupportScore.FULLY_SUPPORTED and utility == UtilityScore.HIGHLY_USEFUL:
                best_answer = answer
                best_chunks = filtered_chunks
                best_support = support
                metadata["iterations"] = iteration + 1
                break
            
            # Update best if better
            if support.value > best_support.value:
                best_answer = answer
                best_chunks = filtered_chunks
                best_support = support
            
            metadata["iterations"] = iteration + 1
        
        # Use best answer found
        if best_answer is None:
            best_answer = self._generate_without_retrieval(query)
            confidence = 0.3
        else:
            # Calculate confidence based on support
            if best_support == SupportScore.FULLY_SUPPORTED:
                confidence = 0.9
            elif best_support == SupportScore.PARTIALLY_SUPPORTED:
                confidence = 0.6
            else:
                confidence = 0.3
        
        return RAGResponse(
            query=query,
            answer=best_answer,
            retrieved_chunks=best_chunks,
            confidence_score=confidence,
            guardrail_triggered=False,
            guardrail_reason=None,
            metadata=metadata,
        )

    def _generate_without_retrieval(self, query: str) -> str:
        """Generate answer without retrieval.
        
        Args:
            query: User query
            
        Returns:
            Generated answer
        """
        prompt = f"""Answer the following question based on your knowledge:

Question: {query}

Answer:"""

        try:
            answer = self.llm_client.generate(prompt, max_tokens=200)
            return answer.strip()
        except Exception as e:
            logger.error(f"Generation without retrieval failed: {e}")
            return "I don't have enough information to answer this question confidently."

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Generate answer from query and retrieved chunks.
        
        Args:
            query: Original query text
            chunks: Retrieved chunks
            
        Returns:
            Generated answer
        """
        if not chunks:
            return self._generate_without_retrieval(query)
        
        chunk_texts = [chunk.text for chunk in chunks]
        answer = self.llm_client.generate_rag_answer(query, chunk_texts)
        return answer
