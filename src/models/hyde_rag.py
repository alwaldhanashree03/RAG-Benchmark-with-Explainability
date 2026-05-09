"""HyDE RAG: Hypothetical Document Embeddings for query expansion.

Industry best practice: Generate hypothetical answer, embed it, use for retrieval.
Reference: Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)
"""

from typing import List
import numpy as np
from loguru import logger

from src.models.base_rag import BaseRAG, RetrievedChunk
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.llm_client import LLMClient


class HyDERAG(BaseRAG):
    """HyDE RAG: Use hypothetical document embeddings for better retrieval.
    
    Implementation:
    1. Generate hypothetical answer to the query using LLM
    2. Embed the hypothetical answer
    3. Use hypothetical answer embedding for retrieval
    4. Generate final answer from retrieved context
    
    Intuition: A hypothetical answer is often closer to relevant documents
    in embedding space than the query itself.
    
    Reference: 
    - Gao et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels
    - arXiv:2212.10496
    """

    def __init__(self, vector_store: VectorStore):
        """Initialize HyDE RAG.
        
        Args:
            vector_store: Vector store with indexed documents
        """
        super().__init__("HyDE (Hypothetical Document Embeddings)")
        
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()
        
        logger.info("HyDERAG initialized")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve chunks using HyDE approach.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with scores
        """
        # Step 1: Generate hypothetical document/answer
        hypothetical_doc = self._generate_hypothetical_document(query)
        
        logger.debug(f"Generated hypothetical document: {hypothetical_doc[:100]}...")
        
        # Step 2: Embed the hypothetical document
        hyde_embedding = self.embedding_generator.generate_embedding(hypothetical_doc)
        
        # Step 3: Search using hypothetical document embedding
        ids, scores, documents, metadatas = self.vector_store.search(
            query_embedding=hyde_embedding,
            top_k=top_k,
        )
        
        # Create RetrievedChunk objects
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
        
        logger.debug(
            f"HyDE retrieval: {len(chunks)} chunks with scores: "
            f"{[f'{c.score:.3f}' for c in chunks]}"
        )
        
        return chunks

    def _generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document that would answer the query.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
            
        Best practice: Generate a detailed, fact-rich answer
        that resembles the documents you want to retrieve.
        """
        system_prompt = (
            "You are an expert at writing detailed, informative passages. "
            "Generate a comprehensive passage that would perfectly answer the given question. "
            "Write as if you are an encyclopedia or textbook. "
            "Include relevant details, facts, and context."
        )
        
        user_prompt = f"""Write a detailed passage that answers this question:

Question: {query}

Requirements:
1. Write 3-5 sentences
2. Be factual and informative
3. Include specific details
4. Use a formal, encyclopedic tone
5. Do NOT say "I don't know" or hedge - write as if you know the answer

Passage:"""

        try:
            hypothetical_doc = self.llm_client.generate(
                user_prompt,
                system_prompt,
                max_tokens=200
            )
            return hypothetical_doc.strip()
            
        except Exception as e:
            logger.error(f"Hypothetical document generation failed: {e}")
            # Fallback to original query
            return query

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


class MultiHyDERAG(BaseRAG):
    """Multi-HyDE: Generate multiple hypothetical documents for diversity.
    
    Extension: Generate multiple hypothetical documents from different
    perspectives, retrieve with each, and merge results.
    """

    def __init__(self, vector_store: VectorStore, num_hypothetical: int = 3):
        """Initialize Multi-HyDE RAG.
        
        Args:
            vector_store: Vector store with indexed documents
            num_hypothetical: Number of hypothetical documents to generate
        """
        super().__init__(f"Multi-HyDE (n={num_hypothetical})")
        
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()
        self.num_hypothetical = num_hypothetical
        
        logger.info(f"MultiHyDERAG initialized with {num_hypothetical} hypothetical docs")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve using multiple hypothetical documents.
        
        Args:
            query: Query text
            top_k: Number of final chunks to return
            
        Returns:
            List of retrieved chunks with scores
        """
        # Generate multiple hypothetical documents
        hypothetical_docs = self._generate_multiple_hypothetical(query)
        
        # Retrieve with each hypothetical document
        all_chunks = {}  # Use dict to deduplicate by chunk_id
        
        for i, hyde_doc in enumerate(hypothetical_docs):
            # Embed hypothetical document
            hyde_embedding = self.embedding_generator.generate_embedding(hyde_doc)
            
            # Retrieve
            ids, scores, documents, metadatas = self.vector_store.search(
                query_embedding=hyde_embedding,
                top_k=top_k,
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
        
        # Sort by score and take top-k
        sorted_chunks = sorted(
            all_chunks.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]
        
        # Update ranks
        for rank, chunk in enumerate(sorted_chunks):
            chunk.rank = rank + 1
        
        logger.debug(
            f"Multi-HyDE retrieval: {len(sorted_chunks)} unique chunks "
            f"from {len(hypothetical_docs)} hypothetical documents"
        )
        
        return sorted_chunks

    def _generate_multiple_hypothetical(self, query: str) -> List[str]:
        """Generate multiple hypothetical documents from different perspectives.
        
        Args:
            query: User query
            
        Returns:
            List of hypothetical documents
        """
        perspectives = [
            "technical and detailed",
            "simple and explanatory",
            "comprehensive and encyclopedic",
        ]
        
        hypothetical_docs = []
        
        for i in range(self.num_hypothetical):
            perspective = perspectives[i % len(perspectives)]
            
            system_prompt = (
                f"You are an expert writer. Generate a {perspective} passage "
                f"that would answer the given question."
            )
            
            user_prompt = f"""Write a passage that answers this question from a {perspective} perspective:

Question: {query}

Write 3-5 sentences. Be factual and informative.

Passage:"""

            try:
                hyde_doc = self.llm_client.generate(
                    user_prompt,
                    system_prompt,
                    max_tokens=200
                )
                hypothetical_docs.append(hyde_doc.strip())
                
            except Exception as e:
                logger.error(f"Hypothetical document {i+1} generation failed: {e}")
                # Use query as fallback
                hypothetical_docs.append(query)
        
        return hypothetical_docs

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
