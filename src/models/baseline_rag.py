"""Baseline RAG: Semantic search + GPT generation.

Industry best practice: Simple semantic search with vector similarity.
Reference: RAG paper (Lewis et al., 2020)
"""

from typing import List

from loguru import logger

from src.models.base_rag import BaseRAG, RetrievedChunk
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.llm_client import LLMClient


class BaselineRAG(BaseRAG):
    """Baseline RAG configuration: semantic search + generation.
    
    Implementation:
    1. Embed query using OpenAI embeddings
    2. Retrieve top-k most similar chunks from vector store
    3. Generate answer using GPT with retrieved context
    
    Reference: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    (Lewis et al., 2020, NeurIPS)
    """

    def __init__(self, vector_store: VectorStore):
        """Initialize baseline RAG.
        
        Args:
            vector_store: Vector store with indexed documents
        """
        super().__init__("Baseline Semantic Search")
        
        self.vector_store = vector_store
        self.embedding_generator = EmbeddingGenerator()
        self.llm_client = LLMClient()
        
        logger.info("BaselineRAG initialized with semantic search")

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        """Retrieve chunks using semantic similarity search.
        
        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with similarity scores
            
        Best practice: Use cosine similarity for semantic search
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        ids, scores, documents, metadatas = self.vector_store.search(
            query_embedding=query_embedding,
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
            f"Retrieved {len(chunks)} chunks with scores: "
            f"{[f'{c.score:.3f}' for c in chunks]}"
        )
        
        return chunks

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """Generate answer from query and retrieved chunks.
        
        Args:
            query: Query text
            chunks: Retrieved chunks
            
        Returns:
            Generated answer
        """
        # Extract chunk texts
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Generate answer using LLM
        answer = self.llm_client.generate_rag_answer(query, chunk_texts)
        
        return answer
