"""Semantic chunking: Split documents by topic/semantic boundaries.

Industry best practice: Chunk by semantic similarity rather than fixed size.
Reference: Context-aware text chunking for better retrieval.
"""

from typing import List, Dict
import numpy as np
from loguru import logger

from src.data.embedding_generator import EmbeddingGenerator
from src.utils.config_loader import get_config


class SemanticChunker:
    """Chunk documents based on semantic similarity.
    
    Implementation:
    1. Split document into sentences
    2. Embed each sentence
    3. Calculate similarity between adjacent sentences
    4. Split where similarity drops below threshold
    
    Best practice: Semantic chunking preserves topical coherence.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        """Initialize semantic chunker.
        
        Args:
            similarity_threshold: Similarity threshold for splitting (0-1)
        """
        self.config = get_config()
        self.embedding_generator = EmbeddingGenerator()
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"SemanticChunker initialized with threshold={similarity_threshold}")

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text by semantic boundaries.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of semantic chunks
        """
        if not text or not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        # Embed sentences
        embeddings = self.embedding_generator.generate_embeddings(
            sentences,
            show_progress=False
        )
        
        # Calculate similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Find split points where similarity drops
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_indices.append(i + 1)
        split_indices.append(len(sentences))
        
        # Create chunks
        chunks = []
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]
            chunk = " ".join(sentences[start:end])
            chunks.append(chunk)
        
        logger.debug(f"Semantic chunking: {len(sentences)} sentences -> {len(chunks)} chunks")
        
        return chunks

    def chunk_documents(self, documents: List[str]) -> List[Dict]:
        """Chunk multiple documents with metadata.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of chunk dictionaries with metadata
        """
        all_chunks = []
        chunk_id_counter = 0
        
        for doc_idx, doc_text in enumerate(documents):
            chunks = self.chunk_text(doc_text)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_data = {
                    "chunk_id": f"semantic_doc_{doc_idx}_chunk_{chunk_idx}",
                    "text": chunk_text,
                    "source_doc_id": doc_idx,
                    "chunk_index": chunk_idx,
                    "chunking_method": "semantic",
                }
                all_chunks.append(chunk_data)
                chunk_id_counter += 1
        
        logger.info(
            f"Semantic chunked {len(documents)} documents into {len(all_chunks)} chunks "
            f"(avg {len(all_chunks)/len(documents):.1f} chunks/doc)"
        )
        
        return all_chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        import re
        
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class HierarchicalChunker:
    """Hierarchical chunking: Create parent-child chunk relationships.
    
    Implementation:
    1. Create large parent chunks (e.g., sections)
    2. Create smaller child chunks within each parent
    3. Maintain parent-child relationships
    
    Best practice: Retrieve child chunks but include parent context.
    """

    def __init__(self, parent_size: int = 2000, child_size: int = 500):
        """Initialize hierarchical chunker.
        
        Args:
            parent_size: Parent chunk size (tokens)
            child_size: Child chunk size (tokens)
        """
        self.parent_size = parent_size
        self.child_size = child_size
        
        # Use standard chunker for token-based splitting
        from src.data.text_chunker import TextChunker
        self.text_chunker = TextChunker()
        
        logger.info(
            f"HierarchicalChunker initialized: "
            f"parent={parent_size}, child={child_size}"
        )

    def chunk_documents(self, documents: List[str]) -> List[Dict]:
        """Create hierarchical chunks.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of chunk dictionaries with parent-child relationships
        """
        all_chunks = []
        
        for doc_idx, doc_text in enumerate(documents):
            # Create parent chunks
            parent_chunks = self._create_parent_chunks(doc_text, doc_idx)
            
            # Create child chunks for each parent
            for parent in parent_chunks:
                child_chunks = self._create_child_chunks(
                    parent["text"],
                    parent["chunk_id"],
                    doc_idx
                )
                
                # Add children to output
                all_chunks.extend(child_chunks)
        
        logger.info(f"Created {len(all_chunks)} hierarchical chunks")
        
        return all_chunks

    def _create_parent_chunks(self, text: str, doc_idx: int) -> List[Dict]:
        """Create parent-level chunks.
        
        Args:
            text: Document text
            doc_idx: Document index
            
        Returns:
            List of parent chunk dicts
        """
        # Temporarily set chunker size to parent size
        original_size = self.text_chunker.chunk_size
        self.text_chunker.chunk_size = self.parent_size
        
        parent_texts = self.text_chunker.chunk_text(text)
        
        # Restore original size
        self.text_chunker.chunk_size = original_size
        
        parents = []
        for idx, parent_text in enumerate(parent_texts):
            parents.append({
                "chunk_id": f"parent_doc_{doc_idx}_chunk_{idx}",
                "text": parent_text,
                "level": "parent",
            })
        
        return parents

    def _create_child_chunks(
        self,
        parent_text: str,
        parent_id: str,
        doc_idx: int
    ) -> List[Dict]:
        """Create child chunks within a parent.
        
        Args:
            parent_text: Parent chunk text
            parent_id: Parent chunk ID
            doc_idx: Document index
            
        Returns:
            List of child chunk dicts
        """
        # Temporarily set chunker size to child size
        original_size = self.text_chunker.chunk_size
        self.text_chunker.chunk_size = self.child_size
        
        child_texts = self.text_chunker.chunk_text(parent_text)
        
        # Restore original size
        self.text_chunker.chunk_size = original_size
        
        children = []
        for idx, child_text in enumerate(child_texts):
            children.append({
                "chunk_id": f"{parent_id}_child_{idx}",
                "text": child_text,
                "level": "child",
                "parent_id": parent_id,
                "parent_text": parent_text,  # Include parent for context
                "source_doc_id": doc_idx,
                "chunking_method": "hierarchical",
            })
        
        return children
