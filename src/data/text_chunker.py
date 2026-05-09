"""Text chunking using LangChain RecursiveCharacterTextSplitter.

Industry best practice: Chunk with overlap to preserve context across boundaries.
Reference: LangChain documentation on text splitting strategies.
"""

from typing import List

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from src.utils.config_loader import get_config


class TextChunker:
    """Chunk documents for embedding and retrieval.
    
    Best practice: Use recursive splitting with overlap to maintain semantic coherence.
    Reference: LangChain RecursiveCharacterTextSplitter
    """

    def __init__(self):
        """Initialize text chunker with configuration."""
        self.config = get_config()
        
        # Get chunking parameters
        self.chunk_size = self.config.get("chunking.chunk_size", 512)
        self.chunk_overlap = self.config.get("chunking.chunk_overlap", 50)
        self.separators = self.config.get("chunking.separators", ["\n\n", "\n", " ", ""])
        
        # Initialize tokenizer for accurate token counting
        # Using cl100k_base encoding (GPT-3.5, GPT-4)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Create text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=self.separators,
        )
        
        logger.info(
            f"TextChunker initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def _token_length(self, text: str) -> int:
        """Calculate token length using tiktoken.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        """Chunk a single text into smaller pieces.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = self.splitter.split_text(text)
        return chunks

    def chunk_documents(self, documents: List[str]) -> List[dict]:
        """Chunk multiple documents with metadata.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of chunk dictionaries with metadata
            
        Example:
            >>> chunker = TextChunker()
            >>> docs = ["Long document text...", "Another document..."]
            >>> chunks = chunker.chunk_documents(docs)
            >>> chunks[0]
            {
                'chunk_id': 'doc_0_chunk_0',
                'text': 'chunk text...',
                'source_doc_id': 0,
                'chunk_index': 0,
                'num_tokens': 150
            }
        """
        all_chunks = []
        chunk_id_counter = 0
        
        for doc_idx, doc_text in enumerate(documents):
            chunks = self.chunk_text(doc_text)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_data = {
                    "chunk_id": f"doc_{doc_idx}_chunk_{chunk_idx}",
                    "text": chunk_text,
                    "source_doc_id": doc_idx,
                    "chunk_index": chunk_idx,
                    "num_tokens": self._token_length(chunk_text),
                }
                all_chunks.append(chunk_data)
                chunk_id_counter += 1
        
        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks "
            f"(avg {len(all_chunks)/len(documents):.1f} chunks/doc)"
        )
        
        return all_chunks

    def get_stats(self, chunks: List[dict]) -> dict:
        """Get statistics about chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {}
        
        token_counts = [c["num_tokens"] for c in chunks]
        
        return {
            "num_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
        }
