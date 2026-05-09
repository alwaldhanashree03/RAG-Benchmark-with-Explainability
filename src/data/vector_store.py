"""Vector database management using ChromaDB.

Industry best practice: Persistent vector store with metadata filtering.
Reference: ChromaDB documentation, BEIR benchmark
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Monkey-patch to fix telemetry capture() signature error
import sys
from unittest.mock import MagicMock

# Create mock for posthog before chromadb imports it
if 'chromadb.telemetry.posthog' not in sys.modules:
    mock_posthog = MagicMock()
    sys.modules['chromadb.telemetry.posthog'] = mock_posthog
    sys.modules['posthog'] = MagicMock()

import chromadb
from chromadb.config import Settings
from loguru import logger
import numpy as np

from src.utils.config_loader import get_config


class VectorStore:
    """Vector database for storing and retrieving document embeddings.
    
    Best practices:
    - Persistent storage for reproducibility
    - Metadata filtering for flexibility
    - Efficient similarity search
    
    Reference: ChromaDB documentation
    https://docs.trychroma.com/
    """

    def __init__(self, collection_name: Optional[str] = None):
        """Initialize vector store.
        
        Args:
            collection_name: Name of collection (default from config)
        """
        self.config = get_config()
        
        # Configuration
        persist_dir = Path(self.config.get("vector_db.persist_directory", "./data/vector_db"))
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name or self.config.get(
            "vector_db.collection_name", "rag_benchmark"
        )
        
        distance_metric = self.config.get("vector_db.distance_metric", "cosine")
        
        # Initialize ChromaDB client
        # Telemetry disabled via environment variable to avoid capture() errors
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        
        # Create or get collection
        # Map distance metrics to ChromaDB space types
        space_map = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",  # Inner product
        }
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": space_map.get(distance_metric, "cosine")},
        )
        
        logger.info(
            f"VectorStore initialized: collection={self.collection_name}, "
            f"metric={distance_metric}, path={persist_dir}"
        )

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Add documents to vector store.
        
        Args:
            ids: Document IDs
            embeddings: Document embeddings
            texts: Document texts
            metadatas: Optional metadata dictionaries
            
        Best practice: Include metadata for filtering and provenance
        """
        if not ids or len(ids) != len(embeddings) != len(texts):
            raise ValueError("ids, embeddings, and texts must have same length")
        
        # Convert numpy arrays to lists for ChromaDB
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # ChromaDB has a maximum batch size limit (typically 5461)
        # Industry best practice: Batch large insertions to avoid API limits
        MAX_BATCH_SIZE = 5000
        total_docs = len(ids)
        
        for i in range(0, total_docs, MAX_BATCH_SIZE):
            end_idx = min(i + MAX_BATCH_SIZE, total_docs)
            
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings_list[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx] if metadatas else None
            
            # Add batch to collection
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
            )
            
            logger.info(f"Added batch {i//MAX_BATCH_SIZE + 1}: {len(batch_ids)} documents ({i+len(batch_ids)}/{total_docs})")
        
        logger.info(f"Successfully added {total_docs} documents to vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> Tuple[List[str], List[float], List[str], List[Dict]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Tuple of (ids, distances, documents, metadatas)
            
        Best practice: Use metadata filtering for targeted retrieval
        """
        # Convert numpy array to list
        query_embedding_list = query_embedding.tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "distances", "metadatas"],
        )
        
        # Extract results (ChromaDB returns lists of lists)
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        # Convert distances to similarity scores (1 - distance for cosine)
        # ChromaDB returns distances, we want similarity scores
        similarity_scores = [1 - d for d in distances]
        
        return ids, similarity_scores, documents, metadatas

    def batch_search(
        self,
        query_embeddings: List[np.ndarray],
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Tuple[List[str], List[float], List[str], List[Dict]]]:
        """Search for multiple queries in batch.
        
        Args:
            query_embeddings: List of query embeddings
            top_k: Number of results per query
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results for each query
        """
        results = []
        for query_emb in query_embeddings:
            result = self.search(query_emb, top_k, filter_metadata)
            results.append(result)
        return results

    def get_count(self) -> int:
        """Get number of documents in collection.
        
        Returns:
            Document count
        """
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the collection.
        
        Warning: This will permanently delete all data in the collection.
        """
        self.client.delete_collection(name=self.collection_name)
        logger.warning(f"Deleted collection: {self.collection_name}")

    def reset(self) -> None:
        """Reset the vector store (delete and recreate collection)."""
        self.delete_collection()
        
        # Recreate collection
        distance_metric = self.config.get("vector_db.distance_metric", "cosine")
        space_map = {"cosine": "cosine", "l2": "l2", "ip": "ip"}
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": space_map.get(distance_metric, "cosine")},
        )
        
        logger.info(f"Reset vector store: {self.collection_name}")
