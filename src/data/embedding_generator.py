"""Embedding generation using OpenAI API.

Batch processing with retry logic, cost tracking, and in-memory caching.
Reference: OpenAI embeddings documentation
"""

import hashlib
import time
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from loguru import logger
from tqdm import tqdm

from src.utils.config_loader import get_config
from src.utils.cost_tracker import get_cost_tracker


# Module-level shared cache so all EmbeddingGenerator instances share it
_embedding_cache: Dict[str, np.ndarray] = {}


def get_cache_stats() -> Dict[str, int]:
    """Return current cache statistics."""
    return {"cached_embeddings": len(_embedding_cache)}


def clear_embedding_cache() -> None:
    """Clear the shared embedding cache."""
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API with in-memory caching.

    All instances share a module-level cache so that identical texts
    are never embedded twice within a session. This eliminates redundant
    API calls when benchmarking multiple RAG configs on the same queries.

    Reference: OpenAI Embeddings API
    https://platform.openai.com/docs/guides/embeddings
    """

    def __init__(self):
        """Initialize embedding generator."""
        self.config = get_config()
        self.cost_tracker = get_cost_tracker()

        # Get API key and initialize client
        import os
        api_key = self.config.get_api_key("openai")
        os.environ.pop('OPENAI_ORG_ID', None)
        self.client = OpenAI(api_key=api_key)

        # Configuration
        self.model = self.config.get("embeddings.model", "text-embedding-3-small")
        self.dimensions = self.config.get("embeddings.dimensions", 1536)
        self.batch_size = self.config.get("embeddings.batch_size", 100)
        self.max_retries = self.config.get("embeddings.max_retries", 3)

        logger.info(f"EmbeddingGenerator initialized: model={self.model}, dim={self.dimensions}")

    @staticmethod
    def _cache_key(text: str) -> str:
        """Deterministic cache key for a text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text (cached).

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        key = self._cache_key(text)
        if key in _embedding_cache:
            return _embedding_cache[key]

        embeddings = self.generate_embeddings([text], show_progress=False)
        return embeddings[0]

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with batching and caching.

        Texts already in cache are returned immediately. Only uncached texts
        are sent to the API, then stored in cache for future calls.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        # Separate cached vs uncached
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in _embedding_cache:
                results[i] = _embedding_cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        cache_hits = len(texts) - len(uncached_texts)

        if uncached_texts:
            # Process uncached texts in batches
            batches = [
                uncached_texts[i:i + self.batch_size]
                for i in range(0, len(uncached_texts), self.batch_size)
            ]

            if show_progress and len(batches) > 1:
                desc = f"Embedding {len(uncached_texts)} texts ({cache_hits} cached)"
                iterator = tqdm(batches, desc=desc)
            else:
                iterator = batches

            new_embeddings: List[np.ndarray] = []
            for batch in iterator:
                batch_embeddings = self._generate_batch(batch)
                new_embeddings.extend(batch_embeddings)

            # Store in cache and fill results
            for idx, emb, text in zip(uncached_indices, new_embeddings, uncached_texts):
                key = self._cache_key(text)
                _embedding_cache[key] = emb
                results[idx] = emb

            logger.debug(
                f"Embeddings: {len(uncached_texts)} generated, {cache_hits} from cache"
            )
        else:
            logger.debug(f"Embeddings: all {cache_hits} from cache")

        return results  # type: ignore[return-value]

    def _generate_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch with retry logic.
        
        Args:
            texts: Batch of texts
            
        Returns:
            List of embeddings
        """
        for attempt in range(self.max_retries):
            try:
                # Call OpenAI API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions,
                )
                
                # Extract embeddings
                embeddings = [np.array(item.embedding) for item in response.data]
                
                # Track cost
                total_tokens = response.usage.total_tokens
                self.cost_tracker.add_entry(
                    service="embeddings",
                    operation="generate",
                    tokens=total_tokens,
                    model=self.model,
                    metadata={"batch_size": len(texts)},
                )
                
                return embeddings
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Embedding generation failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Embedding generation failed after {self.max_retries} attempts: {e}")
                    raise

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.dimensions
