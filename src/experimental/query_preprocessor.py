"""Query preprocessing pipeline for better retrieval.

Industry best practice: Clean and enhance queries before retrieval.
Reference: Query understanding and preprocessing patterns.
"""

from typing import List, Dict, Optional
import re
from loguru import logger

from src.models.llm_client import LLMClient


class QueryPreprocessor:
    """Preprocess queries for better retrieval.
    
    Preprocessing steps:
    1. Spell correction
    2. Query expansion
    3. Entity recognition
    4. Stop word handling
    5. Query classification
    """

    def __init__(self):
        """Initialize query preprocessor."""
        self.llm_client = LLMClient()
        logger.info("QueryPreprocessor initialized")

    def preprocess(self, query: str) -> Dict[str, any]:
        """Preprocess query with all enhancements.
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with preprocessed query and metadata
        """
        result = {
            "original": query,
            "cleaned": self._clean_query(query),
            "corrected": self._spell_correct(query),
            "expanded": self._expand_query(query),
            "entities": self._extract_entities(query),
            "query_type": self._classify_query(query),
        }
        
        # Use corrected or expanded version as final
        result["final"] = result["expanded"] or result["corrected"] or result["cleaned"]
        
        logger.debug(f"Preprocessed query: {query} -> {result['final']}")
        
        return result

    def _clean_query(self, query: str) -> str:
        """Clean query text.
        
        Args:
            query: Original query
            
        Returns:
            Cleaned query
        """
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query)
        
        # Remove special characters (keep basic punctuation)
        cleaned = re.sub(r'[^\w\s\?\.\,\-]', '', cleaned)
        
        # Trim
        cleaned = cleaned.strip()
        
        return cleaned

    def _spell_correct(self, query: str) -> str:
        """Correct spelling errors in query.
        
        Args:
            query: Original query
            
        Returns:
            Spell-corrected query
        """
        # Simple spell correction using LLM
        prompt = f"""Correct any spelling or grammar errors in this question. 
If there are no errors, return the question unchanged.
Return ONLY the corrected question, nothing else.

Question: {query}

Corrected:"""

        try:
            corrected = self.llm_client.generate(prompt, max_tokens=100)
            corrected = corrected.strip()
            
            # Only return if significantly different
            if corrected.lower() != query.lower():
                logger.debug(f"Spell correction: {query} -> {corrected}")
                return corrected
            
            return query
            
        except Exception as e:
            logger.error(f"Spell correction failed: {e}")
            return query

    def _expand_query(self, query: str) -> Optional[str]:
        """Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query or None
        """
        prompt = f"""Expand this query by adding important synonyms and related terms that would help retrieve relevant documents.
Keep it concise. Return ONLY the expanded query.

Original: {query}

Expanded:"""

        try:
            expanded = self.llm_client.generate(prompt, max_tokens=100)
            expanded = expanded.strip()
            
            # Only return if actually expanded
            if len(expanded) > len(query):
                logger.debug(f"Query expansion: {query} -> {expanded}")
                return expanded
            
            return None
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return None

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query.
        
        Args:
            query: Query text
            
        Returns:
            List of extracted entities
        """
        prompt = f"""Extract all named entities (people, places, organizations, dates, etc.) from this question.
Return as a comma-separated list, or "None" if no entities.

Question: {query}

Entities:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=50)
            
            if "none" in response.lower():
                return []
            
            # Parse entities
            entities = [e.strip() for e in response.split(',')]
            entities = [e for e in entities if e and len(e) > 1]
            
            logger.debug(f"Extracted entities: {entities}")
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    def _classify_query(self, query: str) -> str:
        """Classify query type.
        
        Args:
            query: Query text
            
        Returns:
            Query type (factual, opinion, how-to, etc.)
        """
        prompt = f"""Classify this question into ONE of these types:
- factual: Asking for facts or information
- how-to: Asking how to do something
- opinion: Asking for opinions or subjective answers
- comparison: Comparing things
- definition: Asking what something is
- other: None of the above

Question: {query}

Type:"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=20)
            query_type = response.strip().lower()
            
            valid_types = ["factual", "how-to", "opinion", "comparison", "definition", "other"]
            
            for vtype in valid_types:
                if vtype in query_type:
                    return vtype
            
            return "other"
            
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return "other"


class QueryRewriter:
    """Rewrite queries for better retrieval."""

    def __init__(self):
        """Initialize query rewriter."""
        self.llm_client = LLMClient()

    def rewrite_for_retrieval(self, query: str) -> str:
        """Rewrite query to be more suitable for retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query optimized for retrieval
        """
        prompt = f"""Rewrite this question to be more suitable for document retrieval.
Make it more specific and include key terms that would appear in relevant documents.

Original: {query}

Rewritten:"""

        try:
            rewritten = self.llm_client.generate(prompt, max_tokens=100)
            return rewritten.strip()
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
