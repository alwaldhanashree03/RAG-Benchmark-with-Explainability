"""Citation and source attribution for RAG answers.

Industry best practice: Provide verifiable sources for generated answers.
Reference: Explainable AI and source attribution patterns.
"""

from typing import List, Dict, Tuple
import re
from loguru import logger

from src.models.base_rag import RetrievedChunk
from src.models.llm_client import LLMClient


class CitationGenerator:
    """Generate citations and source attributions for RAG answers.
    
    Best practices:
    1. Link claims to source documents
    2. Provide inline citations
    3. Generate bibliography
    4. Track citation accuracy
    """

    def __init__(self):
        """Initialize citation generator."""
        self.llm_client = LLMClient()
        logger.info("CitationGenerator initialized")

    def add_citations(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> Dict[str, any]:
        """Add citations to answer.
        
        Args:
            answer: Generated answer
            chunks: Retrieved source chunks
            
        Returns:
            Dictionary with cited answer and metadata
        """
        # Map claims to sources
        claim_sources = self._map_claims_to_sources(answer, chunks)
        
        # Generate cited answer
        cited_answer = self._generate_cited_answer(answer, claim_sources)
        
        # Generate bibliography
        bibliography = self._generate_bibliography(chunks)
        
        # Calculate citation coverage
        coverage = len(claim_sources) / max(1, len(self._extract_claims(answer)))
        
        result = {
            "original_answer": answer,
            "cited_answer": cited_answer,
            "bibliography": bibliography,
            "claim_source_mapping": claim_sources,
            "citation_coverage": coverage,
        }
        
        logger.debug(f"Citations added: {len(claim_sources)} claims cited")
        
        return result

    def _extract_claims(self, answer: str) -> List[str]:
        """Extract claims from answer.
        
        Args:
            answer: Answer text
            
        Returns:
            List of claims (sentences)
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', answer)
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return claims

    def _map_claims_to_sources(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> List[Tuple[str, List[int]]]:
        """Map each claim to supporting source chunks.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            List of (claim, [chunk_indices]) tuples
        """
        claims = self._extract_claims(answer)
        claim_sources = []
        
        for claim in claims:
            # Find which chunks support this claim
            supporting_chunks = self._find_supporting_chunks(claim, chunks)
            
            if supporting_chunks:
                claim_sources.append((claim, supporting_chunks))
        
        return claim_sources

    def _find_supporting_chunks(
        self,
        claim: str,
        chunks: List[RetrievedChunk],
    ) -> List[int]:
        """Find which chunks support a claim.
        
        Args:
            claim: Claim text
            chunks: Retrieved chunks
            
        Returns:
            List of chunk indices that support the claim
        """
        supporting = []
        
        for i, chunk in enumerate(chunks):
            # Simple lexical overlap check (can be improved with NLI)
            claim_words = set(claim.lower().split())
            chunk_words = set(chunk.text.lower().split())
            
            overlap = claim_words.intersection(chunk_words)
            overlap_ratio = len(overlap) / max(1, len(claim_words))
            
            # If sufficient overlap, consider it supporting
            if overlap_ratio > 0.3:
                supporting.append(i + 1)  # 1-indexed
        
        return supporting

    def _generate_cited_answer(
        self,
        answer: str,
        claim_sources: List[Tuple[str, List[int]]],
    ) -> str:
        """Generate answer with inline citations.
        
        Args:
            answer: Original answer
            claim_sources: Claim-to-source mapping
            
        Returns:
            Answer with citations like [1], [2]
        """
        cited_answer = answer
        
        # Add citations to each claim
        for claim, sources in claim_sources:
            if claim in cited_answer and sources:
                # Format citation
                citation = "[" + ",".join(str(s) for s in sources) + "]"
                
                # Add citation after the claim
                cited_answer = cited_answer.replace(
                    claim,
                    f"{claim} {citation}",
                    1  # Only replace first occurrence
                )
        
        return cited_answer

    def _generate_bibliography(
        self,
        chunks: List[RetrievedChunk],
    ) -> List[Dict[str, str]]:
        """Generate bibliography from source chunks.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            List of bibliography entries
        """
        bibliography = []
        
        for i, chunk in enumerate(chunks):
            entry = {
                "index": i + 1,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "score": f"{chunk.score:.3f}",
                "source": chunk.metadata.get("source", "Unknown"),
            }
            bibliography.append(entry)
        
        return bibliography

    def format_bibliography(self, bibliography: List[Dict[str, str]]) -> str:
        """Format bibliography as readable text.
        
        Args:
            bibliography: Bibliography entries
            
        Returns:
            Formatted bibliography string
        """
        lines = ["Sources:"]
        
        for entry in bibliography:
            lines.append(
                f"[{entry['index']}] {entry['text']} (Score: {entry['score']})"
            )
        
        return "\n".join(lines)

    def verify_citations(
        self,
        cited_answer: str,
        chunks: List[RetrievedChunk],
    ) -> Dict[str, any]:
        """Verify that citations are accurate.
        
        Args:
            cited_answer: Answer with citations
            chunks: Retrieved chunks
            
        Returns:
            Verification results
        """
        # Extract citation numbers from answer
        citations = re.findall(r'\[(\d+(?:,\d+)*)\]', cited_answer)
        
        total_citations = len(citations)
        valid_citations = 0
        
        for citation_str in citations:
            citation_nums = [int(n) for n in citation_str.split(',')]
            
            # Check if all cited chunks exist
            if all(1 <= n <= len(chunks) for n in citation_nums):
                valid_citations += 1
        
        accuracy = valid_citations / max(1, total_citations)
        
        return {
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "accuracy": accuracy,
            "invalid_citations": total_citations - valid_citations,
        }


class SourceTracker:
    """Track and manage source attribution."""

    def __init__(self):
        """Initialize source tracker."""
        self.sources = {}
        self.citation_count = {}

    def add_source(self, chunk_id: str, chunk: RetrievedChunk):
        """Add a source to tracker.
        
        Args:
            chunk_id: Unique chunk ID
            chunk: Retrieved chunk
        """
        self.sources[chunk_id] = chunk
        self.citation_count[chunk_id] = 0

    def cite_source(self, chunk_id: str):
        """Record a citation of a source.
        
        Args:
            chunk_id: Chunk ID being cited
        """
        if chunk_id in self.citation_count:
            self.citation_count[chunk_id] += 1

    def get_most_cited(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Get most cited sources.
        
        Args:
            top_k: Number of top sources to return
            
        Returns:
            List of (chunk_id, citation_count) tuples
        """
        sorted_sources = sorted(
            self.citation_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_sources[:top_k]

    def get_statistics(self) -> Dict[str, any]:
        """Get citation statistics.
        
        Returns:
            Statistics dictionary
        """
        total_sources = len(self.sources)
        cited_sources = sum(1 for count in self.citation_count.values() if count > 0)
        total_citations = sum(self.citation_count.values())
        
        return {
            "total_sources": total_sources,
            "cited_sources": cited_sources,
            "uncited_sources": total_sources - cited_sources,
            "total_citations": total_citations,
            "avg_citations_per_source": total_citations / max(1, cited_sources),
        }
