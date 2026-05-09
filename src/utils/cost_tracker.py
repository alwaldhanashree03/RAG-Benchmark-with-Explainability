"""Cost tracking for API usage.

Industry best practice: Track and limit API costs to prevent overruns.
Reference: OpenAI pricing documentation
"""

from dataclasses import dataclass, field
from typing import Dict, List

from loguru import logger

from src.utils.config_loader import get_config


@dataclass
class CostEntry:
    """Single cost entry for tracking."""
    
    service: str  # openai, cohere, etc.
    operation: str  # embedding, completion, rerank
    tokens: int
    cost_usd: float
    timestamp: str
    metadata: Dict = field(default_factory=dict)


class CostTracker:
    """Track API costs across different services.
    
    Best practice: Enforce cost limits to prevent unexpected charges.
    """

    def __init__(self):
        """Initialize cost tracker."""
        self.config = get_config()
        self.entries: List[CostEntry] = []
        self.max_cost = float(self.config.get("rate_limiting.max_cost_limit", 50.0))
        self._cost_config = self.config.get("cost_config", {})

    def add_entry(
        self,
        service: str,
        operation: str,
        tokens: int,
        model: str = None,
        metadata: Dict = None,
    ) -> float:
        """Add cost entry and return cost.
        
        Args:
            service: Service name (embeddings, llm, reranker)
            operation: Operation type
            tokens: Number of tokens used
            model: Model name
            metadata: Additional metadata
            
        Returns:
            Cost in USD
            
        Raises:
            RuntimeError: If cost limit exceeded
        """
        import datetime
        
        cost_usd = self._calculate_cost(service, model, tokens)
        
        entry = CostEntry(
            service=service,
            operation=operation,
            tokens=tokens,
            cost_usd=cost_usd,
            timestamp=datetime.datetime.now().isoformat(),
            metadata=metadata or {},
        )
        
        self.entries.append(entry)
        
        total_cost = self.get_total_cost()
        if total_cost > self.max_cost:
            logger.error(f"Cost limit exceeded: ${total_cost:.4f} > ${self.max_cost:.2f}")
            raise RuntimeError(
                f"Cost limit exceeded: ${total_cost:.4f}. "
                f"Increase MAX_COST_LIMIT in .env or reduce usage."
            )
        
        logger.debug(
            f"Cost entry: {service}/{operation} - {tokens} tokens = ${cost_usd:.6f} "
            f"(Total: ${total_cost:.4f})"
        )
        
        return cost_usd

    def _calculate_cost(self, service: str, model: str, tokens: int) -> float:
        """Calculate cost based on service and model.
        
        Args:
            service: Service name
            model: Model name
            tokens: Number of tokens
            
        Returns:
            Cost in USD
        """
        cost_per_million = 0.0
        
        if service == "embeddings":
            cost_per_million = self._cost_config.get("embeddings", {}).get(model, 0.02)
        elif service == "llm":
            cost_per_million = self._cost_config.get("llm", {}).get(model, 0.5)
        elif service == "reranker":
            # Cohere rerank is per 1K searches, not tokens
            cost_per_thousand = self._cost_config.get("reranker", {}).get("cohere_rerank", 2.0)
            return (tokens / 1000) * cost_per_thousand
        
        return (tokens / 1_000_000) * cost_per_million

    def get_total_cost(self) -> float:
        """Get total cost across all entries.
        
        Returns:
            Total cost in USD
        """
        return sum(entry.cost_usd for entry in self.entries)

    def get_cost_by_service(self) -> Dict[str, float]:
        """Get cost breakdown by service.
        
        Returns:
            Dictionary mapping service to total cost
        """
        costs = {}
        for entry in self.entries:
            costs[entry.service] = costs.get(entry.service, 0.0) + entry.cost_usd
        return costs

    def get_summary(self) -> Dict:
        """Get cost summary.
        
        Returns:
            Summary dictionary with total, by service, and entry count
        """
        return {
            "total_cost_usd": self.get_total_cost(),
            "by_service": self.get_cost_by_service(),
            "num_entries": len(self.entries),
            "limit_usd": self.max_cost,
            "remaining_usd": self.max_cost - self.get_total_cost(),
        }

    def reset(self) -> None:
        """Reset cost tracker."""
        self.entries = []
        logger.info("Cost tracker reset")


# Global cost tracker instance
_cost_tracker = None


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance (singleton).
    
    Returns:
        CostTracker instance
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
