"""FastAPI REST API for RAG system.

Industry best practice: Provide REST API for production deployments.
Reference: FastAPI best practices and async patterns.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
from loguru import logger

from src.data.vector_store import VectorStore
from src.models.baseline_rag import BaselineRAG
from src.models.hybrid_rag import HybridRAG
from src.models.reranker_rag import RerankerRAG
from src.models.query_decomposition_rag import QueryDecompositionRAG
from src.models.hyde_rag import HyDERAG, MultiHyDERAG
from src.models.self_rag import SelfRAG
from src.models.multi_query_rag import MultiQueryRAG, FusionRAG
from src.guardrails.guardrail_checker import GuardrailChecker
from src.evaluation.ragas_metrics import get_ragas_evaluator
from src.evaluation.llm_judge import get_llm_judge
from src.utils.cost_tracker import get_cost_tracker
from src.utils.logger import setup_logger
from src.middleware.rate_limiter import RateLimitMiddleware
from src.middleware.auth import get_api_key


# Initialize FastAPI app
app = FastAPI(
    title="RAG Benchmark API",
    description="Production-ready RAG system with security, rate limiting, and evaluation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    rate_limit=60,  # 60 requests per minute
    window_seconds=60
)


# Pydantic models for request/response
class RetrievedChunkResponse(BaseModel):
    """Retrieved chunk response model."""
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = {}


class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., description="User query", min_length=1)
    config: str = Field(
        default="baseline",
        description="RAG configuration to use",
        pattern="^(baseline|hybrid|reranker|query_decomposition|hyde|multi_hyde|self_rag|multi_query|fusion)$"
    )
    top_k: int = Field(default=3, description="Number of chunks to retrieve", ge=1, le=20)
    apply_guardrails: bool = Field(default=True, description="Apply hallucination guardrails")


class QueryResponse(BaseModel):
    """Query response model."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunkResponse]
    confidence_score: float
    guardrail_triggered: bool
    guardrail_reason: Optional[str] = None
    metadata: Dict[str, Any]
    latency_ms: float
    cost_usd: float


class EvaluationRequest(BaseModel):
    """Evaluation request model."""
    query: str
    answer: str
    retrieved_contexts: List[str]
    ground_truth_answer: Optional[str] = None
    ground_truth_contexts: Optional[List[str]] = None
    use_ragas: bool = Field(default=True, description="Use RAGAS metrics")
    use_llm_judge: bool = Field(default=True, description="Use LLM-as-Judge")


class EvaluationResponse(BaseModel):
    """Evaluation response model."""
    ragas_scores: Optional[Dict[str, float]] = None
    llm_judge_scores: Optional[Dict[str, float]] = None
    combined_score: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    vector_store_docs: int
    cost_summary: Dict[str, Any]


# Global state (initialized on startup)
vector_store = None
rag_models = {}
guardrail_checker = None
ragas_evaluator = None
llm_judge = None


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global vector_store, rag_models, guardrail_checker, ragas_evaluator, llm_judge

    setup_logger()
    logger.info("Starting RAG Benchmark API...")

    # Initialize vector store
    vector_store = VectorStore()
    doc_count = vector_store.get_count()
    logger.info(f"Vector store loaded: {doc_count} documents")

    # Load corpus texts for BM25 (needed by HybridRAG)
    corpus_texts = []
    corpus_ids = []
    if doc_count > 0:
        try:
            from pathlib import Path
            import pandas as pd
            from src.data.text_chunker import TextChunker
            processed_dir = Path("./data/processed")
            if (processed_dir / "passages.parquet").exists():
                passages_df = pd.read_parquet(processed_dir / "passages.parquet")
                chunker = TextChunker()
                chunks = chunker.chunk_documents(passages_df["text"].tolist())
                corpus_texts = [c["text"] for c in chunks]
                corpus_ids = [c["chunk_id"] for c in chunks]
                logger.info(f"Loaded {len(corpus_texts)} chunks for BM25")
        except Exception as e:
            logger.warning(f"Could not load corpus for BM25: {e}. Hybrid model will be unavailable.")

    # Initialize RAG models
    rag_models = {
        "baseline": BaselineRAG(vector_store),
        "reranker": RerankerRAG(vector_store),
        "query_decomposition": QueryDecompositionRAG(vector_store),
        "hyde": HyDERAG(vector_store),
        "multi_hyde": MultiHyDERAG(vector_store, num_hypothetical=3),
        "self_rag": SelfRAG(vector_store, max_iterations=2),
        "multi_query": MultiQueryRAG(vector_store, num_queries=3),
        "fusion": FusionRAG(vector_store, num_queries=3),
    }

    # Only add hybrid if corpus is available
    if corpus_texts:
        rag_models["hybrid"] = HybridRAG(vector_store, corpus_texts, corpus_ids)
    else:
        logger.warning("HybridRAG not initialized: no corpus loaded for BM25")

    logger.info(f"Initialized {len(rag_models)} RAG configurations")
    
    # Initialize guardrails
    guardrail_checker = GuardrailChecker()
    logger.info("Guardrail checker initialized")
    
    # Initialize evaluators
    ragas_evaluator = get_ragas_evaluator()
    llm_judge = get_llm_judge()
    logger.info("Evaluators initialized")
    
    logger.info("API ready to serve requests")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Benchmark API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cost_tracker = get_cost_tracker()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store_docs=vector_store.get_count() if vector_store else 0,
        cost_summary=cost_tracker.get_summary(),
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    """Query the RAG system (requires API key).
    
    Args:
        request: Query request with configuration
        api_key: API key from X-API-Key header
        
    Returns:
        Query response with answer and metadata
    """
    if request.config not in rag_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid config. Choose from: {list(rag_models.keys())}"
        )
    
    # Get RAG model
    rag_model = rag_models[request.config]
    
    # Track time
    start_time = time.time()
    
    # Get initial cost
    cost_tracker = get_cost_tracker()
    initial_cost = cost_tracker.get_total_cost()
    
    try:
        # Query RAG system
        response = rag_model.answer(
            query=request.query,
            top_k=request.top_k,
            apply_guardrails=False,  # We'll apply separately
        )
        
        # Apply guardrails if requested
        if request.apply_guardrails and guardrail_checker:
            triggered, reason, details = guardrail_checker.check_guardrails(
                query=request.query,
                chunks=response.retrieved_chunks,
                answer=response.answer,
            )
            response.guardrail_triggered = triggered
            response.guardrail_reason = reason
            response.metadata.update(details)
        
        # Calculate latency and cost
        latency_ms = (time.time() - start_time) * 1000
        cost_usd = cost_tracker.get_total_cost() - initial_cost
        
        # Convert to response model
        retrieved_chunks = [
            RetrievedChunkResponse(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk.score,
                rank=chunk.rank,
                metadata=chunk.metadata,
            )
            for chunk in response.retrieved_chunks
        ]
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            retrieved_chunks=retrieved_chunks,
            confidence_score=response.confidence_score,
            guardrail_triggered=response.guardrail_triggered,
            guardrail_reason=response.guardrail_reason,
            metadata=response.metadata,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_response(
    request: EvaluationRequest,
    api_key: str = Depends(get_api_key)
):
    """Evaluate a RAG response (requires API key).
    
    Args:
        request: Evaluation request with answer and contexts
        api_key: API key from X-API-Key header
        
    Returns:
        Evaluation scores from RAGAS and LLM-as-Judge
    """
    ragas_scores = None
    llm_judge_scores = None
    
    try:
        # RAGAS evaluation
        if request.use_ragas and ragas_evaluator:
            from src.models.base_rag import RAGResponse, RetrievedChunk
            
            # Create mock response for evaluation
            chunks = [
                RetrievedChunk(
                    chunk_id=f"chunk_{i}",
                    text=ctx,
                    score=1.0,
                    metadata={},
                    rank=i+1
                )
                for i, ctx in enumerate(request.retrieved_contexts)
            ]
            
            mock_response = RAGResponse(
                query=request.query,
                answer=request.answer,
                retrieved_chunks=chunks,
                confidence_score=1.0,
                guardrail_triggered=False,
                guardrail_reason=None,
                metadata={},
            )
            
            ragas_scores = ragas_evaluator.evaluate_response(
                query=request.query,
                response=mock_response,
                ground_truth_answer=request.ground_truth_answer,
                ground_truth_contexts=request.ground_truth_contexts,
            )
        
        # LLM-as-Judge evaluation
        if request.use_llm_judge and llm_judge:
            llm_judge_scores = llm_judge.evaluate_answer_quality(
                query=request.query,
                answer=request.answer,
                retrieved_contexts=request.retrieved_contexts,
                ground_truth=request.ground_truth_answer,
            )
        
        # Calculate combined score
        all_scores = []
        if ragas_scores:
            all_scores.extend([v for v in ragas_scores.values() if v > 0])
        if llm_judge_scores:
            all_scores.extend([v for v in llm_judge_scores.values() if v > 0])
        
        combined_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return EvaluationResponse(
            ragas_scores=ragas_scores,
            llm_judge_scores=llm_judge_scores,
            combined_score=combined_score,
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.get("/configs", response_model=List[str])
async def list_configs():
    """List available RAG configurations."""
    return list(rag_models.keys())


@app.get("/costs", response_model=Dict[str, Any])
async def get_costs():
    """Get current cost tracking information."""
    cost_tracker = get_cost_tracker()
    return cost_tracker.get_summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
