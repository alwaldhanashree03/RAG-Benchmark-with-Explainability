"""Generate sample data for quick testing without downloading full MS MARCO dataset.

Industry best practice: Provide sample data for development and testing.
"""

import json
from pathlib import Path
from typing import List, Dict
from loguru import logger


class SampleDataGenerator:
    """Generate sample question-answer pairs for testing RAG system."""
    
    # Sample passages based on CS/AI topics (relevant to the course)
    SAMPLE_PASSAGES = [
        {
            "passage_id": "sample_001",
            "text": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. RAG systems first retrieve relevant documents from a knowledge base, then use those documents as context for a language model to generate accurate, grounded responses. This approach reduces hallucination and improves factual accuracy.",
            "source": "RAG Tutorial"
        },
        {
            "passage_id": "sample_002",
            "text": "Vector databases store high-dimensional embeddings and enable efficient similarity search. Common distance metrics include cosine similarity, Euclidean distance (L2), and inner product. Vector databases like ChromaDB, Pinecone, and Weaviate are essential for modern RAG systems.",
            "source": "Vector DB Guide"
        },
        {
            "passage_id": "sample_003",
            "text": "BM25 (Best Matching 25) is a lexical retrieval algorithm based on term frequency and inverse document frequency. Unlike semantic search which uses embeddings, BM25 performs exact keyword matching. Hybrid search combines BM25 with semantic search for improved retrieval accuracy.",
            "source": "Information Retrieval Basics"
        },
        {
            "passage_id": "sample_004",
            "text": "Large Language Models (LLMs) like GPT-3.5 and GPT-4 can generate human-like text but may hallucinate facts. Hallucination occurs when the model generates plausible-sounding but incorrect information. RAG systems mitigate this by grounding responses in retrieved documents.",
            "source": "LLM Safety Guide"
        },
        {
            "passage_id": "sample_005",
            "text": "Text chunking is the process of splitting documents into smaller segments for embedding and retrieval. Common strategies include fixed-size chunking, sentence-based chunking, and recursive chunking. Overlap between chunks helps maintain context continuity.",
            "source": "Text Processing Methods"
        },
        {
            "passage_id": "sample_006",
            "text": "Embeddings are dense vector representations of text that capture semantic meaning. OpenAI's text-embedding-3-small model produces 1536-dimensional vectors. Similar texts have embeddings that are close in vector space, enabling semantic search.",
            "source": "Embedding Models"
        },
        {
            "passage_id": "sample_007",
            "text": "Reranking improves retrieval quality by re-scoring candidate documents. After initial retrieval, a reranker model like Cohere Rerank examines query-document pairs and produces refined relevance scores. This two-stage approach balances speed and accuracy.",
            "source": "Advanced Retrieval Techniques"
        },
        {
            "passage_id": "sample_008",
            "text": "Query decomposition breaks complex questions into simpler sub-queries. For example, 'Compare RAG and fine-tuning for domain adaptation' might decompose into separate questions about RAG and fine-tuning. Responses are then synthesized from multiple retrieval operations.",
            "source": "Query Processing Strategies"
        },
        {
            "passage_id": "sample_009",
            "text": "Evaluation metrics for RAG systems include retrieval metrics (Precision@k, MRR), generation metrics (ROUGE, BLEU), and faithfulness scores. Faithfulness measures whether generated answers are entailed by retrieved context using NLI models.",
            "source": "RAG Evaluation Guide"
        },
        {
            "passage_id": "sample_010",
            "text": "The EU AI Act requires transparency and explainability for high-risk AI systems. RAG systems should provide source attribution and show which documents influenced generated answers. This builds user trust and enables verification.",
            "source": "AI Regulations"
        },
        {
            "passage_id": "sample_011",
            "text": "Temperature controls randomness in language model outputs. Temperature=0 produces deterministic results, while higher values increase diversity. For factual QA systems, low temperature is preferred to ensure consistent, accurate responses.",
            "source": "LLM Configuration"
        },
        {
            "passage_id": "sample_012",
            "text": "Natural Language Inference (NLI) models classify premise-hypothesis pairs as entailment, contradiction, or neutral. In RAG systems, NLI models like DeBERTa can verify if generated answers are supported by retrieved documents, detecting hallucination.",
            "source": "NLI Applications"
        },
        {
            "passage_id": "sample_013",
            "text": "Cost optimization in RAG systems involves balancing API costs, latency, and quality. Using smaller embedding models, caching results, and batching requests can reduce expenses. GPT-3.5-turbo costs $0.50 per million tokens versus $15 for GPT-4.",
            "source": "Cost Management"
        },
        {
            "passage_id": "sample_014",
            "text": "LangChain is a framework for building LLM applications. It provides abstractions for document loading, text splitting, embeddings, vector stores, and chains. LangChain simplifies RAG implementation with modular components.",
            "source": "LangChain Documentation"
        },
        {
            "passage_id": "sample_015",
            "text": "Precision@k measures the proportion of top-k retrieved documents that are relevant. MRR (Mean Reciprocal Rank) evaluates how highly the first relevant document ranks. These metrics assess retrieval quality independent of generation.",
            "source": "Information Retrieval Metrics"
        },
    ]
    
    # Sample queries with ground truth answers
    SAMPLE_QUERIES = [
        {
            "query_id": "q_001",
            "query": "What is RAG and how does it reduce hallucination?",
            "relevant_passages": ["sample_001", "sample_004"],
            "ground_truth": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It reduces hallucination by grounding LLM responses in retrieved documents from a knowledge base, improving factual accuracy."
        },
        {
            "query_id": "q_002",
            "query": "What distance metrics are used in vector databases?",
            "relevant_passages": ["sample_002"],
            "ground_truth": "Common distance metrics for vector databases include cosine similarity, Euclidean distance (L2), and inner product."
        },
        {
            "query_id": "q_003",
            "query": "How does hybrid search combine BM25 and semantic search?",
            "relevant_passages": ["sample_003"],
            "ground_truth": "Hybrid search combines BM25 (lexical/keyword matching) with semantic search (embedding-based) to improve retrieval accuracy by leveraging both exact keyword matches and semantic similarity."
        },
        {
            "query_id": "q_004",
            "query": "What is query decomposition in RAG systems?",
            "relevant_passages": ["sample_008"],
            "ground_truth": "Query decomposition breaks complex questions into simpler sub-queries. Each sub-query is processed separately through retrieval, and the responses are then synthesized into a final answer."
        },
        {
            "query_id": "q_005",
            "query": "What metrics evaluate RAG system performance?",
            "relevant_passages": ["sample_009", "sample_015"],
            "ground_truth": "RAG evaluation metrics include retrieval metrics (Precision@k, MRR), generation metrics (ROUGE, BLEU), and faithfulness scores that measure if answers are supported by retrieved context."
        },
        {
            "query_id": "q_006",
            "query": "Why is reranking used in retrieval?",
            "relevant_passages": ["sample_007"],
            "ground_truth": "Reranking improves retrieval quality by re-scoring candidate documents after initial retrieval. Models like Cohere Rerank provide refined relevance scores, balancing speed and accuracy in a two-stage approach."
        },
        {
            "query_id": "q_007",
            "query": "How can NLI models detect hallucination in RAG?",
            "relevant_passages": ["sample_012"],
            "ground_truth": "NLI (Natural Language Inference) models classify whether generated answers are entailed by retrieved documents. If the answer contradicts or is not supported by the context, it indicates potential hallucination."
        },
        {
            "query_id": "q_008",
            "query": "What is text chunking and why use overlap?",
            "relevant_passages": ["sample_005"],
            "ground_truth": "Text chunking splits documents into smaller segments for embedding and retrieval. Overlap between chunks helps maintain context continuity across chunk boundaries."
        },
        {
            "query_id": "q_009",
            "query": "Why use low temperature for factual QA?",
            "relevant_passages": ["sample_011"],
            "ground_truth": "Low temperature (e.g., 0) produces deterministic, consistent outputs with less randomness. For factual QA systems, this ensures accurate and reliable responses."
        },
        {
            "query_id": "q_010",
            "query": "How do embeddings enable semantic search?",
            "relevant_passages": ["sample_006"],
            "ground_truth": "Embeddings are dense vectors that capture semantic meaning. Similar texts have embeddings close in vector space, enabling semantic search by finding documents with similar meaning rather than just keyword matches."
        },
    ]
    
    def __init__(self, output_dir: str = "data/raw"):
        """Initialize sample data generator.
        
        Args:
            output_dir: Directory to save sample data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self) -> Dict[str, Path]:
        """Generate and save sample dataset.
        
        Returns:
            Dictionary with paths to generated files
        """
        logger.info("Generating sample data for quick testing...")
        
        # Save passages
        passages_path = self.output_dir / "sample_passages.jsonl"
        with open(passages_path, 'w', encoding='utf-8') as f:
            for passage in self.SAMPLE_PASSAGES:
                f.write(json.dumps(passage) + '\n')
        
        logger.info(f"Generated {len(self.SAMPLE_PASSAGES)} sample passages")
        
        # Save queries
        queries_path = self.output_dir / "sample_queries.jsonl"
        with open(queries_path, 'w', encoding='utf-8') as f:
            for query in self.SAMPLE_QUERIES:
                f.write(json.dumps(query) + '\n')
        
        logger.info(f"Generated {len(self.SAMPLE_QUERIES)} sample queries")
        
        # Save as a single JSON for easy loading
        combined_path = self.output_dir / "sample_dataset.json"
        dataset = {
            "passages": self.SAMPLE_PASSAGES,
            "queries": self.SAMPLE_QUERIES,
            "metadata": {
                "num_passages": len(self.SAMPLE_PASSAGES),
                "num_queries": len(self.SAMPLE_QUERIES),
                "description": "Sample dataset for testing RAG benchmark system"
            }
        }
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"[SUCCESS] Sample data saved to {self.output_dir}")
        
        return {
            "passages": passages_path,
            "queries": queries_path,
            "combined": combined_path
        }
    
    @staticmethod
    def load_sample_data(data_dir: str = "data/raw") -> Dict:
        """Load sample dataset from disk.
        
        Args:
            data_dir: Directory containing sample data
            
        Returns:
            Dictionary with passages and queries
        """
        combined_path = Path(data_dir) / "sample_dataset.json"
        
        if not combined_path.exists():
            logger.warning("Sample data not found. Generating...")
            generator = SampleDataGenerator(data_dir)
            generator.generate()
        
        with open(combined_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded sample data: {data['metadata']['num_queries']} queries, {data['metadata']['num_passages']} passages")
        return data


if __name__ == "__main__":
    # Generate sample data when run directly
    generator = SampleDataGenerator()
    paths = generator.generate()
    logger.info("Sample data generated successfully")
    for key, path in paths.items():
        logger.info(f"  {key}: {path}")
