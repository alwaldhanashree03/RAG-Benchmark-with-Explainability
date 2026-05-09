"""Streamlit UI for RAG system with explainability.

Industry best practice: User-friendly interface with evidence transparency.
Reference: Explainable AI principles - show evidence and reasoning
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import time
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger
from src.data.data_loader import DatasetLoader
from src.data.text_chunker import TextChunker
from src.data.embedding_generator import EmbeddingGenerator
from src.data.vector_store import VectorStore
from src.models.baseline_rag import BaselineRAG
from src.models.hybrid_rag import HybridRAG
from src.models.reranker_rag import RerankerRAG
from src.models.query_decomposition_rag import QueryDecompositionRAG
from src.guardrails.guardrail_checker import GuardrailChecker
from src.utils.cost_tracker import get_cost_tracker


# Page configuration
st.set_page_config(
    page_title="RAG Benchmark with Explainability",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .chunk-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-score {
        color: #28a745;
        font-weight: bold;
    }
    .med-score {
        color: #ffc107;
        font-weight: bold;
    }
    .low-score {
        color: #dc3545;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize RAG system components (cached)."""
    setup_logger()
    config = get_config()
    
    # Load data
    loader = DatasetLoader()
    queries_df, passages_df = loader.load()
    
    # Chunk documents
    chunker = TextChunker()
    chunks = chunker.chunk_documents(passages_df["text"].tolist())
    
    # Generate embeddings and build vector store
    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore()
    
    if vector_store.get_count() == 0:
        st.info("Building vector store (first run)... This may take a few minutes.")
        
        chunk_texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]
        
        embeddings = embedding_gen.generate_embeddings(chunk_texts, show_progress=True)
        
        vector_store.add_documents(
            ids=chunk_ids,
            embeddings=embeddings,
            texts=chunk_texts,
        )
    
    # Initialize RAG models
    rag_models = {
        "Baseline": BaselineRAG(vector_store),
        "Hybrid": HybridRAG(
            vector_store,
            corpus_texts=[c["text"] for c in chunks],
            corpus_ids=[c["chunk_id"] for c in chunks],
        ),
        "Reranker": RerankerRAG(vector_store),
        "Query Decomposition": QueryDecompositionRAG(vector_store),
    }
    
    guardrail_checker = GuardrailChecker()
    
    return queries_df, rag_models, guardrail_checker


def initialize_custom_system(passages_df, queries_df=None):
    """Initialize RAG system with custom uploaded data."""
    setup_logger()
    
    # Chunk documents
    chunker = TextChunker()
    chunks = chunker.chunk_documents(passages_df["text"].tolist())
    
    # Generate embeddings and build vector store
    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore(collection_name="custom_rag_benchmark")
    
    # Always rebuild for custom data
    st.info(f"Processing {len(chunks)} chunks from your uploaded documents...")
    
    chunk_texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    
    embeddings = embedding_gen.generate_embeddings(chunk_texts, show_progress=True)
    
    vector_store.add_documents(
        ids=chunk_ids,
        embeddings=embeddings,
        texts=chunk_texts,
    )
    
    # Initialize RAG models
    rag_models = {
        "Baseline": BaselineRAG(vector_store),
        "Hybrid": HybridRAG(
            vector_store,
            corpus_texts=[c["text"] for c in chunks],
            corpus_ids=[c["chunk_id"] for c in chunks],
        ),
        "Reranker": RerankerRAG(vector_store),
        "Query Decomposition": QueryDecompositionRAG(vector_store),
    }
    
    guardrail_checker = GuardrailChecker()
    
    # Return empty queries if not provided
    if queries_df is None or queries_df.empty:
        queries_df = pd.DataFrame(columns=["query_id", "query"])
    
    return queries_df, rag_models, guardrail_checker


def display_retrieved_chunks(chunks, guardrail_details):
    """Display retrieved chunks with scores and highlighting."""
    st.subheader("Retrieved Evidence")
    
    # Visualization: Score distribution
    if len(chunks) > 0:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Rank {c.rank}" for c in chunks],
            y=[c.score for c in chunks],
            marker_color=['#28a745' if c.score >= 0.8 else '#ffc107' if c.score >= 0.6 else '#dc3545' for c in chunks],
            text=[f"{c.score:.3f}" for c in chunks],
            textposition='auto',
            hovertemplate='<b>Rank %{x}</b><br>Score: %{y:.3f}<extra></extra>',
        ))
        fig.update_layout(
            title="Retrieval Score Distribution",
            xaxis_title="Chunk Rank",
            yaxis_title="Similarity Score",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display chunks
    for chunk in chunks:
        # Color code based on score
        if chunk.score >= 0.8:
            score_color = "HIGH"
            score_class = "high-score"
        elif chunk.score >= 0.6:
            score_color = "MEDIUM"
            score_class = "med-score"
        else:
            score_color = "LOW"
            score_class = "low-score"
        
        with st.expander(
            f"**Rank #{chunk.rank}** | {score_color} | Score: {chunk.score:.3f}",
            expanded=(chunk.rank <= 2)
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Content:**")
                st.info(chunk.text)
            
            with col2:
                st.markdown(f"**Chunk ID:**")
                st.code(chunk.chunk_id, language=None)
                
                st.markdown(f"**Score:**")
                st.markdown(f"<span class='{score_class}'>{chunk.score:.4f}</span>", unsafe_allow_html=True)
            
            # Show metadata if available
            if chunk.metadata:
                st.markdown("**Metadata:**")
                st.json(chunk.metadata)
    
    # Show guardrail details
    if guardrail_details:
        st.divider()
        st.subheader("Guardrail Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            retrieval_score = guardrail_details.get('retrieval_score', 0)
            threshold = guardrail_details.get('retrieval_threshold', 0)
            delta_val = retrieval_score - threshold
            st.metric(
                "Retrieval Score",
                f"{retrieval_score:.3f}",
                delta=f"{delta_val:+.3f}",
                delta_color="normal" if delta_val >= 0 else "inverse"
            )
        
        with col2:
            if "nli_score" in guardrail_details:
                nli_score = guardrail_details['nli_score']
                nli_threshold = guardrail_details.get('nli_threshold', 0)
                nli_delta = nli_score - nli_threshold
                st.metric(
                    "NLI Entailment",
                    f"{nli_score:.3f}",
                    delta=f"{nli_delta:+.3f}",
                    delta_color="normal" if nli_delta >= 0 else "inverse"
                )
        
        with col3:
            confidence = guardrail_details.get("confidence_level", "unknown")
            st.metric("Confidence Level", f"{confidence.upper()}")
        
        with col4:
            passed = guardrail_details.get("passed", True)
            st.metric("Guardrail Status", "PASSED" if passed else "FAILED")


def display_answer(response):
    """Display generated answer with confidence indicator."""
    st.subheader("Generated Answer")
    
    if response.guardrail_triggered:
        st.warning(f"**Guardrail Triggered:** {response.guardrail_reason}")
        st.error(response.answer)
    else:
        # Confidence indicator
        confidence_score = response.confidence_score
        
        # Visual confidence bar
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if confidence_score >= 0.8:
                st.success(f"**High Confidence** ({confidence_score:.2%})")
            elif confidence_score >= 0.6:
                st.info(f"**Medium Confidence** ({confidence_score:.2%})")
            else:
                st.warning(f"**Low Confidence** ({confidence_score:.2%})")
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffebee"},
                        {'range': [40, 60], 'color': "#fff9e6"},
                        {'range': [60, 80], 'color': "#e8f5e9"},
                        {'range': [80, 100], 'color': "#c8e6c9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        # Answer text
        st.markdown("### Answer:")
        st.markdown(f"<div style='background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #667eea;'>{response.answer}</div>", unsafe_allow_html=True)


def compare_models(query, rag_models, guardrail_checker, top_k, apply_guardrails):
    """Compare multiple RAG models side-by-side."""
    results = {}
    
    for model_name, model in rag_models.items():
        start_time = time.time()
        response = model.answer(query, top_k=top_k, apply_guardrails=False)
        
        # Apply guardrails (always check to get confidence level)
        triggered, reason, details = guardrail_checker.check_guardrails(
            query=query,
            chunks=response.retrieved_chunks,
            answer=response.answer,
        )
        
        if apply_guardrails:
            response.guardrail_triggered = triggered
            response.guardrail_reason = reason
        
        # Always update metadata with confidence level
        response.metadata.update(details)
        
        end_time = time.time()
        
        results[model_name] = {
            'response': response,
            'latency': (end_time - start_time) * 1000
        }
    
    return results


def display_comparison_table(results):
    """Display comparison table for multiple models."""
    comparison_data = []
    
    for model_name, data in results.items():
        response = data['response']
        comparison_data.append({
            'Model': model_name,
            'Confidence': f"{response.confidence_score:.2%}",
            'Chunks': len(response.retrieved_chunks),
            'Avg Score': f"{sum(c.score for c in response.retrieved_chunks) / len(response.retrieved_chunks):.3f}" if response.retrieved_chunks else "N/A",
            'Latency (ms)': f"{data['latency']:.0f}",
            'Guardrail': "FAILED" if response.guardrail_triggered else "PASSED"
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export comparison
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Export Comparison", use_container_width=True):
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'models': []
            }
            
            for model_name, data in results.items():
                response = data['response']
                export_data['models'].append({
                    'name': model_name,
                    'query': response.query,
                    'answer': response.answer,
                    'confidence': response.confidence_score,
                    'latency_ms': data['latency'],
                    'guardrail_triggered': response.guardrail_triggered,
                    'chunks': [
                        {
                            'rank': c.rank,
                            'score': c.score,
                            'text': c.text,
                            'chunk_id': c.chunk_id
                        } for c in response.retrieved_chunks
                    ]
                })
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )


def main():
    """Main Streamlit app."""
    # Header with custom styling
    st.markdown('<p class="main-header">RAG Benchmark with Explainability</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transparent Retrieval-Augmented Generation System with Advanced Guardrails</p>', unsafe_allow_html=True)
    
    # Initialize session state for query history
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'use_custom_data' not in st.session_state:
        st.session_state.use_custom_data = False
    if 'custom_data_loaded' not in st.session_state:
        st.session_state.custom_data_loaded = False
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
    st.sidebar.title("Configuration")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        options=["MS MARCO Dataset", "Upload Custom File"],
        help="Use the default MS MARCO dataset or upload your own documents"
    )
    
    queries_df = None
    rag_models = None
    guardrail_checker = None
    
    if data_source == "Upload Custom File":
        st.sidebar.divider()
        st.sidebar.subheader("Upload Your Data")
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload documents",
            type=["txt", "json", "jsonl", "csv", "pdf", "docx"],
            help="Upload a file containing your documents. Supports TXT, JSON, JSONL, CSV, PDF, and DOCX formats."
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load custom data
                with st.spinner("Loading your custom data..."):
                    loader = DatasetLoader()
                    custom_queries_df, custom_passages_df = loader.load_from_file(tmp_file_path)
                    
                    st.sidebar.success(f"Loaded {len(custom_passages_df)} documents")
                    if not custom_queries_df.empty:
                        st.sidebar.info(f"Found {len(custom_queries_df)} queries in file")
                
                # Initialize custom system
                with st.spinner("Initializing RAG system with your data..."):
                    queries_df, rag_models, guardrail_checker = initialize_custom_system(
                        custom_passages_df, custom_queries_df
                    )
                    st.session_state.use_custom_data = True
                    st.session_state.custom_data_loaded = True
                
                # Clean up temp file
                import os
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
                st.sidebar.info("**Supported formats:**\n"
                              "- **TXT**: One document per line or paragraph\n"
                              "- **JSON**: `{\"documents\": [...], \"queries\": [...]}`\n"
                              "- **JSONL**: One JSON object per line with 'text' field\n"
                              "- **CSV**: Must have 'text' column\n"
                              "- **PDF**: Extracts text from all pages\n"
                              "- **DOCX**: Extracts text from paragraphs and tables")
                
                # Fall back to default dataset
                with st.spinner("Initializing with default MS MARCO dataset..."):
                    queries_df, rag_models, guardrail_checker = initialize_system()
                    st.session_state.use_custom_data = False
        else:
            st.sidebar.info("Please upload a file to continue")
            # Initialize with default for now
            with st.spinner("Initializing RAG system..."):
                queries_df, rag_models, guardrail_checker = initialize_system()
    else:
        # Use default MS MARCO dataset
        st.session_state.use_custom_data = False
        with st.spinner("Initializing RAG system..."):
            queries_df, rag_models, guardrail_checker = initialize_system()
    
    # Don't proceed if models aren't initialized
    if rag_models is None:
        st.warning("Please upload a file or select MS MARCO dataset to continue.")
        return
    
    st.sidebar.divider()
    
    # Mode selection
    mode = st.sidebar.radio(
        "Execution Mode",
        options=["Single Model", "Compare Models"],
        help="Choose whether to test one model or compare multiple models"
    )
    
    # Model selection
    if mode == "Single Model":
        selected_config = st.sidebar.selectbox(
            "RAG Configuration",
            options=list(rag_models.keys()),
            help="Choose which RAG configuration to use"
        )
    else:
        selected_configs = st.sidebar.multiselect(
            "Select Models to Compare",
            options=list(rag_models.keys()),
            default=list(rag_models.keys())[:2],
            help="Choose which RAG configurations to compare"
        )
    
    st.sidebar.divider()
    
    # Guardrail toggle
    apply_guardrails = st.sidebar.checkbox(
        "Apply Guardrails",
        value=True,
        help="Enable hallucination guardrails"
    )
    
    # Top-k slider
    top_k = st.sidebar.slider(
        "Number of Chunks to Retrieve",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of context chunks to retrieve"
    )
    
    st.sidebar.divider()
    
    # Sample queries
    st.sidebar.subheader("Sample Queries")
    if st.sidebar.button("Load Random Query", use_container_width=True):
        st.session_state.random_query = queries_df.sample(1)["query"].values[0]
    
    # Query history
    if st.session_state.query_history:
        st.sidebar.subheader("Recent Queries")
        for i, hist_query in enumerate(st.session_state.query_history[-5:][::-1]):
            if st.sidebar.button(f"{hist_query[:50]}...", key=f"history_{i}", use_container_width=True):
                st.session_state.random_query = hist_query
    
    st.sidebar.divider()
    
    # Cost tracker in sidebar
    st.sidebar.subheader("Cost Tracker")
    cost_tracker = get_cost_tracker()
    cost_summary = cost_tracker.get_summary()
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        st.metric("Total Cost", f"${cost_summary['total_cost_usd']:.4f}")
    with col_b:
        st.metric("Budget Left", f"${cost_summary['remaining_usd']:.2f}")
    
    progress_pct = min(cost_summary['total_cost_usd'] / cost_summary['limit_usd'], 1.0)
    st.sidebar.progress(progress_pct)
    
    # Main content area
    st.subheader("Ask a Question")
    
    # Query input
    default_query = st.session_state.get("random_query", "")
    query = st.text_area(
        "Enter your question:",
        value=default_query,
        height=100,
        placeholder="e.g., What is machine learning? How does neural network training work?",
        key="query_input"
    )
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 3])
    
    with col_btn1:
        ask_button = st.button("Get Answer", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_button = st.button("Clear", use_container_width=True)
    
    with col_btn3:
        export_history = st.button("Export History", use_container_width=True)
    
    if clear_button:
        st.session_state.random_query = ""
        st.rerun()
    
    # Export query history
    if export_history and st.session_state.query_history:
        history_data = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(st.session_state.query_history),
            'queries': st.session_state.query_history
        }
        
        st.download_button(
            label="Download History",
            data=json.dumps(history_data, indent=2),
            file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Process query
    if ask_button and query:
        # Add to history
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
        
        st.divider()
        
        if mode == "Single Model":
            # Single model mode
            with st.spinner(f"Processing with {selected_config} configuration..."):
                rag_model = rag_models[selected_config]
                
                start_time = time.time()
                response = rag_model.answer(query, top_k=top_k, apply_guardrails=False)
                
                # Apply guardrails (always check to get confidence level)
                triggered, reason, details = guardrail_checker.check_guardrails(
                    query=query,
                    chunks=response.retrieved_chunks,
                    answer=response.answer,
                )
                
                if apply_guardrails:
                    response.guardrail_triggered = triggered
                    response.guardrail_reason = reason
                
                # Always update metadata with confidence level
                response.metadata.update(details)
                
                end_time = time.time()
            
            # Display results
            display_answer(response)
            
            st.divider()
            
            # Retrieved chunks with guardrail info (always show metadata)
            display_retrieved_chunks(
                response.retrieved_chunks,
                response.metadata
            )
            
            # Metadata
            st.divider()
            st.subheader("Query Metadata")
            
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            
            with meta_col1:
                st.metric("Latency", f"{(end_time - start_time)*1000:.0f} ms")
            
            with meta_col2:
                st.metric("Chunks Retrieved", len(response.retrieved_chunks))
            
            with meta_col3:
                st.metric("Configuration", selected_config)
            
            with meta_col4:
                avg_score = sum(c.score for c in response.retrieved_chunks) / len(response.retrieved_chunks) if response.retrieved_chunks else 0
                st.metric("Avg Retrieval Score", f"{avg_score:.3f}")
        
        else:
            # Compare models mode
            if not selected_configs:
                st.warning("Please select at least one model to compare.")
            else:
                with st.spinner(f"Comparing {len(selected_configs)} models..."):
                    models_to_compare = {name: rag_models[name] for name in selected_configs}
                    results = compare_models(query, models_to_compare, guardrail_checker, top_k, apply_guardrails)
                
                # Display comparison table
                st.subheader("Model Comparison")
                display_comparison_table(results)
                
                st.divider()
                
                # Display each model's results in tabs
                tabs = st.tabs([f"{name}" for name in selected_configs])
                
                for tab, model_name in zip(tabs, selected_configs):
                    with tab:
                        response = results[model_name]['response']
                        latency = results[model_name]['latency']
                        
                        # Answer
                        display_answer(response)
                        
                        st.divider()
                        
                        # Retrieved chunks (always show metadata)
                        display_retrieved_chunks(
                            response.retrieved_chunks,
                            response.metadata
                        )
                        
                        # Metadata
                        st.divider()
                        st.subheader("Metadata")
                        
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        
                        with meta_col1:
                            st.metric("Latency", f"{latency:.0f} ms")
                        
                        with meta_col2:
                            st.metric("Chunks", len(response.retrieved_chunks))
                        
                        with meta_col3:
                            avg_score = sum(c.score for c in response.retrieved_chunks) / len(response.retrieved_chunks) if response.retrieved_chunks else 0
                            st.metric("Avg Score", f"{avg_score:.3f}")


if __name__ == "__main__":
    main()
