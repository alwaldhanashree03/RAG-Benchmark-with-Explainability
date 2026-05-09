# Quick Start Guide - No Training Required!

## What is RAG?

RAG (Retrieval-Augmented Generation) retrieves information from existing documents and uses a pre-trained language model (like GPT) to generate answers. **You don't train anything** - you just:
1. Load documents into a vector database
2. Query the system
3. Get answers based on your documents

## Prerequisites

### 1. Install Python
- Python 3.9 or higher
- Check: `python --version`

### 2. Get API Keys (Required)

**OpenAI API Key** (Required):
- Sign up at https://platform.openai.com/
- Go to API Keys section
- Create new key
- Copy the key (starts with `sk-`)

**Cohere API Key** (Optional - only for Reranker model):
- Sign up at https://cohere.com/
- Get free API key from dashboard

## Installation (5 minutes)

### Step 1: Clone the Repository
```bash
git clone https://github.com/KonetiBalaji/RAG-Benchmark-with-Explainability.git
cd RAG-Benchmark-with-Explainability
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs all necessary libraries (no training required).

### Step 4: Configure API Keys

Create a `.env` file:
```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Edit `.env` and add your keys:
```
OPENAI_API_KEY=sk-your_actual_openai_key_here
COHERE_API_KEY=your_cohere_key_here
```

## Running the System (3 Options)

### Option 1: Interactive UI (Recommended for First Time)

**Start the Streamlit UI:**
```bash
streamlit run src/ui/app.py
```

**What happens:**
1. Browser opens at http://localhost:8501
2. System loads MS MARCO dataset (10,000 documents)
3. Builds vector index (first time only - takes 2-3 minutes)
4. Ready to answer questions!

**Try these queries:**
- "What is machine learning?"
- "What was the Manhattan Project?"
- "How does neural network training work?"

**Or upload your own documents:**
1. Click "Upload Custom File" in sidebar
2. Upload PDF, DOCX, TXT, JSON, or CSV
3. System processes it automatically
4. Ask questions about YOUR documents!

### Option 2: Command Line (Quick Test)

**Run a quick test:**
```bash
python main.py quick-test
```

**What happens:**
- Loads small dataset
- Tests all 4 RAG models
- Generates comparison report
- Exports results to Excel

### Option 3: REST API (For Integration)

**Start the API server:**
```bash
python main.py api
# Or directly: uvicorn src.api.main:app --reload
```

**Test the API:**
```bash
# Check health
curl http://localhost:8000/health

# Query the system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "config": "baseline",
    "top_k": 3
  }'
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## First Time Setup (Detailed)

### What Happens During First Run?

**1. Dataset Loading (30 seconds)**
```
Loading MS MARCO dataset from HuggingFace...
- Downloads 10,000 passages
- Downloads 500 test queries
- Saves to data/raw/
```

**2. Text Chunking (10 seconds)**
```
Splitting documents into chunks...
- Each chunk: 512 tokens
- Overlap: 50 tokens
- Creates ~15,000 chunks
```

**3. Embedding Generation (2-3 minutes)**
```
Generating embeddings with OpenAI...
- Model: text-embedding-3-small
- Dimensions: 1536
- Batch size: 100
- Cost: ~$0.10
```

**4. Vector Index Building (30 seconds)**
```
Building ChromaDB vector index...
- Stores in data/vector_db/
- Creates HNSW index
- Persisted for future runs
```

**5. System Ready!**
```
All RAG models initialized
Ready to answer questions
```

**Subsequent runs:** Only takes 5-10 seconds (loads cached index)

## Understanding the RAG Models

You have 4 pre-configured RAG strategies (no training needed):

### 1. Baseline RAG
- Simple semantic search
- Fast and reliable
- **Use for:** General questions

### 2. Hybrid RAG
- Combines keyword (BM25) + semantic search
- Best retrieval quality
- **Use for:** Technical terms, specific keywords

### 3. Reranker RAG
- Uses Cohere to rerank results
- Highest accuracy
- **Use for:** Complex questions, best quality

### 4. Query Decomposition RAG
- Breaks complex questions into sub-questions
- Comprehensive answers
- **Use for:** Multi-part questions

## Common Questions

### Q: Do I need to train models?
**A: No!** RAG uses pre-trained models (GPT-3.5, OpenAI embeddings). You just load your documents.

### Q: How much does it cost?
**A:**
- Embedding generation: ~$0.10 for 10,000 documents (one-time)
- Each query: ~$0.001-0.002
- First run total: ~$0.15
- Budget limit: $10 (configurable in config.yaml)

### Q: What if I don't have MS MARCO dataset?
**A:** It downloads automatically from HuggingFace on first run (free).

### Q: Can I use my own documents?
**A:** Yes! Upload via UI:
- PDF (research papers, reports)
- DOCX (business docs)
- TXT (plain text)
- JSON, CSV (structured data)

### Q: How long does setup take?
**A:**
- Installation: 5 minutes
- First run: 3-4 minutes
- Subsequent runs: 5-10 seconds

### Q: What if it's slow?
**A:**
- First run is slow (building index)
- Cached runs are fast
- Reduce `num_passages` in config.yaml for faster setup

### Q: Do I need a GPU?
**A:** No! Everything runs on CPU. Embeddings/LLM calls use OpenAI API.

### Q: What if I get API errors?
**A:**
- Check API keys in .env
- Check internet connection
- Check OpenAI account has credits
- Check rate limits (60 req/min configured)

## Step-by-Step First Run

```bash
# 1. Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 2. Verify API key is set
python -c "import os; print('OpenAI key:', os.getenv('OPENAI_API_KEY')[:10] + '...')"

# 3. Start the UI
streamlit run src/ui/app.py

# 4. Wait for initialization (2-3 minutes first time)
# You'll see:
# - Loading dataset...
# - Chunking documents...
# - Generating embeddings...
# - Building vector store...
# - System ready!

# 5. Ask your first question!
# Type in text area: "What is machine learning?"
# Click "Get Answer"
# See retrieved evidence and generated answer
```

## Troubleshooting

### "OpenAI API key not found"
```bash
# Create .env file with your key
echo OPENAI_API_KEY=sk-your_key_here > .env
```

### "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### "Vector store is empty"
```bash
# Build the index first
python main.py build-index
```

### Out of memory
```bash
# Edit configs/config.yaml
# Reduce num_passages: 1000
# Reduce batch_size: 50
```

### API rate limit errors
```bash
# System has built-in rate limiting (60 req/min)
# Wait a minute and retry
# Or reduce batch_size in config.yaml
```

## What You Get

After setup, you can:
- Ask questions about 10,000 documents
- Upload your own documents (PDF, DOCX, etc.)
- Compare 4 different RAG strategies
- See confidence scores and guardrails
- Export results for analysis
- Use via UI, CLI, or REST API

## Next Steps

1. **Try the UI** - Most intuitive way to explore
2. **Upload custom documents** - Test on your own data
3. **Compare models** - See which RAG strategy works best
4. **Check the API** - Integrate into your apps
5. **Read README.md** - Detailed documentation

## No Training, Just Retrieval!

Remember: RAG doesn't train models. It:
1. Stores your documents as vectors (one-time indexing)
2. Retrieves relevant chunks when you query
3. Uses GPT to generate answers from those chunks

**That's it! No training, fine-tuning, or model updates needed.**

---

**Need help?** Check the full README.md or open an issue on GitHub.
