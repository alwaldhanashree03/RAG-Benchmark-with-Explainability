"""Setup script for RAG benchmark package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="rag-benchmark",
    version="1.0.0",
    description="RAG System Benchmark with Explainability and Hallucination Guardrails",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MalikaReddy Eppa, DhanaSree",
    author_email="",
    url="https://github.com/KonetiBalaji/RAG-Benchmark-with-Explainability",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.12.0",
        "langchain>=0.1.6",
        "chromadb>=0.4.22",
        "streamlit>=1.31.0",
        "transformers>=4.37.2",
        "torch>=2.2.0",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "scipy>=1.12.0",
        "cohere>=4.47",
        "rank-bm25>=0.2.2",
        "ragas>=0.1.5",
        "rouge-score>=0.1.2",
        "loguru>=0.7.2",
        "python-dotenv>=1.0.1",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.1",
        "plotly>=5.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-benchmark=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="rag retrieval-augmented-generation llm nlp benchmarking explainability",
)
