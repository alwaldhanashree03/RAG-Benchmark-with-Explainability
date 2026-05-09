"""System installation test - verifies packages, config, env vars, and directories."""

import sys
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    required = [
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
        ("loguru", "loguru"),
        ("openai", "openai"),
        ("chromadb", "chromadb"),
        ("langchain", "langchain"),
        ("streamlit", "streamlit"),
    ]
    failed = []
    for module_name, package_name in required:
        try:
            __import__(module_name)
            print(f"  [OK] {package_name}")
        except ImportError:
            print(f"  [FAIL] {package_name}")
            failed.append(package_name)

    if failed:
        print(f"\nMissing: {', '.join(failed)}. Run: pip install -r requirements.txt")
        return False
    print()
    return True


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from src.utils.config_loader import get_config
        config = get_config()
        for section in ["dataset", "llm", "rag_configs", "evaluation"]:
            val = config.get(section, None)
            status = "[OK]" if val is not None else "[FAIL]"
            print(f"  {status} Section '{section}'")
            if val is None:
                return False
        print()
        return True
    except Exception as e:
        print(f"  [ERROR] {e}\n")
        return False


def test_env():
    """Test environment variables."""
    print("Testing environment variables...")
    import os
    from dotenv import load_dotenv
    load_dotenv()

    keys = {"OPENAI_API_KEY": "Required for LLM/embeddings", "COHERE_API_KEY": "Required for reranker"}
    missing = []
    for var, desc in keys.items():
        val = os.getenv(var)
        if val:
            print(f"  [OK] {var}: {val[:8]}...{val[-4:]}")
        else:
            print(f"  [FAIL] {var}: Not set - {desc}")
            missing.append(var)
    print()
    return len(missing) == 0


def test_directories():
    """Test required directories exist."""
    print("Testing directories...")
    for d in ["data/raw", "data/processed", "data/vector_db", "logs", "results", "configs"]:
        p = Path(d)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"  [CREATED] {d}")
        else:
            print(f"  [OK] {d}")
    print()
    return True


def main():
    print("=" * 60)
    print("RAG Benchmark - Installation Test")
    print("=" * 60 + "\n")

    results = []
    for name, fn in [("Imports", test_imports), ("Config", test_config),
                     ("Env vars", test_env), ("Directories", test_directories)]:
        try:
            results.append((name, fn()))
        except Exception as e:
            print(f"  [ERROR] {name}: {e}\n")
            results.append((name, False))

    print("=" * 60)
    for name, passed in results:
        print(f"{'[PASS]' if passed else '[FAIL]'} {name}")

    if all(r for _, r in results):
        print("\nAll tests passed. Run: python main.py prepare-data")
        return 0
    else:
        print("\nSome tests failed. Fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
