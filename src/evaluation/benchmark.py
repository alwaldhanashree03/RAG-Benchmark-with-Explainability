"""Benchmarking pipeline with statistical analysis.

Precomputes query embeddings once and shares them across all RAG configs
via the EmbeddingGenerator's module-level cache, eliminating redundant
OpenAI API calls.

Reference: Paired t-test for comparing RAG configurations
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger
from tqdm import tqdm

from src.models.base_rag import BaseRAG, RAGResponse
from src.evaluation.metrics import RAGMetrics, aggregate_metrics
from src.guardrails.guardrail_checker import GuardrailChecker
from src.data.embedding_generator import EmbeddingGenerator, get_cache_stats
from src.utils.config_loader import get_config
from src.utils.cost_tracker import get_cost_tracker


class RAGBenchmark:
    """Benchmark multiple RAG configurations.

    Key optimization: query embeddings are precomputed in a single batched
    API call and cached in-memory. All RAG configs then hit the cache
    instead of making redundant API calls.

    Reference: Experimental design for IR systems evaluation
    """

    def __init__(self):
        """Initialize benchmark."""
        self.config = get_config()
        self.metrics_calculator = RAGMetrics()
        self.guardrail_checker = GuardrailChecker()
        self.cost_tracker = get_cost_tracker()

        logger.info("RAGBenchmark initialized")

    def _precompute_query_embeddings(self, queries: List[Dict[str, Any]]) -> None:
        """Precompute and cache embeddings for all query texts.

        After this call, every subsequent ``generate_embedding(query_text)``
        call inside any RAG model will be a cache hit (free, instant).

        Args:
            queries: List of query dicts containing a ``query`` key.
        """
        query_texts = [q["query"] for q in queries]
        unique_texts = list(set(query_texts))

        logger.info(
            f"Precomputing embeddings for {len(unique_texts)} unique queries "
            f"(from {len(query_texts)} total)..."
        )

        emb_gen = EmbeddingGenerator()
        emb_gen.generate_embeddings(unique_texts, show_progress=True)

        stats = get_cache_stats()
        logger.info(f"Embedding cache warmed: {stats['cached_embeddings']} entries")

    def run_benchmark(
        self,
        rag_configs: Dict[str, BaseRAG],
        queries: List[Dict[str, Any]],
        apply_guardrails: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run benchmark across multiple RAG configurations.

        Args:
            rag_configs: Dictionary mapping config name to RAG instance
            queries: List of query dictionaries with 'query', 'relevant_ids', 'answer'
            apply_guardrails: Whether to apply guardrails
            output_dir: Directory to save results

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Starting benchmark: {len(rag_configs)} configs, {len(queries)} queries")

        # Precompute all query embeddings once (single batched API call)
        self._precompute_query_embeddings(queries)

        all_results = {}

        for config_name, rag_model in rag_configs.items():
            logger.info(f"Benchmarking configuration: {config_name}")

            config_results = self._benchmark_single_config(
                rag_model,
                queries,
                apply_guardrails,
            )

            all_results[config_name] = config_results

        # Aggregate and analyze results
        summary = self._analyze_results(all_results)

        # Save results if output directory provided
        if output_dir:
            self._save_results(all_results, summary, output_dir)

        logger.info("Benchmark complete")

        return {
            "detailed_results": all_results,
            "summary": summary,
            "cost_summary": self.cost_tracker.get_summary(),
        }

    def _benchmark_single_config(
        self,
        rag_model: BaseRAG,
        queries: List[Dict[str, Any]],
        apply_guardrails: bool,
    ) -> Dict[str, Any]:
        """Benchmark a single RAG configuration.

        Args:
            rag_model: RAG model instance
            queries: List of queries
            apply_guardrails: Whether to apply guardrails

        Returns:
            Configuration results
        """
        query_results = []
        errors = 0

        for query_data in tqdm(queries, desc=f"  {rag_model.name}", unit="q"):
            query = query_data["query"]
            relevant_ids = query_data.get("relevant_passage_ids", [])
            reference_answer = query_data.get("answers", [""])[0]

            # Run RAG with timing
            start_time = time.time()

            try:
                response = rag_model.answer(
                    query=query,
                    top_k=3,
                    apply_guardrails=apply_guardrails,
                )

                # Apply guardrails if not already done
                if apply_guardrails and not response.guardrail_triggered:
                    triggered, reason, details = self.guardrail_checker.check_guardrails(
                        query=query,
                        chunks=response.retrieved_chunks,
                        answer=response.answer,
                    )

                    if triggered:
                        response.guardrail_triggered = True
                        response.guardrail_reason = reason
                        response.metadata.update(details)

                end_time = time.time()

                # Evaluate metrics
                metrics = self.metrics_calculator.evaluate_complete(
                    response=response,
                    relevant_ids=relevant_ids if relevant_ids else None,
                    reference_answer=reference_answer if reference_answer else None,
                    start_time=start_time,
                    end_time=end_time,
                )

                query_results.append(metrics)

            except Exception as e:
                errors += 1
                if errors <= 3:
                    logger.error(f"Error processing query '{query[:80]}': {e}")
                elif errors == 4:
                    logger.error("Suppressing further per-query errors...")
                query_results.append({
                    "query": query,
                    "error": str(e),
                })

        if errors:
            logger.warning(f"{rag_model.name}: {errors}/{len(queries)} queries failed")

        # Aggregate metrics
        aggregated = aggregate_metrics(query_results)

        return {
            "query_results": query_results,
            "aggregated_metrics": aggregated,
        }

    def _analyze_results(
        self,
        all_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze results across configurations.

        Args:
            all_results: Results from all configurations

        Returns:
            Analysis summary including statistical tests
        """
        summary = {
            "configurations": list(all_results.keys()),
            "aggregated_metrics": {},
            "statistical_tests": {},
        }

        # Collect aggregated metrics for each config
        for config_name, results in all_results.items():
            summary["aggregated_metrics"][config_name] = results["aggregated_metrics"]

        # Statistical significance tests
        summary["statistical_tests"] = self._perform_statistical_tests(all_results)

        # Identify best configuration for each metric
        summary["best_configs"] = self._identify_best_configs(all_results)

        return summary

    def _perform_statistical_tests(
        self,
        all_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform paired t-tests between configurations.

        Args:
            all_results: Results from all configurations

        Returns:
            Statistical test results
        """
        tests = {}

        config_names = list(all_results.keys())
        if len(config_names) < 2:
            return tests

        # Get baseline (first config)
        baseline_name = config_names[0]
        baseline_results = all_results[baseline_name]["query_results"]

        # Compare each config to baseline
        for config_name in config_names[1:]:
            config_results = all_results[config_name]["query_results"]

            # Compare key metrics
            for metric in ["precision@3", "rouge_l_f1", "faithfulness", "latency_ms"]:
                baseline_values = [
                    r[metric] for r in baseline_results
                    if metric in r and isinstance(r[metric], (int, float))
                ]
                config_values = [
                    r[metric] for r in config_results
                    if metric in r and isinstance(r[metric], (int, float))
                ]

                if len(baseline_values) == len(config_values) and len(baseline_values) > 1:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(baseline_values, config_values)

                    comparison_key = f"{baseline_name}_vs_{config_name}_{metric}"
                    tests[comparison_key] = {
                        "metric": metric,
                        "baseline": baseline_name,
                        "comparison": config_name,
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                        "baseline_mean": np.mean(baseline_values),
                        "comparison_mean": np.mean(config_values),
                        "improvement": np.mean(config_values) - np.mean(baseline_values),
                    }

        return tests

    def _identify_best_configs(
        self,
        all_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """Identify best configuration for each metric.

        Args:
            all_results: Results from all configurations

        Returns:
            Dictionary mapping metric to best config name
        """
        best_configs = {}

        # Metrics to optimize (higher is better)
        maximize_metrics = [
            "precision@3_mean", "mrr_mean", "rouge_l_f1_mean",
            "faithfulness_mean", "confidence_score_mean"
        ]

        # Metrics to minimize (lower is better)
        minimize_metrics = ["latency_ms_mean"]

        for metric in maximize_metrics + minimize_metrics:
            best_config = None
            best_value = None

            for config_name, results in all_results.items():
                aggregated = results["aggregated_metrics"]

                if metric in aggregated:
                    value = aggregated[metric]

                    if best_value is None:
                        best_config = config_name
                        best_value = value
                    elif metric in maximize_metrics and value > best_value:
                        best_config = config_name
                        best_value = value
                    elif metric in minimize_metrics and value < best_value:
                        best_config = config_name
                        best_value = value

            if best_config:
                best_configs[metric] = {
                    "config": best_config,
                    "value": best_value,
                }

        return best_configs

    def _save_results(
        self,
        all_results: Dict[str, Any],
        summary: Dict[str, Any],
        output_dir: Path,
    ) -> None:
        """Save benchmark results to files.

        Args:
            all_results: Detailed results
            summary: Summary analysis
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results for each config
        for config_name, results in all_results.items():
            config_file = output_dir / f"{config_name.replace(' ', '_')}_results.csv"
            df = pd.DataFrame(results["query_results"])
            df.to_csv(config_file, index=False)
            logger.info(f"Saved detailed results: {config_file}")

        # Save aggregated metrics
        aggregated_df = pd.DataFrame(summary["aggregated_metrics"]).T
        aggregated_file = output_dir / "aggregated_metrics.csv"
        aggregated_df.to_csv(aggregated_file)
        logger.info(f"Saved aggregated metrics: {aggregated_file}")

        # Save statistical tests
        if summary["statistical_tests"]:
            tests_df = pd.DataFrame(summary["statistical_tests"]).T
            tests_file = output_dir / "statistical_tests.csv"
            tests_df.to_csv(tests_file)
            logger.info(f"Saved statistical tests: {tests_file}")

        # Save best configs
        best_df = pd.DataFrame(summary["best_configs"]).T
        best_file = output_dir / "best_configs.csv"
        best_df.to_csv(best_file)
        logger.info(f"Saved best configs: {best_file}")
