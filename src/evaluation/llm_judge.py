"""LLM-as-Judge evaluation using GPT-4.

Industry best practice: Use powerful LLM to evaluate answer quality.
Reference: Judging LLM-as-a-Judge (Zheng et al., 2023)
"""

from typing import Dict, List, Optional
import json
import re
from loguru import logger

from src.models.base_rag import RAGResponse
from src.models.llm_client import LLMClient
from src.utils.config_loader import get_config


class LLMJudge:
    """Use GPT-4 as a judge to evaluate RAG answer quality.
    
    Best practice: LLM-as-Judge provides nuanced evaluation
    that correlates well with human judgment.
    
    Reference: 
    - Judging LLM-as-a-Judge with MT-Bench (Zheng et al., 2023)
    - G-Eval: NLG Evaluation using GPT-4 (Liu et al., 2023)
    """

    def __init__(self, judge_model: str = "gpt-4o-mini"):
        """Initialize LLM judge.
        
        Args:
            judge_model: Model to use for judging (gpt-4, gpt-4o-mini)
        """
        self.config = get_config()
        self.judge_model = judge_model
        
        # Create separate LLM client for judging
        self.llm_client = LLMClient()
        # Override model for judging
        self.llm_client.model = judge_model
        
        logger.info(f"LLMJudge initialized with {judge_model}")

    def evaluate_answer_quality(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate answer quality across multiple dimensions.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_contexts: Retrieved context chunks
            ground_truth: Ground truth answer (optional)
            
        Returns:
            Dictionary with scores for:
            - relevance: How relevant is answer to query (0-5)
            - coherence: How coherent/well-written is answer (0-5)
            - correctness: How correct is answer (0-5, requires ground_truth)
            - completeness: How complete is answer (0-5)
            - groundedness: How grounded in context (0-5)
        """
        scores = {}
        
        # Evaluate relevance
        scores["judge_relevance"] = self._evaluate_relevance(query, answer)
        
        # Evaluate coherence
        scores["judge_coherence"] = self._evaluate_coherence(answer)
        
        # Evaluate groundedness
        scores["judge_groundedness"] = self._evaluate_groundedness(
            answer, retrieved_contexts
        )
        
        # Evaluate completeness
        scores["judge_completeness"] = self._evaluate_completeness(
            query, answer, retrieved_contexts
        )
        
        # Evaluate correctness (if ground truth available)
        if ground_truth:
            scores["judge_correctness"] = self._evaluate_correctness(
                answer, ground_truth
            )
        
        # Compute overall score (average)
        score_values = [v for v in scores.values() if v > 0]
        scores["judge_overall"] = sum(score_values) / len(score_values) if score_values else 0.0
        
        logger.debug(f"LLM Judge scores: {scores}")
        return scores

    def _evaluate_relevance(self, query: str, answer: str) -> float:
        """Evaluate answer relevance to query.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            Relevance score (0-5, normalized to 0-1)
        """
        prompt = f"""Evaluate the relevance of the answer to the question on a scale of 0-5.

Question: {query}

Answer: {answer}

Scoring criteria:
- 5: Perfectly relevant, directly addresses the question
- 4: Highly relevant, addresses most aspects of the question
- 3: Moderately relevant, addresses some aspects
- 2: Somewhat relevant, tangentially related
- 1: Barely relevant, mostly off-topic
- 0: Not relevant at all

Provide ONLY a number from 0-5 as your response."""

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            score = self._extract_score(response)
            return score / 5.0  # Normalize to 0-1
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return 0.0

    def _evaluate_coherence(self, answer: str) -> float:
        """Evaluate answer coherence and fluency.
        
        Args:
            answer: Generated answer
            
        Returns:
            Coherence score (0-5, normalized to 0-1)
        """
        prompt = f"""Evaluate the coherence and fluency of this text on a scale of 0-5.

Text: {answer}

Scoring criteria:
- 5: Perfectly coherent, fluent, well-structured
- 4: Highly coherent, minor issues
- 3: Moderately coherent, some awkward phrasing
- 2: Somewhat coherent, multiple issues
- 1: Barely coherent, hard to follow
- 0: Incoherent, doesn't make sense

Provide ONLY a number from 0-5 as your response."""

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            score = self._extract_score(response)
            return score / 5.0
        except Exception as e:
            logger.error(f"Coherence evaluation failed: {e}")
            return 0.0

    def _evaluate_groundedness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Evaluate if answer is grounded in provided contexts.
        
        Args:
            answer: Generated answer
            contexts: Retrieved context chunks
            
        Returns:
            Groundedness score (0-5, normalized to 0-1)
        """
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Evaluate how well the answer is grounded in the provided contexts on a scale of 0-5.

Contexts:
{context_text}

Answer: {answer}

Scoring criteria:
- 5: Fully grounded, all claims supported by contexts
- 4: Mostly grounded, minor unsupported details
- 3: Partially grounded, some claims not supported
- 2: Weakly grounded, many unsupported claims
- 1: Barely grounded, mostly unsupported
- 0: Not grounded, contradicts or ignores contexts

Provide ONLY a number from 0-5 as your response."""

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            score = self._extract_score(response)
            return score / 5.0
        except Exception as e:
            logger.error(f"Groundedness evaluation failed: {e}")
            return 0.0

    def _evaluate_completeness(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Evaluate answer completeness.
        
        Args:
            query: User query
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            Completeness score (0-5, normalized to 0-1)
        """
        prompt = f"""Evaluate the completeness of the answer on a scale of 0-5.

Question: {query}

Answer: {answer}

Scoring criteria:
- 5: Fully complete, addresses all aspects
- 4: Mostly complete, minor aspects missing
- 3: Partially complete, some key aspects missing
- 2: Incomplete, many aspects missing
- 1: Very incomplete, barely addresses question
- 0: Does not address the question

Provide ONLY a number from 0-5 as your response."""

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            score = self._extract_score(response)
            return score / 5.0
        except Exception as e:
            logger.error(f"Completeness evaluation failed: {e}")
            return 0.0

    def _evaluate_correctness(
        self,
        answer: str,
        ground_truth: str,
    ) -> float:
        """Evaluate answer correctness against ground truth.
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            
        Returns:
            Correctness score (0-5, normalized to 0-1)
        """
        prompt = f"""Compare the answer to the ground truth and evaluate correctness on a scale of 0-5.

Generated Answer: {answer}

Ground Truth: {ground_truth}

Scoring criteria:
- 5: Completely correct, semantically equivalent
- 4: Mostly correct, minor inaccuracies
- 3: Partially correct, some errors
- 2: Somewhat correct, significant errors
- 1: Mostly incorrect, only minor correct parts
- 0: Completely incorrect or contradictory

Provide ONLY a number from 0-5 as your response."""

        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            score = self._extract_score(response)
            return score / 5.0
        except Exception as e:
            logger.error(f"Correctness evaluation failed: {e}")
            return 0.0

    def _extract_score(self, response: str) -> float:
        """Extract numeric score from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted score (0-5)
        """
        # Try to find a number in the response
        numbers = re.findall(r'\b[0-5](?:\.[0-9]+)?\b', response)
        
        if numbers:
            try:
                score = float(numbers[0])
                # Clamp to 0-5 range
                return max(0.0, min(5.0, score))
            except ValueError:
                pass
        
        logger.warning(f"Could not extract score from: {response}")
        return 0.0

    def evaluate_with_explanation(
        self,
        query: str,
        answer: str,
        retrieved_contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, any]:
        """Evaluate with detailed explanations.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_contexts: Retrieved contexts
            ground_truth: Ground truth answer (optional)
            
        Returns:
            Dictionary with scores and explanations
        """
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        gt_section = ""
        if ground_truth:
            gt_section = f"\nGround Truth Answer: {ground_truth}"
        
        prompt = f"""Evaluate this RAG system answer across multiple dimensions and provide scores with explanations.

Question: {query}

Answer: {answer}

Retrieved Contexts:
{context_text}{gt_section}

Please evaluate the answer on these dimensions (0-5 scale each):
1. Relevance: How relevant to the question?
2. Coherence: How well-written and coherent?
3. Groundedness: How well supported by contexts?
4. Completeness: How complete is the answer?
{f"5. Correctness: How correct compared to ground truth?" if ground_truth else ""}

Provide your evaluation in this JSON format:
{{
  "relevance": {{"score": X, "explanation": "..."}},
  "coherence": {{"score": X, "explanation": "..."}},
  "groundedness": {{"score": X, "explanation": "..."}},
  "completeness": {{"score": X, "explanation": "..."}}{"," if ground_truth else ""}
  {"\"correctness\": {\"score\": X, \"explanation\": \"...\"}}" if ground_truth else ""}
}}"""

        try:
            response = self.llm_client.generate(prompt, max_tokens=500)
            
            # Try to parse JSON
            result = json.loads(response)
            
            # Normalize scores to 0-1
            for key in result:
                if isinstance(result[key], dict) and "score" in result[key]:
                    result[key]["score"] = result[key]["score"] / 5.0
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation with explanation failed: {e}")
            return {}


def get_llm_judge(judge_model: str = "gpt-4o-mini") -> LLMJudge:
    """Get LLM judge instance.
    
    Args:
        judge_model: Model to use for judging
        
    Returns:
        LLMJudge instance
    """
    return LLMJudge(judge_model=judge_model)
