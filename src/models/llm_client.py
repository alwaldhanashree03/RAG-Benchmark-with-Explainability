"""LLM client for text generation using OpenAI API.

Industry best practice: Centralized LLM interface with retry logic and cost tracking.
"""

import time
from typing import List, Optional

from openai import OpenAI
from loguru import logger

from src.utils.config_loader import get_config
from src.utils.cost_tracker import get_cost_tracker


class LLMClient:
    """Client for LLM text generation.
    
    Best practices:
    - Deterministic generation (temperature=0)
    - Cost tracking
    - Retry logic with exponential backoff
    - Token usage monitoring
    
    Reference: OpenAI Chat Completions API
    https://platform.openai.com/docs/guides/chat-completions
    """

    def __init__(self):
        """Initialize LLM client."""
        self.config = get_config()
        self.cost_tracker = get_cost_tracker()
        
        # Get API key and initialize client
        import os
        api_key = self.config.get_api_key("openai")
        # Initialize client without organization to avoid mismatch errors
        # Remove OPENAI_ORG_ID from environment if set
        os.environ.pop('OPENAI_ORG_ID', None)
        self.client = OpenAI(api_key=api_key)
        
        # Configuration
        self.model = self.config.get("llm.model", "gpt-3.5-turbo")
        self.temperature = self.config.get("llm.temperature", 0.0)
        self.max_tokens = self.config.get("llm.max_tokens", 512)
        self.seed = self.config.get("llm.seed", 42)
        self.max_retries = self.config.get("rate_limiting.max_retries", 3)
        
        logger.info(
            f"LLMClient initialized: model={self.model}, "
            f"temp={self.temperature}, max_tokens={self.max_tokens}"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text completion.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Optional max tokens override
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Generate with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    seed=self.seed,
                )
                
                # Extract generated text
                generated_text = response.choices[0].message.content
                
                # Track cost
                total_tokens = response.usage.total_tokens
                self.cost_tracker.add_entry(
                    service="llm",
                    operation="completion",
                    tokens=total_tokens,
                    model=self.model,
                    metadata={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                )
                
                return generated_text
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"LLM generation failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM generation failed after {self.max_retries} attempts: {e}")
                    raise

    def generate_rag_answer(
        self,
        query: str,
        context_chunks: List[str],
    ) -> str:
        """Generate answer using RAG pattern.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Generated answer
            
        Best practice: Use clear instructions and structured context
        """
        # Build context from chunks
        context = "\n\n".join([
            f"[{i+1}] {chunk}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # System prompt for RAG
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Follow these rules:\n"
            "1. Answer ONLY using information from the context\n"
            "2. If the context doesn't contain enough information, say so clearly\n"
            "3. Be concise and accurate\n"
            "4. Cite which context piece(s) you used (e.g., 'According to [1]...')"
        )
        
        # User prompt with query and context
        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        return self.generate(user_prompt, system_prompt)
