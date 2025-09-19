"""Perplexity API client for code analysis."""

import os
import requests
from typing import Optional


class PerplexityClient:
    """Simple Perplexity API client."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai"
        
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable required")
    
    def analyze_code(self, code_content: str, file_type: str) -> str:
        """Analyze code content and return insights."""
        prompt = f"""
        Analyze this {file_type} code and provide:
        1. Main purpose of the code
        2. Key components (classes, functions, imports)
        3. Code quality assessment (1-5 scale)
        4. Any notable patterns or issues
        
        Code:
        ```{file_type}
        {code_content[:2000]}  # Limit to first 2000 chars
        ```
        
        Provide a concise analysis in JSON format.
        """
        
        return self._make_request(prompt)
    
    def analyze_project_structure(self, file_summaries: list) -> str:
        """Analyze overall project structure and identify research topics."""
        prompt = f"""
        Analyze this project structure and identify:
        1. Research topics (transformers, clustering, explainable AI, etc.)
        2. Framework usage (TensorFlow, PyTorch, scikit-learn, etc.)
        3. Project type (research, production, prototype)
        4. Estimated development timeline based on complexity
        
        File summaries:
        {file_summaries}
        
        Provide insights in structured format.
        """
        
        return self._make_request(prompt)
    
    def _make_request(self, prompt: str) -> str:
        """Make request to Perplexity API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Request failed: {str(e)}"