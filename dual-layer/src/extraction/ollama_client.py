import os
import requests
import json
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Optional

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest")

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))
def generate(
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    model: Optional[str] = None
) -> str:
    """Call Ollama generate endpoint with retry logic"""
    url = f"{BASE_URL}/api/generate"
    payload = {
        "model": model or MODEL,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens
        },
        "stream": False
    }
    
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["response"]

def health_check() -> bool:
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

