"""Ollama LLM configuration for the skincare agent."""

import os

from langchain_ollama import ChatOllama

DEFAULT_MODEL = os.environ.get("SKINCARE_LLM_MODEL", "llama3.2:latest")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(model: str | None = None, temperature: float = 0.3):
    return ChatOllama(
        model=model or DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        temperature=temperature,
    )
