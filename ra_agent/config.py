from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    azure_openai_api_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: Optional[str] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    edgar_identity: str = os.getenv("EDGAR_IDENTITY", "Your Name your@email.com")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "logs")
    data_dir: str = os.getenv("DATA_DIR", "data")
    outputs_dir: str = os.getenv("OUTPUTS_DIR", "outputs")


settings = Settings()

# Ensure base directories exist
os.makedirs(settings.log_dir, exist_ok=True)
os.makedirs(settings.data_dir, exist_ok=True)
os.makedirs(settings.outputs_dir, exist_ok=True)
