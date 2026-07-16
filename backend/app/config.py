import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # Model settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Semantic Cache
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.85))
    
    # Evaluation settings
    EVALUATION_RATE: float = float(os.getenv("EVALUATION_RATE", 1.0))
    
    # Paths
    DATA_DIR: Path = BASE_DIR / "data"

    @property
    def is_simulation_mode(self) -> bool:
        """Returns True if any necessary API key is missing, enabling mocks."""
        return not (self.GEMINI_API_KEY and self.GROQ_API_KEY)

settings = Settings()

# Ensure data directory exists
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
