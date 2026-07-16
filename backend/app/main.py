import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.routes import router

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="Hybrid Dual-LLM RAG with Semantic Caching",
    description="Resume-grade backend featuring local Qdrant memory, local semantic caching, Groq inference and Gemini evaluation.",
    version="1.0.0"
)

# Enable CORS for local React + Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, open to all. In production restrict to frontend domain.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Dual-LLM RAG Backend Gateway is active.",
        "config": {
            "simulation_mode": settings.is_simulation_mode,
            "cache_threshold": settings.SEMANTIC_CACHE_THRESHOLD,
            "evaluation_rate": settings.EVALUATION_RATE
        }
    }

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Initializing Hybrid Dual-LLM & Semantic Cache Server...")
    if settings.is_simulation_mode:
        logger.warning("WARNING: Running in SIMULATION MODE. API keys for Groq/Gemini are missing.")
        logger.warning("Fidelity responses and evaluators will be simulated.")
    else:
        logger.info("SUCCESS: Running in PRODUCTION MODE with loaded API Keys.")
        logger.info(f"Groq API Key: {'Loaded' if settings.GROQ_API_KEY else 'Missing'}")
        logger.info(f"Gemini API Key: {'Loaded' if settings.GEMINI_API_KEY else 'Missing'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
