import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# 1. Load environment variables first
load_dotenv()

from app.api.app_router import backend_router
from app.core.config import Settings

# 2. Configure Production Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ecu_rag_system")

# 3. Initialize Global Settings instance
settings = Settings()

# 4. Lifespan Management (Startup and Shutdown events)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logic to run on startup
    logger.info("Initializing ECU RAG SYSTEM...")
    try:
        # Validate critical settings here if needed (e.g., database connections)
        logger.info(f"Allowed CORS Origin: {settings.FRONTEND_URL}")
    except Exception as e:
        logger.error(f"Startup validation failed: {str(e)}")
        raise e
        
    yield
    
    # Logic to run on shutdown
    logger.info("Shutting down ECU RAG SYSTEM...")

# 5. Initialize FastAPI App
app = FastAPI(
    title="ECU RAG SYSTEM",
    description="Upload Files and Manage RAG Workflows",
    version="1.0.0",
    lifespan=lifespan
)

# 6. Global Exception Handler (Bulletproof Error Handling)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches any unhandled exceptions, logs the full error on the server side,
    and returns a clean, secure 500 JSON response to the client.
    """
    logger.error(f"Unhandled exception occurred on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again later.",
            "error_type": exc.__class__.__name__
        }
    )

# 7. CORS Middleware Configuration
origins = [settings.FRONTEND_URL] if settings.FRONTEND_URL else ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],    
)

# 8. Register Routes
app.include_router(backend_router)


# 9. Core Endpoints
@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to the ECU RAG SYSTEM API"}


@app.get("/health", tags=["Health"])
async def health():
    """
    Health check endpoint for monitoring tools (e.g., AWS, Kubernetes, UptimeRobot).
    """
    return {
        "status": "healthy",
        "version": app.version
    }


# 10. Local Development Execution
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting local development server...")
    try:
        uvicorn.run(
            "main:app",  # Assumes this file is named main.py
            host="127.0.0.1", 
            port=8000, 
            reload=True
        )
    except KeyboardInterrupt:
        logger.info("Server manually stopped by user.")
    except Exception as e:
        logger.critical(f"Server failed to start: {e}")