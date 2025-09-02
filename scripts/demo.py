from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag_system import InsuranceRAGSystem
from src.config import SystemConfig
from src.logging_config import setup_logging
import uvicorn
import logging
import os


setup_logging(logs_dir=os.getenv("LOGS_DIR"), log_level=os.getenv("LOG_LEVEL"))
logger = logging.getLogger(__name__)
app = FastAPI(title="Insurance RAG System Demo")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    query_type: str
    confidence_score: float
    relevant_chunks: list[dict]

class DocumentRequest(BaseModel):
    url: str
    document_title: str

system = None

@app.on_event("startup")
async def startup_event():
    global system
    load_dotenv()
    config = SystemConfig.from_env()
    errors = config.validate()
    if errors:
        logger.error("Configuration errors: %s", errors)
        raise HTTPException(status_code=500, detail=f"Configuration errors: {errors}")
    system = InsuranceRAGSystem(config)
    system.setup_system()
    logger.info("Insurance RAG System initialized")

@app.get("/stats")
async def get_stats():
    try:
        stats = system.get_system_stats()
        return stats
    except Exception as e:
        logger.error("Failed to get system stats: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    try:
        result = system.query(request.query)
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            query_type=result.query_type,
            confidence_score=result.confidence_score,
            relevant_chunks=result.relevant_chunks
        )
    except Exception as e:
        logger.error("Failed to process query: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-document")
async def add_document(request: DocumentRequest):
    try:
        system.add_new_document(request.url, request.document_title)
        return {"message": f"Document {request.document_title} added successfully"}
    except Exception as e:
        logger.error("Failed to add document: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)