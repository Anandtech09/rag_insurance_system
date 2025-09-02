from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag_system import InsuranceRAGSystem
from src.config import SystemConfig
from evaluation.evaluation_metrics import EvaluationMetrics
import json
import uvicorn
from datetime import datetime
import logging
import os
from src.logging_config import setup_logging

setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
app = FastAPI(title="Insurance RAG System Evaluation")

class TestQuery(BaseModel):
    query: str
    expected_aspects: list[str] = []

# Define request and response models
class TestRequest(BaseModel):
    query: str
    expected_aspects: list[str] = []

class TestResponse(BaseModel):
    evaluation_summary: dict

system = None
metrics = None

@app.on_event("startup")
async def startup_event():
    global system, metrics
    load_dotenv()
    config = SystemConfig.from_env()
    errors = config.validate()
    if errors:
        logger.error("Configuration errors: %s", errors)
        raise HTTPException(status_code=500, detail=f"Configuration errors: {errors}")
    system = InsuranceRAGSystem(config)
    system.setup_system()
    metrics = EvaluationMetrics()
    logger.info("Insurance RAG System initialized for evaluation")

# Updated endpoint to handle a single query
@app.post("/evaluate", response_model=TestResponse)
async def evaluate_system(request: TestRequest):
    try:
        # Process the single query
        result = system.query(request.query)
        evaluation = metrics.evaluate_result(request.query, result, request.expected_aspects)
        
        # Get the evaluation summary
        summary = metrics.get_evaluation_summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"evaluation_report_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return TestResponse(evaluation_summary=summary)
    except Exception as e:
        logger.error("Failed to evaluate system: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)