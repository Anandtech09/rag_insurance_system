from dotenv import load_dotenv
from src.rag_system import InsuranceRAGSystem
from src.config import SystemConfig
from src.logging_config import setup_logging
import logging, os

logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    config = SystemConfig.from_env()
    errors = config.validate()
    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"- {error}")
        return

    logs_dir = os.getenv("LOGS_DIR")
    log_level=os.getenv("LOG_LEVEL")
    setup_logging(logs_dir=logs_dir, log_level=log_level)

    logger.info("Setting up Insurance RAG System for chunking and storage...")
    system = InsuranceRAGSystem(config)
    system.setup_system()

    stats = system.get_system_stats()
    logger.info("System Statistics:")
    logger.info(f"- Total chunks: {stats['total_chunks']}")
    logger.info(f"- Documents directory: {stats['documents_directory']}")
    logger.info(f"- Documents: {', '.join(stats['documents'])}")
    logger.info("Chunking and storage completed.")

if __name__ == "__main__":
    main()