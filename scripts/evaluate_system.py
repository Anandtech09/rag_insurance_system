import time
import argparse
import os
from dotenv import load_dotenv
from src.rag_system import InsuranceRAGSystem
from src.config import SystemConfig
from src.logging_config import setup_logging
import logging
import sys

setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL"))
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Insurance RAG System with specific queries")
    parser.add_argument("--query", help="Single query to run")
    parser.add_argument("--query-file", help="File containing queries (one per line)")
    parser.add_argument("--output-file", default="evaluation_output.txt", help="Output file to store results")
    args = parser.parse_args()

    load_dotenv()
    config = SystemConfig.from_env()
    errors = config.validate()
    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"- {error}")
        return

    logger.info("Setting up Insurance RAG System for evaluation...")
    system = InsuranceRAGSystem(config)
    
    # Ensures system is ready (loads existing chunks)
    system.setup_system()  
    queries = []
    if args.query:
        queries = [args.query]
    elif args.query_file:
        try:
            with open(args.query_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to read query file {args.query_file}: {str(e)}")
            return
    else:
        queries = [
            "What is the difference in deductibles between in-network and out-of-network services?",
            "How much would I pay for a specialist visit before and after meeting my deductible?",
            "What preventive care services are covered at no cost?",
            "Compare the coverage for emergency room visits vs urgent care visits",
            "What is the coinsurance for mental health services (both inpatient and outpatient)?",
            "How many physical therapy visits are covered per year and what's the cost?",
            "What prescription drug tiers are available and what are the copays for each?",
            "Is there coverage for childbirth and what are the associated costs?",
            "What services require preauthorization and what happens if I don't get it?",
            "What is the out-of-pocket maximum for in-network vs out-of-network care?",
            "Which plan has better coverage for diagnostic imaging (MRI, CT scans)?",
            "Compare the prescription drug coverage between the plans"
        ]

    # Open the output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        # Custom print function to write to both file and console
        def dual_print(*args, **kwargs):
            print(*args, **kwargs)  # Print to console
            print(*args, file=f, **kwargs)  # Print to file

        dual_print("\n" + "="*80)
        dual_print("TESTING INSURANCE RAG SYSTEM")
        dual_print("="*80)

        stats = system.get_system_stats()
        dual_print(f"\nSystem Statistics:")
        dual_print(f"- Total chunks: {stats['total_chunks']}")
        dual_print(f"- Documents directory: {stats['documents_directory']}")
        dual_print(f"- Documents: {', '.join(stats['documents'])}")

        for i, query in enumerate(queries, 1):
            dual_print(f"\n{'='*80}")
            dual_print(f"QUERY {i}: {query}")
            dual_print('='*80)
            try:
                result = system.query(query)
                dual_print(f"\nQuery Type: {result.query_type.upper()}")
                dual_print(f"Confidence Score: {result.confidence_score:.2f}")
                dual_print(f"\nANSWER:")
                dual_print("-" * 50)
                dual_print(result.answer)
                dual_print(f"\nSOURCES:")
                dual_print("-" * 50)
                for source in result.sources[:3]:
                    dual_print(f"- {source}")
                if len(result.sources) > 3:
                    dual_print(f"... and {len(result.sources) - 3} more sources")
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {str(e)}")
            time.sleep(2)  # Respect rate limits

        dual_print(f"\n{'='*80}")
        dual_print("EVALUATION COMPLETED")
        dual_print("="*80)

if __name__ == "__main__":
    main()