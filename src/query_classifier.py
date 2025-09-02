import re
import logging, os
from src.logging_config import setup_logging

setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class QueryClassifier:
    def __init__(self):
        self.query_patterns = {
            'comparative': [
                r'compare|comparison|versus|vs\.?|difference|better|best|worst',
                r'which.*(?:plan|policy|coverage).*(?:better|best)',
                r'.*difference.*between.*'
            ],
            'aggregate': [
                r'total|sum|all|across.*(?:plans|policies)',
                r'list.*all|what.*all',
                r'.*maximum.*across.*'
            ],
            'conditional': [
                r'if.*then|what if|in case of|when.*happens',
                r'.*should I use.*',
                r'.*traveling.*|.*emergency.*'
            ],
            'gap_analysis': [
                r'not covered|gap|missing|overlap',
                r'what.*not.*covered',
                r'where.*coverage.*'
            ],
            'specific': [
                r'how much|cost|price|copay|deductible',
                r'what.*coverage.*for',
                r'.*covered.*'
            ]
        }

    def classify_query(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Classified query as {query_type}: {query}")
                    return query_type
        logger.info(f"Classified query as general: {query}")
        return 'general'