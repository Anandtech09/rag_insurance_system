"""
Evaluation metrics for Insurance RAG System
Implements evaluation criteria from the assignment
"""

import logging, os
from typing import Dict, List, Any
import pandas as pd
from src.response_synthesizer import SynthesisResult
from src.logging_config import setup_logging

setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    def __init__(self):
        self.results = []
        self.evaluation_criteria = {
            'retrieval_quality': 0.30,
            'synthesis_accuracy': 0.25,
            'source_attribution': 0.20,
            'query_understanding': 0.15,
            'completeness': 0.10
        }

    def evaluate_result(self, query: str, result: SynthesisResult, expected_aspects: List[str] = None) -> Dict[str, float]:
        """Evaluate a single query result"""
        metrics = {
            'retrieval_quality': self._evaluate_retrieval_quality(result),
            'synthesis_accuracy': self._evaluate_synthesis_accuracy(result),
            'source_attribution': self._evaluate_source_attribution(result),
            'query_understanding': self._evaluate_query_understanding(query, result),
            'completeness': self._evaluate_completeness(result, expected_aspects or [])
        }
        metrics['overall_score'] = sum(metrics[c] * w for c, w in self.evaluation_criteria.items())
        metrics['query'] = query
        metrics['query_type'] = result.query_type
        self.results.append(metrics)
        return metrics

    def _evaluate_retrieval_quality(self, result: SynthesisResult) -> float:
        """Evaluate retrieval quality (30% weight)"""
        if not result.relevant_chunks:
            return 0.0
        avg_similarity = sum(chunk['similarity_score'] for chunk in result.relevant_chunks) / len(result.relevant_chunks)
        unique_documents = len(set(chunk['metadata']['document_title'] for chunk in result.relevant_chunks))
        diversity_score = min(unique_documents / 2.0, 1.0)
        high_relevance = sum(1 for chunk in result.relevant_chunks if chunk['similarity_score'] > 0.7)
        relevance_ratio = high_relevance / len(result.relevant_chunks)
        return (avg_similarity * 0.5 + diversity_score * 0.3 + relevance_ratio * 0.2)

    def _evaluate_synthesis_accuracy(self, result: SynthesisResult) -> float:
        """Evaluate synthesis accuracy (25% weight)"""
        answer = result.answer.lower()
        insurance_terms = ['deductible', 'copay', 'coinsurance', 'in-network', 'out-of-network', 
                          'coverage', 'benefit', 'premium', 'claim', 'preauthorization']
        terms_present = sum(1 for term in insurance_terms if term in answer)
        terminology_score = min(terms_present / 5.0, 1.0)
        has_specific_values = any(char in answer for char in ['$', '%'])
        specificity_score = 1.0 if has_specific_values else 0.5
        answer_length = len(result.answer.split())
        length_score = 1.0 if 20 <= answer_length <= 200 else 0.7
        return (terminology_score * 0.3 + specificity_score * 0.2 + length_score * 0.2 + result.confidence_score * 0.3)

    def _evaluate_source_attribution(self, result: SynthesisResult) -> float:
        """Evaluate source attribution (20% weight)"""
        if not result.sources:
            return 0.0
        properly_formatted = sum(1 for source in result.sources if 'Source' in source and 'Page' in source)
        formatting_score = properly_formatted / len(result.sources)
        has_page_numbers = sum(1 for source in result.sources if 'Page' in source)
        page_citation_score = has_page_numbers / len(result.sources)
        has_sections = sum(1 for source in result.sources if 'Section' in source)
        section_citation_score = has_sections / len(result.sources) if result.sources else 0
        return (formatting_score * 0.4 + page_citation_score * 0.3 + section_citation_score * 0.3)

    def _evaluate_query_understanding(self, query: str, result: SynthesisResult) -> float:
        """Evaluate query understanding (15% weight)"""
        query_lower = query.lower()
        answer_lower = result.answer.lower()
        type_mapping = {
            'comparative': ['compare', 'difference', 'better', 'versus', 'vs'],
            'aggregate': ['total', 'all', 'sum', 'maximum', 'list'],
            'conditional': ['if', 'when', 'case', 'should'],
            'gap_analysis': ['not covered', 'gap', 'missing', 'overlap'],
            'specific': ['how much', 'what', 'cost', 'copay']
        }
        expected_keywords = type_mapping.get(result.query_type, [])
        type_accuracy = 1.0 if any(keyword in query_lower for keyword in expected_keywords) else 0.5
        query_words = set(query_lower.split())
        answer_words = set(answer_lower.split())
        overlap_ratio = len(query_words & answer_words) / len(query_words)
        response_appropriateness = self._check_response_appropriateness(query, result)
        return (type_accuracy * 0.4 + overlap_ratio * 0.3 + response_appropriateness * 0.3)

    def _check_response_appropriateness(self, query: str, result: SynthesisResult) -> float:
        """Check if response style matches query type"""
        answer_lower = result.answer.lower()
        if result.query_type == 'comparative':
            comparison_words = ['higher', 'lower', 'better', 'worse', 'more', 'less', 'than']
            return 1.0 if any(word in answer_lower for word in comparison_words) else 0.6
        elif result.query_type == 'aggregate':
            aggregate_words = ['total', 'combined', 'all', 'sum', 'include']
            return 1.0 if any(word in answer_lower for word in aggregate_words) else 0.6
        elif result.query_type == 'conditional':
            conditional_words = ['would', 'should', 'if', 'when', 'case', 'scenario']
            return 1.0 if any(word in answer_lower for word in conditional_words) else 0.6
        elif result.query_type == 'gap_analysis':
            gap_words = ['not covered', 'excluded', 'gap', 'missing', 'overlap', 'duplicate']
            return 1.0 if any(word in answer_lower for word in gap_words) else 0.6
        return 0.8

    def _evaluate_completeness(self, result: SynthesisResult, expected_aspects: List[str]) -> float:
        """Evaluate completeness (10% weight)"""
        if not expected_aspects:
            answer_length = len(result.answer.split())
            has_sources = len(result.sources) > 0
            has_specific_info = '$' in result.answer or '%' in result.answer
            return (
                min(answer_length / 50.0, 1.0) * 0.4 +
                (1.0 if has_sources else 0.0) * 0.3 +
                (1.0 if has_specific_info else 0.5) * 0.3
            )
        answer_lower = result.answer.lower()
        covered_aspects = sum(1 for aspect in expected_aspects if aspect.lower() in answer_lower)
        return covered_aspects / len(expected_aspects)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive evaluation summary"""
        if not self.results:
            return {}
        df = pd.DataFrame(self.results)
        return {
            'total_queries': len(self.results),
            'average_scores': {
                'overall': df['overall_score'].mean(),
                'retrieval_quality': df['retrieval_quality'].mean(),
                'synthesis_accuracy': df['synthesis_accuracy'].mean(),
                'source_attribution': df['source_attribution'].mean(),
                'query_understanding': df['query_understanding'].mean(),
                'completeness': df['completeness'].mean()
            },
            'score_distribution': {
                'excellent': len(df[df['overall_score'] >= 0.9]),
                'good': len(df[(df['overall_score'] >= 0.7) & (df['overall_score'] < 0.9)]),
                'satisfactory': len(df[(df['overall_score'] >= 0.5) & (df['overall_score'] < 0.7)]),
                'needs_improvement': len(df[df['overall_score'] < 0.5])
            },
            'query_type_performance': df.groupby('query_type')['overall_score'].mean().to_dict(),
            'detailed_results': self.results
        }