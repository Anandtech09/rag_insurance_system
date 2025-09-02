import logging
from src.logging_config import setup_logging
import os
from typing import List, Dict, Optional
from groq import Groq
import google.generativeai as genai
from .config import SystemConfig
from dataclasses import dataclass
from time import sleep
from random import uniform

setup_logging(logs_dir=os.getenv("LOGS_DIR", "./logs"), log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    id: str
    content: str
    metadata: Dict
    similarity_score: float

@dataclass
class SynthesisResult:
    answer: str
    sources: List[str]
    confidence_score: float
    query_type: str
    relevant_chunks: List[Dict]

class ResponseSynthesizer:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.service = os.getenv("SERVICE", "groq").lower()
        logger.info(f"Initializing ResponseSynthesizer with service: {self.service.upper()}")

        if self.service == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is required for SERVICE=groq. Set it in .env file.")
            try:
                self.client = Groq(api_key=api_key)
                # Test API key validity
                self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=10
                )
                self.model = "llama-3.3-70b-versatile"
                logger.info(f"Groq client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")
                raise ValueError(f"Invalid GROQ_API_KEY or connectivity issue: {str(e)}")
        elif self.service == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for SERVICE=gemini. Set it in .env file.")
            try:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel("gemini-1.5-pro")
                self.model = "gemini-1.5-pro"
                logger.info(f"Gemini client initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {str(e)}")
                raise ValueError(f"Invalid GEMINI_API_KEY or connectivity issue: {str(e)}")
        else:
            raise ValueError(f"Unsupported service: {self.service}. Set SERVICE=groq or gemini in .env")

        self.max_output_tokens = config.synthesis.max_output_tokens
        self.temperature = config.synthesis.temperature

    def synthesize_answer(self, query: str, retrieved_chunks: List[RetrievalResult], query_type: str) -> SynthesisResult:
        """Synthesize an answer from retrieved chunks, with rate limit handling"""
        logger.info(f"Synthesizing answer using {self.service.upper()} for query: {query[:50]}...")
        try:
            if self.service == "gemini":
                sleep(6)  # Gemini rate limit delay
            prompt = self._build_prompt(query, retrieved_chunks, query_type)
            for attempt in range(3):
                try:
                    if self.service == "groq":
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=self.max_output_tokens,
                            temperature=self.temperature
                        )
                        answer = response.choices[0].message.content
                    else:
                        response = self.client.generate_content(
                            prompt,
                            generation_config={
                                "max_output_tokens": self.max_output_tokens,
                                "temperature": self.temperature
                            }
                        )
                        answer = response.text
                    logger.info(f"Successfully generated answer using {self.service.upper()}")
                    break
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < 2:
                        sleep(2 ** attempt + uniform(0, 0.1))
                    else:
                        raise e

            sources = [
                f"[Source {i+1}: {chunk.metadata['document_title']}, Page {chunk.metadata.get('page_number', 'N/A')}, Section: {chunk.metadata.get('section_title', 'N/A')}]"
                for i, chunk in enumerate(retrieved_chunks) if chunk.similarity_score >= self.config.similarity_threshold
            ]
            confidence = max(sum(chunk.similarity_score for chunk in retrieved_chunks) / max(len(retrieved_chunks), 1), 0.65)

            logger.info(f"Generated answer with confidence: {confidence:.2f} using {self.service.upper()}")
            return SynthesisResult(
                answer=answer,
                sources=sources,
                confidence_score=confidence,
                query_type=query_type,
                relevant_chunks=[{
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'similarity_score': chunk.similarity_score
                } for chunk in retrieved_chunks]
            )
        except Exception as e:
            logger.error(f"Failed to synthesize answer with {self.service.upper()}: {str(e)}")
            return SynthesisResult(
                answer=f"Error processing query with {self.service.upper()}: {str(e)}. Please check your API quota or key.",
                sources=[
                    f"[Source {i+1}: {chunk.metadata['document_title']}, Page {chunk.metadata.get('page_number', 'N/A')}, Section: {chunk.metadata.get('section_title', 'N/A')}]"
                    for i, chunk in enumerate(retrieved_chunks)
                ],
                confidence_score=0.0,
                query_type=query_type,
                relevant_chunks=[{
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'similarity_score': chunk.similarity_score
                } for chunk in retrieved_chunks]
            )

    def _build_prompt(self, query: str, retrieved_chunks: List[RetrievalResult], query_type: str) -> str:
        """Build the prompt based on query type"""
        context = "\n".join([
            f"Source {i+1} (Document: {chunk.metadata['document_title']}, Page {chunk.metadata.get('page_number', 'N/A')}, Section: {chunk.metadata.get('section_title', 'N/A')}):\n{chunk.content}\n"
            for i, chunk in enumerate(retrieved_chunks) if chunk.similarity_score >= self.config.similarity_threshold
        ])

        prompt_templates = {
            "comparative": (
                f"You are an expert in analyzing insurance policies. Compare the following aspects based on the provided context: {query}\n"
                f"Context:\n{context}\n\n"
                "Provide a detailed comparison, citing specific details from the sources. Include source references in the format [Source N: Document Title, Page X, Section Y]. "
                "If information is missing for a comparison, state so explicitly. Do not make assumptions beyond the provided context."
            ),
            "aggregate": (
                f"You are an expert in analyzing insurance policies. Aggregate information to answer: {query}\n"
                f"Context:\n{context}\n\n"
                "Summarize the relevant information across all sources, citing specific details with source references in the format [Source N: Document Title, Page X, Section Y]. "
                "If information is incomplete, indicate what is missing."
            ),
            "conditional": (
                f"You are an expert in analyzing insurance policies. Answer the conditional question: {query}\n"
                f"Context:\n{context}\n\n"
                "Provide a clear answer based on the conditions specified, citing specific details with source references in the format [Source N: Document Title, Page X, Section Y]. "
                "If the condition cannot be fully evaluated, explain why."
            ),
            "gap_analysis": (
                f"You are an expert in analyzing insurance policies. Identify gaps or differences in coverage for: {query}\n"
                f"Context:\n{context}\n\n"
                "Highlight any missing coverage or discrepancies between plans, citing specific details with source references in the format [Source N: Document Title, Page X, Section Y]. "
                "If no gaps are found, state so explicitly."
            ),
            "specific": (
                f"You are an expert in analyzing insurance policies. Answer the specific question: {query}\n"
                f"Context:\n{context}\n\n"
                "Provide a precise answer with specific details from the sources, using source references in the format [Source N: Document Title, Page X, Section Y]. "
                "If the information is not available, state so explicitly."
            ),
            "general": (
                f"You are an expert in analyzing insurance policies. Answer the question: {query}\n"
                f"Context:\n{context}\n\n"
                "Provide a comprehensive answer with details from the sources, using source references in the format [Source N: Document Title, Page X, Section Y]. "
                "If information is missing, indicate what is not covered in the provided context."
            )
        }

        return prompt_templates.get(query_type, prompt_templates["general"])