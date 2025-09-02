import pytest
import os
from unittest.mock import patch, Mock
from .embedding_test import EmbeddingService, VectorDatabase, QueryClassifier, ResponseSynthesizer, InsuranceRAGSystem, SystemConfig, SynthesisConfig, RetrievalResult, DocumentChunk, SynthesisResult

@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)

@pytest.fixture
def system_config(temp_dir):
    return SystemConfig(
        documents_dir=os.path.join(temp_dir, "documents"),
        logs_dir=os.path.join(temp_dir, "logs"),
        cache_dir=os.path.join(temp_dir, "cache"),
        db_path=os.path.join(temp_dir, "chroma_db"),
        synthesis=SynthesisConfig(max_output_tokens=2048, temperature=0.2)
    )

class TestEmbeddingService:
    def test_get_embedding(self, temp_dir, system_config):
        service = EmbeddingService(model_name="all-mpnet-base-v2", cache_dir=system_config.cache_dir)
        with patch('sentence_transformers.SentenceTransformer.encode') as mock_encode:
            mock_encode.return_value = [0.1, 0.2, 0.3]
            embedding = service.get_embedding("test text")
            assert len(embedding) == 3
            assert embedding == [0.1, 0.2, 0.3]

    def test_get_embeddings_batch(self, temp_dir, system_config):
        service = EmbeddingService(model_name="all-mpnet-base-v2", cache_dir=system_config.cache_dir)
        with patch('sentence_transformers.SentenceTransformer.encode') as mock_encode:
            mock_encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            embeddings = service.get_embeddings_batch(["text1", "text2"])
            assert len(embeddings) == 2
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]

class TestVectorDatabase:
    def test_document_exists(self, temp_dir, system_config):
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            mock_collection.get.return_value = {'ids': ['id1'], 'documents': [], 'metadatas': []}
            db = VectorDatabase(db_path=system_config.db_path)
            assert db.document_exists("title") is True

class TestQueryClassifier:
    def test_classify_query(self):
        classifier = QueryClassifier()
        assert classifier.classify_query("Compare deductibles") == 'comparative'
        assert classifier.classify_query("Total out-of-pocket maximum") == 'aggregate'
        assert classifier.classify_query("If I have a heart attack") == 'conditional'

class TestResponseSynthesizer:
    def test_synthesize_answer(self, system_config):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test answer"))]
        mock_client.chat.completions.create.return_value = mock_response
        synthesizer = ResponseSynthesizer(system_config, client=mock_client)
        chunk = RetrievalResult(
            id="id1",
            content="content",
            metadata={'document_title': 'title', 'page_number': '1', 'section_title': 'section'},
            similarity_score=0.9
        )
        result = synthesizer.synthesize_answer("test query", [chunk], "specific")
        assert isinstance(result, SynthesisResult)
        assert result.answer == "Test answer"

    def test_build_prompt(self, system_config):
        mock_client = Mock()
        synthesizer = ResponseSynthesizer(system_config, client=mock_client)
        chunk = RetrievalResult(
            id="id1",
            content="content",
            metadata={'document_title': 'title', 'page_number': '1', 'section_title': 'section'},
            similarity_score=0.9
        )
        prompt = synthesizer._build_prompt("test query", [chunk], "specific")
        assert "test query" in prompt
        assert "Source 1: title, Page 1, Section: section" in prompt

class TestInsuranceRAGSystem:
    def test_query(self, temp_dir, system_config):
        with patch('sentence_transformers.SentenceTransformer.encode') as mock_encode, \
             patch('chromadb.PersistentClient') as mock_client:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Answer"))]
            mock_groq_client = Mock()
            mock_groq_client.chat.completions.create.return_value = mock_response
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                'ids': [['id1']],
                'documents': [['content']],
                'metadatas': [[{'document_title': 'title', 'page_number': '1', 'section_title': 'section'}]],
                'distances': [[0.1]]
            }
            mock_encode.return_value = [0.1, 0.2, 0.3]
            system = InsuranceRAGSystem(system_config, client=mock_groq_client)
            chunk = DocumentChunk(
                id="id1",
                content="content",
                document_source="source",
                document_title="title",
                page_number=1,
                section_title="section",
                chunk_index=0,
                metadata={}
            )
            system.vector_db.add_chunks([chunk], [[0.1, 0.2, 0.3]])
            result = system.query("What is the deductible?")
            assert isinstance(result, SynthesisResult)
            assert result.answer == "Answer"