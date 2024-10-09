import pytest
from unittest import mock
from src.orchestrator.main import interact_with_llm_huggingface_streaming  # Ajusta la importación según tu estructura
import os
import requests

class TestInteractWithLlmHuggingfaceStreaming:

    @mock.patch('requests.post')
    def test_successful_post_request(self, mock_post):
        # Arrange
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter([b"line 1", b"line 2", b"line 3"])
        mock_post.return_value = mock_response

        user_input = "What is the capital of France?"
        extracted_texts = [
            {"title": "France", "link": "http://example.com/france", "content": "France is a country in Europe."}
        ]

        # Act
        result = interact_with_llm_huggingface_streaming(user_input, extracted_texts)

        # Assert
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['headers']['Authorization'].startswith('Bearer ')
        assert kwargs['json']['inputs'].startswith("Información extraída:")
        assert kwargs['json']['stream'] is True
        assert result is not None  # Asegúrate de que haya algún resultado

    @mock.patch('requests.post')
    def test_successful_post_request_bytes(self, mock_post):
        # Arrange
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter([b"line 1", b"line 2", b"line 3"])  # Cambiado a bytes
        mock_post.return_value = mock_response

        user_input = "What is the capital of France?"
        extracted_texts = [
            {"title": "France", "link": "http://example.com/france", "content": "France is a country in Europe."}
        ]

        # Act
        result = interact_with_llm_huggingface_streaming(user_input, extracted_texts)

        # Assert
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['headers']['Authorization'].startswith('Bearer ')
        assert kwargs['json']['inputs'].startswith("Información extraída:")
        assert kwargs['json']['stream'] is True
        assert result is not None

    @mock.patch('requests.post')
    @mock.patch('os.getenv')
    def test_properly_loads_huggingface_api_key(self, mock_getenv, mock_post):
        # Arrange
        mock_getenv.return_value = "mocked_api_key"  # Simula la API key obtenida del entorno
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [b"line 1", b"line 2"]  # Devuelve una lista de líneas como bytes
        mock_post.return_value = mock_response

        user_input = "Test user input"
        extracted_texts = [
            {"title": "Test Title", "link": "http://example.com/test", "content": "Test content"}
        ]

        # Act
        result = interact_with_llm_huggingface_streaming(user_input, extracted_texts)

        # Assert
        assert result is not None  # Asegúrate de que haya algún resultado
        assert "line 1" in result  # Verifica que el contenido esperado esté en el resultado
        assert "line 2" in result

    @mock.patch('requests.post')
    def test_empty_user_input(self, mock_post):
        # Arrange
        user_input = ""
        extracted_texts = [
            {"title": "Title 1", "link": "http://example.com/1", "content": "Content 1"},
            {"title": "Title 2", "link": "http://example.com/2", "content": "Content 2"}
        ]

        # Act
        interact_with_llm_huggingface_streaming(user_input, extracted_texts)

        # Assert
        mock_post.assert_not_called()

    @mock.patch('requests.post')
    def test_correctly_constructs_prompt(self, mock_post):
        # Arrange
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter([b"Generando respuesta 1", b"Generando respuesta 2"])
        mock_post.return_value = mock_response

        user_input = "What is the capital of France?"
        extracted_texts = [
            {"title": "France", "link": "http://example.com/france", "content": "France es un país en Europa."}
        ]

        # Act
        result = interact_with_llm_huggingface_streaming(user_input, extracted_texts)

        # Assert
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['headers']['Authorization'].startswith('Bearer ')
        assert kwargs['json']['inputs'].startswith("Información extraída:")
        assert kwargs['json']['stream'] is True
        assert result is not None

    @mock.patch('requests.post')
    def test_handles_malformed_data_in_extracted_texts(self, mock_post):
        # Arrange
        user_input = "What is the capital of France?"
        extracted_texts = [
            {"title": "France", "link": "http://example.com/france", "content": "France es un país en Europa."},
            {"title": "Spain", "link": "http://example.com/spain"}  # Missing 'content' key
        ]

        # Act
        result = interact_with_llm_huggingface_streaming(user_input, extracted_texts)

        # Assert
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['headers']['Authorization'].startswith('Bearer ')
        assert kwargs['json']['inputs'].startswith("Información extraída:")
        assert kwargs['json']['stream'] is True

        # Verify that the missing 'content' key is handled
        assert 'content' not in extracted_texts[1]
        assert extracted_texts[1]['title'] == 'Spain'

    @mock.patch('requests.post')
    def test_deals_with_network_issues(self, mock_post):
        # Arrange
        mock_response = mock.Mock()
        mock_response.status_code = 500  # Simulando un problema de red
        mock_post.return_value = mock_response

        user_input = "What is the capital of France?"
        extracted_texts = [
            {"title": "France", "link": "http://example.com/france", "content": "France es un país en Europa."}
        ]

        # Act
        result = interact_with_llm_huggingface_streaming(user_input, extracted_texts)

        # Assert
        mock_post.assert_called_once()
        assert result is None  # Suponiendo que maneja errores devolviendo None
