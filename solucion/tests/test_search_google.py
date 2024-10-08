import sys
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
import requests

# Agregar la carpeta src al PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# Importar la función que se va a probar
from orchestrator.main import search_google

class TestSearchGoogle:
    # Successfully retrieves search results when a valid query is provided
    def test_successful_search_with_valid_query(self, mocker):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'organic': [
                {'title': 'Result 1', 'link': 'http://example.com/1'},
                {'title': 'Result 2', 'link': 'http://example.com/2'}
            ]
        }
        mocker.patch('requests.post', return_value=mock_response)
    
        query = "test query"
        expected_links = [
            {'title': 'Result 1', 'link': 'http://example.com/1'},
            {'title': 'Result 2', 'link': 'http://example.com/2'}
        ]
    
        result = search_google(query)
        assert result == expected_links, "Expected the links to match the mock response"

    # Handles empty or null query strings gracefully
    def test_handle_empty_query(self, mocker):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'organic': []}
        mocker.patch('requests.post', return_value=mock_response)
    
        query = ""
        result = search_google(query)
        assert result == [], "Expected an empty list for an empty query"

    # Limits the number of returned search results to 5
    def test_limit_search_results_to_5(self, mocker):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'organic': [
                {'title': 'Result 1', 'link': 'http://example.com/1'},
                {'title': 'Result 2', 'link': 'http://example.com/2'},
                {'title': 'Result 3', 'link': 'http://example.com/3'},
                {'title': 'Result 4', 'link': 'http://example.com/4'},
                {'title': 'Result 5', 'link': 'http://example.com/5'},
                {'title': 'Result 6', 'link': 'http://example.com/6'}
            ]
        }
        mocker.patch('requests.post', return_value=mock_response)

        query = "test query"
        expected_links = [
            {'title': 'Result 1', 'link': 'http://example.com/1'},
            {'title': 'Result 2', 'link': 'http://example.com/2'},
            {'title': 'Result 3', 'link': 'http://example.com/3'},
            {'title': 'Result 4', 'link': 'http://example.com/4'},
            {'title': 'Result 5', 'link': 'http://example.com/5'}
        ]

        result = search_google(query)
        assert result == expected_links, "Expected only 5 links to be returned"

    # Manages HTTP errors by printing error messages and returning an empty list
    def test_manage_http_errors(self, mocker):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mocker.patch('requests.post', return_value=mock_response)
    
        query = "error query"
    
        with patch('builtins.print') as mock_print:
            result = search_google(query)
            mock_print.assert_called_with("Error en la búsqueda: 400 - Bad Request")
            assert result == [], "Expected an empty list on HTTP error"

    # Validates the presence and correctness of the API key
    def test_valid_api_key(self, mocker):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'organic': [
                {'title': 'Result 1', 'link': 'http://example.com/1'},
                {'title': 'Result 2', 'link': 'http://example.com/2'}
            ]
        }
        mocker.patch('requests.post', return_value=mock_response)
    
        serper_api_key = "valid_key"  # Este valor no se utiliza en la prueba
        query = "test query"
        expected_links = [
            {'title': 'Result 1', 'link': 'http://example.com/1'},
            {'title': 'Result 2', 'link': 'http://example.com/2'}
        ]
    
        result = search_google(query)
        assert result == expected_links, "Expected the links to match the mock response"
