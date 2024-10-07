import sys
from pathlib import Path
import pytest
from unittest.mock import Mock
import requests

# Agregar la carpeta src al PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# Importar la función que se va a probar
from orchestrator.main import extract_text_from_url

class TestExtractTextFromUrl:
    # Successfully extract text from a well-formed HTML page with multiple paragraphs
    def test_extract_text_from_well_formed_html(self, mocker):
        url = "http://example.com"
        html_content = "<html><body><p>Paragraph 1</p><p>Paragraph 2</p></body></html>"
        mock_response = mocker.Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch('requests.get', return_value=mock_response)

        result = extract_text_from_url(url)
        expected_result = "Paragraph 1\nParagraph 2"
        assert result == expected_result, "Expected extracted text to match the HTML content"

    # Handle an empty URL input gracefully
    def test_handle_empty_url_input(self, mocker):
        url = ""
        mocker.patch('requests.get', side_effect=requests.RequestException("Invalid URL"))

        result = extract_text_from_url(url)
        expected_result = "No se pudo extraer contenido relevante."
        assert result == expected_result, "Expected specific message for empty URL"

    # Handle and return text from a page with a single paragraph
    def test_extract_text_from_single_paragraph(self, mocker):
        url = "http://example.com"
        html_content = "<html><body><p>Single Paragraph</p></body></html>"
        mock_response = mocker.Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch('requests.get', return_value=mock_response)

        result = extract_text_from_url(url)
        expected_result = "Single Paragraph"
        assert result == expected_result, "Expected extracted text to match single paragraph content"

    # Return extracted text when the page contains nested HTML elements within paragraphs
    def test_extract_text_from_nested_html(self, mocker):
        url = "http://example.com"
        html_content = "<html><body><p>Paragraph 1</p><div><p>Paragraph 2</p></div></body></html>"
        mock_response = mocker.Mock()
        mock_response.text = html_content
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch('requests.get', return_value=mock_response)

        result = extract_text_from_url(url)
        expected_result = "Paragraph 1\nParagraph 2"
        assert result == expected_result, "Expected extracted text to match nested HTML content"

    # Manage URLs that return a non-200 HTTP status code
    def test_non_200_http_status_code(self, mocker):
        url = "http://example.com"
        mock_response = mocker.Mock()
        mock_response.raise_for_status.side_effect = requests.RequestException("404 Client Error: Not Found")
        mocker.patch('requests.get', return_value=mock_response)

        result = extract_text_from_url(url)
        expected_result = "Error al extraer el contenido."  
        assert result == expected_result, "Expected specific error message for non-200 status code"
