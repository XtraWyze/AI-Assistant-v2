"""
Unit tests for Ollama stream parsing functionality.
Tests the OllamaClient.generate_stream() NDJSON parsing logic.
"""
import unittest
import json
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add wyzer to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wyzer.brain.ollama_client import OllamaClient


class TestStreamParsing(unittest.TestCase):
    """Test NDJSON stream parsing from Ollama."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = OllamaClient(base_url="http://localhost:11434", timeout=10)
    
    def test_stream_parsing_normal(self):
        """Test parsing valid NDJSON stream with multiple tokens."""
        # Mock stream data (NDJSON format - one JSON object per line)
        ndjson_lines = [
            b'{"response":"Hello","done":false}\n',
            b'{"response":" ","done":false}\n',
            b'{"response":"world","done":false}\n',
            b'{"response":"!","done":true}\n',
        ]
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.__iter__ = Mock(return_value=iter(ndjson_lines))
        
        with patch.object(self.client.opener, 'open', return_value=mock_response):
            chunks = list(self.client.generate_stream(
                prompt="test",
                model="llama3.1:latest",
                options={}
            ))
        
        # Verify we got all the text chunks
        self.assertEqual(chunks, ["Hello", " ", "world", "!"])
    
    def test_stream_parsing_empty_responses(self):
        """Test that empty response fields are skipped."""
        ndjson_lines = [
            b'{"response":"Start","done":false}\n',
            b'{"response":"","done":false}\n',  # Empty response - should be skipped
            b'{"response":"End","done":true}\n',
        ]
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.__iter__ = Mock(return_value=iter(ndjson_lines))
        
        with patch.object(self.client.opener, 'open', return_value=mock_response):
            chunks = list(self.client.generate_stream(
                prompt="test",
                model="llama3.1:latest",
                options={}
            ))
        
        # Empty response should be skipped
        self.assertEqual(chunks, ["Start", "End"])
    
    def test_stream_parsing_stops_on_done(self):
        """Test that streaming stops when done=true."""
        ndjson_lines = [
            b'{"response":"First","done":false}\n',
            b'{"response":"Second","done":true}\n',
            b'{"response":"Third","done":false}\n',  # Should not be read
        ]
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.__iter__ = Mock(return_value=iter(ndjson_lines))
        
        with patch.object(self.client.opener, 'open', return_value=mock_response):
            chunks = list(self.client.generate_stream(
                prompt="test",
                model="llama3.1:latest",
                options={}
            ))
        
        # Should only get first two chunks
        self.assertEqual(chunks, ["First", "Second"])
    
    def test_stream_parsing_blank_lines(self):
        """Test that blank lines are properly skipped."""
        ndjson_lines = [
            b'{"response":"Hello","done":false}\n',
            b'\n',  # Blank line
            b'{"response":" world","done":true}\n',
        ]
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.__iter__ = Mock(return_value=iter(ndjson_lines))
        
        with patch.object(self.client.opener, 'open', return_value=mock_response):
            chunks = list(self.client.generate_stream(
                prompt="test",
                model="llama3.1:latest",
                options={}
            ))
        
        self.assertEqual(chunks, ["Hello", " world"])
    
    def test_stream_parsing_invalid_json(self):
        """Test that invalid JSON raises a clear error."""
        ndjson_lines = [
            b'{"response":"Valid","done":false}\n',
            b'invalid json here\n',  # Invalid JSON
        ]
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.__iter__ = Mock(return_value=iter(ndjson_lines))
        
        with patch.object(self.client.opener, 'open', return_value=mock_response):
            with self.assertRaises(ValueError) as context:
                list(self.client.generate_stream(
                    prompt="test",
                    model="llama3.1:latest",
                    options={}
                ))
            
            self.assertIn("Invalid JSON in stream", str(context.exception))
    
    def test_stream_parsing_unicode(self):
        """Test that Unicode characters in responses are handled correctly."""
        ndjson_lines = [
            b'{"response":"Hello ","done":false}\n',
            b'{"response":"\xf0\x9f\x91\x8b","done":false}\n',  # Wave emoji in UTF-8
            b'{"response":" World","done":true}\n',
        ]
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.__iter__ = Mock(return_value=iter(ndjson_lines))
        
        with patch.object(self.client.opener, 'open', return_value=mock_response):
            chunks = list(self.client.generate_stream(
                prompt="test",
                model="llama3.1:latest",
                options={}
            ))
        
        result = "".join(chunks)
        self.assertIn("👋", result)  # Wave emoji should be in result


class TestStreamAccumulation(unittest.TestCase):
    """Test that non-streaming mode can use streaming internally."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = OllamaClient(base_url="http://localhost:11434", timeout=10)
    
    def test_generate_non_stream_returns_final_string(self):
        """Test that generate() with stream=False returns accumulated string."""
        ndjson_lines = [
            b'{"response":"The","done":false}\n',
            b'{"response":" answer","done":false}\n',
            b'{"response":" is","done":false}\n',
            b'{"response":" 42","done":true}\n',
        ]
        
        mock_response = MagicMock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_response.__iter__ = Mock(return_value=iter(ndjson_lines))
        
        with patch.object(self.client.opener, 'open', return_value=mock_response):
            result = self.client.generate(
                prompt="test",
                model="llama3.1:latest",
                options={},
                stream=True  # Use streaming internally
            )
        
        # Should accumulate to full string
        self.assertEqual(result, "The answer is 42")


if __name__ == '__main__':
    unittest.main()
