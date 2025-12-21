"""
Unit tests for llama.cpp server manager and client.

These tests do NOT require an actual llama-server binary.
They test config parsing, graceful failure handling, and mock behavior.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path


class TestLlamaServerManager(unittest.TestCase):
    """Tests for wyzer.brain.llama_server_manager module."""
    
    def test_import(self):
        """Test that the module can be imported."""
        from wyzer.brain.llama_server_manager import (
            LlamaServerManager,
            get_llama_server_manager,
            ensure_server_running,
            stop_server,
            healthcheck
        )
        self.assertIsNotNone(LlamaServerManager)
        self.assertIsNotNone(get_llama_server_manager)
    
    def test_singleton_pattern(self):
        """Test that LlamaServerManager is a singleton."""
        from wyzer.brain.llama_server_manager import LlamaServerManager
        
        mgr1 = LlamaServerManager()
        mgr2 = LlamaServerManager()
        self.assertIs(mgr1, mgr2)
    
    def test_ensure_server_running_missing_binary(self):
        """Test graceful failure when binary is missing."""
        from wyzer.brain.llama_server_manager import get_llama_server_manager
        
        mgr = get_llama_server_manager()
        # Reset any existing state
        mgr.process = None
        mgr.base_url = None
        mgr._started_by_wyzer = False
        
        # Test with non-existent paths
        result = mgr.ensure_server_running(
            binary_path="./nonexistent/llama-server.exe",
            model_path="./nonexistent/model.gguf",
            port=8081
        )
        
        # Should return None gracefully, not raise
        self.assertIsNone(result)
    
    def test_ensure_server_running_missing_model(self):
        """Test graceful failure when model is missing but binary exists."""
        from wyzer.brain.llama_server_manager import get_llama_server_manager
        import tempfile
        
        mgr = get_llama_server_manager()
        mgr.process = None
        mgr.base_url = None
        mgr._started_by_wyzer = False
        
        # Create a fake binary file
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            fake_binary = f.name
        
        try:
            result = mgr.ensure_server_running(
                binary_path=fake_binary,
                model_path="./nonexistent/model.gguf",
                port=8081
            )
            # Should fail because model doesn't exist
            self.assertIsNone(result)
        finally:
            os.unlink(fake_binary)
    
    def test_healthcheck_returns_false_on_no_url(self):
        """Test healthcheck returns False when no URL is set."""
        from wyzer.brain.llama_server_manager import get_llama_server_manager
        
        mgr = get_llama_server_manager()
        mgr.base_url = None
        
        result = mgr.healthcheck()
        self.assertFalse(result)
    
    @patch('urllib.request.urlopen')
    def test_healthcheck_success(self, mock_urlopen):
        """Test healthcheck returns True on successful response."""
        from wyzer.brain.llama_server_manager import get_llama_server_manager
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response
        
        mgr = get_llama_server_manager()
        result = mgr.healthcheck("http://127.0.0.1:8081")
        
        self.assertTrue(result)
    
    @patch('urllib.request.urlopen')
    def test_healthcheck_failure(self, mock_urlopen):
        """Test healthcheck returns False on connection error."""
        from wyzer.brain.llama_server_manager import get_llama_server_manager
        import urllib.error
        
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        
        mgr = get_llama_server_manager()
        result = mgr.healthcheck("http://127.0.0.1:8081")
        
        self.assertFalse(result)
    
    def test_get_server_info(self):
        """Test get_server_info returns expected structure."""
        from wyzer.brain.llama_server_manager import get_llama_server_manager
        
        mgr = get_llama_server_manager()
        info = mgr.get_server_info()
        
        self.assertIn("running", info)
        self.assertIn("started_by_wyzer", info)
        self.assertIn("base_url", info)
        self.assertIn("pid", info)
    
    def test_stop_server_no_process(self):
        """Test stop_server is a no-op when no process is running."""
        from wyzer.brain.llama_server_manager import get_llama_server_manager
        
        mgr = get_llama_server_manager()
        mgr.process = None
        
        # Should not raise
        mgr.stop_server()


class TestLlamaCppClient(unittest.TestCase):
    """Tests for wyzer.brain.llamacpp_client module."""
    
    def test_import(self):
        """Test that the client can be imported."""
        from wyzer.brain.llamacpp_client import LlamaCppClient
        self.assertIsNotNone(LlamaCppClient)
    
    def test_client_initialization(self):
        """Test client can be initialized with custom URL."""
        from wyzer.brain.llamacpp_client import LlamaCppClient
        
        client = LlamaCppClient(
            base_url="http://localhost:9999",
            timeout=60
        )
        
        self.assertEqual(client.base_url, "http://localhost:9999")
        self.assertEqual(client.timeout, 60)
    
    def test_ping_success(self):
        """Test ping returns True on successful health check."""
        from wyzer.brain.llamacpp_client import LlamaCppClient
        
        client = LlamaCppClient()
        
        # Mock the opener.open method
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        
        with patch.object(client.opener, 'open', return_value=mock_response):
            result = client.ping()
        
        self.assertTrue(result)
    
    def test_ping_failure(self):
        """Test ping returns False on connection failure."""
        from wyzer.brain.llamacpp_client import LlamaCppClient
        import urllib.error
        
        client = LlamaCppClient()
        
        # Mock the opener.open to raise an error
        with patch.object(client.opener, 'open', side_effect=urllib.error.URLError("Connection refused")):
            result = client.ping()
        
        self.assertFalse(result)


class TestConfigLlamaCppSettings(unittest.TestCase):
    """Tests for Config llamacpp settings."""
    
    def test_config_has_llamacpp_settings(self):
        """Test that Config class has llamacpp settings."""
        from wyzer.core.config import Config
        
        # Check all new config options exist
        self.assertTrue(hasattr(Config, 'LLAMACPP_BIN_PATH'))
        self.assertTrue(hasattr(Config, 'LLAMACPP_MODEL_PATH'))
        self.assertTrue(hasattr(Config, 'LLAMACPP_PORT'))
        self.assertTrue(hasattr(Config, 'LLAMACPP_CTX_SIZE'))
        self.assertTrue(hasattr(Config, 'LLAMACPP_THREADS'))
        self.assertTrue(hasattr(Config, 'LLAMACPP_BASE_URL'))
    
    def test_llm_mode_options(self):
        """Test that LLM_MODE config supports expected values."""
        from wyzer.core.config import Config
        
        # LLM_MODE should be a string
        self.assertIsInstance(Config.LLM_MODE, str)
        # Default should be 'ollama'
        self.assertEqual(Config.LLM_MODE, "ollama")


class TestCLIArgParsing(unittest.TestCase):
    """Tests for CLI argument parsing of llamacpp flags."""
    
    def test_llm_mode_choices(self):
        """Test that --llm accepts llamacpp as a choice."""
        import argparse
        
        # Simulate parsing --llm llamacpp
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--llm",
            type=str,
            default="ollama",
            choices=["ollama", "llamacpp", "off"]
        )
        
        # Should not raise
        args = parser.parse_args(["--llm", "llamacpp"])
        self.assertEqual(args.llm, "llamacpp")
        
        args = parser.parse_args(["--llm", "ollama"])
        self.assertEqual(args.llm, "ollama")
        
        args = parser.parse_args(["--llm", "off"])
        self.assertEqual(args.llm, "off")
    
    def test_llamacpp_flags(self):
        """Test that llamacpp-specific flags are parseable."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--llamacpp-bin", type=str, default="./wyzer/llm_bin/llama-server.exe")
        parser.add_argument("--llamacpp-model", type=str, default="./wyzer/llm_models/model.gguf")
        parser.add_argument("--llamacpp-port", type=int, default=8081)
        parser.add_argument("--llamacpp-ctx", type=int, default=2048)
        parser.add_argument("--llamacpp-threads", type=int, default=4)
        
        args = parser.parse_args([
            "--llamacpp-bin", "/path/to/server",
            "--llamacpp-model", "/path/to/model.gguf",
            "--llamacpp-port", "9000",
            "--llamacpp-ctx", "4096",
            "--llamacpp-threads", "8"
        ])
        
        self.assertEqual(args.llamacpp_bin, "/path/to/server")
        self.assertEqual(args.llamacpp_model, "/path/to/model.gguf")
        self.assertEqual(args.llamacpp_port, 9000)
        self.assertEqual(args.llamacpp_ctx, 4096)
        self.assertEqual(args.llamacpp_threads, 8)


class TestOrchestratorLLMHelper(unittest.TestCase):
    """Tests for orchestrator LLM helper functions."""
    
    def test_get_llm_client_returns_none_when_disabled(self):
        """Test _get_llm_client returns None when NO_OLLAMA is set."""
        from wyzer.core.config import Config
        
        # Save original values
        orig_no_ollama = getattr(Config, 'NO_OLLAMA', False)
        
        try:
            Config.NO_OLLAMA = True
            
            from wyzer.core.orchestrator import _get_llm_client
            client = _get_llm_client()
            
            self.assertIsNone(client)
        finally:
            Config.NO_OLLAMA = orig_no_ollama
    
    def test_get_llm_client_returns_ollama_by_default(self):
        """Test _get_llm_client returns OllamaClient by default."""
        from wyzer.core.config import Config
        from wyzer.brain.ollama_client import OllamaClient
        
        # Save original values
        orig_no_ollama = getattr(Config, 'NO_OLLAMA', False)
        orig_llm_mode = getattr(Config, 'LLM_MODE', 'ollama')
        
        try:
            Config.NO_OLLAMA = False
            Config.LLM_MODE = "ollama"
            
            from wyzer.core.orchestrator import _get_llm_client
            client = _get_llm_client()
            
            self.assertIsInstance(client, OllamaClient)
        finally:
            Config.NO_OLLAMA = orig_no_ollama
            Config.LLM_MODE = orig_llm_mode
    
    def test_get_llm_client_returns_llamacpp_when_mode_set(self):
        """Test _get_llm_client returns LlamaCppClient when mode is llamacpp."""
        from wyzer.core.config import Config
        from wyzer.brain.llamacpp_client import LlamaCppClient
        
        # Save original values
        orig_no_ollama = getattr(Config, 'NO_OLLAMA', False)
        orig_llm_mode = getattr(Config, 'LLM_MODE', 'ollama')
        
        try:
            Config.NO_OLLAMA = False
            Config.LLM_MODE = "llamacpp"
            
            from wyzer.core.orchestrator import _get_llm_client
            client = _get_llm_client()
            
            self.assertIsInstance(client, LlamaCppClient)
        finally:
            Config.NO_OLLAMA = orig_no_ollama
            Config.LLM_MODE = orig_llm_mode
    
    def test_is_llm_available(self):
        """Test _is_llm_available returns correct values."""
        from wyzer.core.config import Config
        
        orig_no_ollama = getattr(Config, 'NO_OLLAMA', False)
        orig_llm_mode = getattr(Config, 'LLM_MODE', 'ollama')
        
        try:
            from wyzer.core.orchestrator import _is_llm_available
            
            Config.NO_OLLAMA = False
            Config.LLM_MODE = "ollama"
            self.assertTrue(_is_llm_available())
            
            Config.LLM_MODE = "llamacpp"
            self.assertTrue(_is_llm_available())
            
            Config.LLM_MODE = "off"
            self.assertFalse(_is_llm_available())
            
            Config.NO_OLLAMA = True
            Config.LLM_MODE = "ollama"
            self.assertFalse(_is_llm_available())
        finally:
            Config.NO_OLLAMA = orig_no_ollama
            Config.LLM_MODE = orig_llm_mode


if __name__ == "__main__":
    unittest.main()
