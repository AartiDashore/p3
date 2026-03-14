"""
Unit tests for LLMClient.

@author: Aarti Dashore
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid=190380
@version: 3.0.0+w26

All tests use mocks so they pass without Ollama running.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from retrieval.llm import LLMClient


@pytest.fixture
def llm():
    """Default LLMClient instance."""
    return LLMClient(base_url="http://localhost:11434", model="qwen2.5:3b")


@pytest.fixture
def openai_llm():
    """LLMClient configured for OpenAI-style API."""
    return LLMClient(
        base_url="https://api.openai.com/v1",
        model="gpt-3.5-turbo",
        api_key="sk-test",
    )


# ── Initialization ─────────────────────────────────────────
def test_default_init():
    """Test default initialization values."""
    client = LLMClient()
    assert client.base_url == "http://localhost:11434"
    assert client.model == "qwen2.5:3b"
    assert client.timeout == 180.0


def test_custom_init():
    """Test custom initialization values."""
    client = LLMClient(
        base_url="http://myserver:11434",
        model="llama3",
        api_key="mykey",
        timeout=60.0,
    )
    assert client.base_url == "http://myserver:11434"
    assert client.model == "llama3"
    assert client.api_key == "mykey"
    assert client.timeout == 60.0


def test_trailing_slash_stripped():
    """Base URL trailing slash should be stripped."""
    client = LLMClient(base_url="http://localhost:11434/")
    assert client.base_url == "http://localhost:11434"


def test_api_key_defaults_to_ollama():
    """When no api_key, defaults to 'ollama'."""
    client = LLMClient()
    assert client.api_key == "ollama"


def test_custom_api_key():
    """Custom api_key is stored."""
    client = LLMClient(api_key="sk-secret")
    assert client.api_key == "sk-secret"


# ── generate() ────────────────────────────────────────────
def test_generate_returns_string(llm):
    """Test that generate() returns a string response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "Test answer"}}]}

    with patch("httpx.post", return_value=mock_response):
        result = llm.generate("What is AI?")

    assert result == "Test answer"
    assert isinstance(result, str)


def test_generate_with_system_prompt(llm):
    """Test generate() includes system prompt in messages."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Response with system"}}]
    }

    with patch("httpx.post", return_value=mock_response) as mock_post:
        llm.generate("Hello", system_prompt="You are a pirate.")

    call_kwargs = mock_post.call_args[1]
    messages = call_kwargs["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a pirate."
    assert messages[1]["role"] == "user"


def test_generate_without_system_prompt(llm):
    """Test generate() without system prompt only has user message."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "No system"}}]}

    with patch("httpx.post", return_value=mock_response) as mock_post:
        llm.generate("Hello")

    messages = mock_post.call_args[1]["json"]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_generate_sends_correct_url(llm):
    """Test that generate() posts to /v1/chat/completions."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch("httpx.post", return_value=mock_response) as mock_post:
        llm.generate("test")

    url = mock_post.call_args[0][0]
    assert url == "http://localhost:11434/v1/chat/completions"


def test_generate_sends_temperature(llm):
    """Test that temperature is passed in payload."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch("httpx.post", return_value=mock_response) as mock_post:
        llm.generate("test", temperature=0.3)

    payload = mock_post.call_args[1]["json"]
    assert payload["temperature"] == 0.3


def test_generate_sends_max_tokens(llm):
    """Test that max_tokens is passed in payload."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch("httpx.post", return_value=mock_response) as mock_post:
        llm.generate("test", max_tokens=250)

    payload = mock_post.call_args[1]["json"]
    assert payload["max_tokens"] == 250


def test_generate_timeout_raises_runtime_error(llm):
    """Test that TimeoutException raises RuntimeError."""
    with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
        with pytest.raises(RuntimeError, match="timed out"):
            llm.generate("test")


def test_generate_http_status_error_raises_runtime_error(llm):
    """Test that HTTPStatusError raises RuntimeError."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch(
        "httpx.post",
        side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response),
    ):
        with pytest.raises(RuntimeError, match="LLM request failed"):
            llm.generate("test")


def test_generate_request_error_raises_runtime_error(llm):
    """Test that RequestError raises RuntimeError."""
    with patch("httpx.post", side_effect=httpx.RequestError("connection refused")):
        with pytest.raises(RuntimeError, match="LLM request error"):
            llm.generate("test")


def test_generate_malformed_response_raises_runtime_error(llm):
    """Test that malformed JSON response raises RuntimeError."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"unexpected": "format"}
    mock_response.text = '{"unexpected": "format"}'

    with patch("httpx.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Unexpected response format"):
            llm.generate("test")


def test_generate_empty_choices_raises_runtime_error(llm):
    """Test that empty choices list raises RuntimeError."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}
    mock_response.text = '{"choices": []}'

    with patch("httpx.post", return_value=mock_response):
        with pytest.raises(RuntimeError):
            llm.generate("test")


# ── is_available() ─────────────────────────────────────────
def test_is_available_returns_true_when_server_up(llm):
    """Test is_available() returns True when server responds 200."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.get", return_value=mock_response):
        assert llm.is_available() is True


def test_is_available_returns_false_when_server_down(llm):
    """Test is_available() returns False on connection error."""
    with patch("httpx.get", side_effect=httpx.RequestError("connection refused")):
        assert llm.is_available() is False


def test_is_available_returns_false_on_timeout(llm):
    """Test is_available() returns False on timeout."""
    with patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
        assert llm.is_available() is False


def test_is_available_returns_false_on_non_200(llm):
    """Test is_available() returns False when status != 200."""
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("httpx.get", return_value=mock_response):
        assert llm.is_available() is False


def test_is_available_checks_correct_url(llm):
    """Test is_available() checks /v1/models endpoint."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.get", return_value=mock_response) as mock_get:
        llm.is_available()

    assert mock_get.call_args[0][0] == "http://localhost:11434/v1/models"


def test_is_available_openai_url(openai_llm):
    """Test is_available() uses correct URL for OpenAI client."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("httpx.get", return_value=mock_response) as mock_get:
        openai_llm.is_available()

    assert mock_get.call_args[0][0] == "https://api.openai.com/v1/v1/models"
