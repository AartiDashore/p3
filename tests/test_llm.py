"""Tests for the LLM client module using mocked HTTP requests.
@author: Aarti Dashore, Sebastian Silva Arcos
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 4.0.0+w26
"""

from unittest.mock import Mock, patch

import pytest

from retrieval.llm import LLMClient

"""
Helper function to create a mock httpx response with OpenAI-style JSON.
"""


def _mock_response(content: str, status_code: int = 200):
    """Build a mock httpx response that returns an OpenAI-style JSON body."""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock_response.text = content
    # raise_for_status() should be a no-op for success responses
    mock_response.raise_for_status = Mock()
    return mock_response


"""
Ollama mode
"""


class TestOllamaMode:
    def setup_method(self):
        self.client = LLMClient(
            base_url="http://localhost:11434",
            model="qwen2.5:3b",
        )

    @patch("retrieval.llm.httpx.post")
    def test_generate_returns_text(self, mock_post):
        mock_post.return_value = _mock_response("Paris is the capital of France.")

        result = self.client.generate("What is the capital of France?")

        assert result == "Paris is the capital of France."

    @patch("retrieval.llm.httpx.post")
    def test_generate_correct_endpoint(self, mock_post):
        mock_post.return_value = _mock_response("answer")

        self.client.generate("hello")

        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/v1/chat/completions"

    @patch("retrieval.llm.httpx.post")
    def test_generate_default_api_key_is_ollama(self, mock_post):
        mock_post.return_value = _mock_response("answer")

        self.client.generate("hello")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer ollama"

    @patch("retrieval.llm.httpx.post")
    def test_generate_with_system_prompt(self, mock_post):
        mock_post.return_value = _mock_response("Sure!")

        self.client.generate("Help me.", system_prompt="You are a helpful assistant.")

        payload = mock_post.call_args[1]["json"]
        assert payload["messages"][0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert payload["messages"][1] == {"role": "user", "content": "Help me."}

    @patch("retrieval.llm.httpx.post")
    def test_generate_without_system_prompt(self, mock_post):
        mock_post.return_value = _mock_response("Sure!")

        self.client.generate("Help me.")

        payload = mock_post.call_args[1]["json"]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    @patch("retrieval.llm.httpx.post")
    def test_generate_payload_keys(self, mock_post):
        mock_post.return_value = _mock_response("answer")

        self.client.generate("test", temperature=0.5, max_tokens=200)

        payload = mock_post.call_args[1]["json"]
        assert "model" in payload
        assert "messages" in payload
        assert "temperature" in payload
        assert "max_tokens" in payload
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 200


"""
OpenAI mode
"""


class TestOpenAIMode:
    def setup_method(self):
        self.client = LLMClient(
            base_url="https://api.openai.com",
            model="gpt-4o-mini",
            api_key="sk-test-key",
        )

    @patch("retrieval.llm.httpx.post")
    def test_generate_correct_endpoint(self, mock_post):
        mock_post.return_value = _mock_response("answer")

        self.client.generate("hello")

        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"

    @patch("retrieval.llm.httpx.post")
    def test_generate_uses_provided_api_key(self, mock_post):
        mock_post.return_value = _mock_response("answer")

        self.client.generate("hello")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer sk-test-key"

    @patch("retrieval.llm.httpx.post")
    def test_generate_returns_text(self, mock_post):
        mock_post.return_value = _mock_response("The answer is 42.")

        result = self.client.generate("What is the answer?")

        assert result == "The answer is 42."


"""
Error handling
"""


class TestErrorHandling:
    def setup_method(self):
        self.client = LLMClient()

    @patch("retrieval.llm.httpx.post")
    def test_timeout_raises_runtime_error(self, mock_post):
        import httpx

        mock_post.side_effect = httpx.TimeoutException("timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            self.client.generate("hello")

    @patch("retrieval.llm.httpx.post")
    def test_http_error_raises_runtime_error(self, mock_post):
        import httpx

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 error", request=Mock(), response=mock_response
        )

        with pytest.raises(RuntimeError, match="500"):
            self.client.generate("hello")

    @patch("retrieval.llm.httpx.post")
    def test_malformed_response_raises_runtime_error(self, mock_post):
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.text = '{"unexpected": "format"}'
        mock_post.return_value = mock_response

        with pytest.raises(RuntimeError, match="Unexpected response format"):
            self.client.generate("hello")

    @patch("retrieval.llm.httpx.post")
    def test_request_error_raises_runtime_error(self, mock_post):
        import httpx

        mock_post.side_effect = httpx.RequestError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            self.client.generate("hello")


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def setup_method(self):
        self.client = LLMClient()

    @patch("retrieval.llm.httpx.get")
    def test_returns_true_when_server_responds(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert self.client.is_available() is True

    @patch("retrieval.llm.httpx.get")
    def test_returns_false_when_server_down(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.RequestError("connection refused")

        assert self.client.is_available() is False

    @patch("retrieval.llm.httpx.get")
    def test_returns_false_on_timeout(self, mock_get):
        import httpx

        mock_get.side_effect = httpx.TimeoutException("timed out")

        assert self.client.is_available() is False

    @patch("retrieval.llm.httpx.get")
    def test_checks_correct_url(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        self.client.is_available()

        assert mock_get.call_args[0][0] == "http://localhost:11434/v1/models"
