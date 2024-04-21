import os
from unittest.mock import MagicMock, patch

import pytest
import openai

from langchain_openai import (
    AuthBaseEmbeddingsOpenAI,
    OpenAIEmbeddings,
)

os.environ["OPENAI_API_KEY"] = "foo"


def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAIEmbeddings(model_kwargs={"model": "foo"})


def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAIEmbeddings(foo="bar")
    assert llm.model_kwargs == {"foo": "bar"}

@pytest.fixture
def mock_embedding():
    return {
        "data": [
            {
                "embedding": [],
            }
        ],
    }

def test_openai_embeddings_invoke_creds(mock_embedding) -> None:
    embeddings = OpenAIEmbeddings()
    auth_validated = False

    class TestAuthBase(AuthBaseEmbeddingsOpenAI):
        def validate_auth(self):
            nonlocal auth_validated
            auth_validated = True

    embeddings.auth_base = TestAuthBase()

    mock_client = MagicMock()
    mock_client.create.return_value = mock_embedding
    with patch.object(
        embeddings,
        "client",
        mock_client,
    ):
        embeddings.embed_query("Hello!")
    assert auth_validated
