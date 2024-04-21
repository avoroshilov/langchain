from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_openai.embeddings.base import (
    AuthBaseEmbeddingsOpenAI,
    OpenAIEmbeddings,
)

__all__ = [
    "OpenAIEmbeddings",
    "AzureOpenAIEmbeddings",
]
