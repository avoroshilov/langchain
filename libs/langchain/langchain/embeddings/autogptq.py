import torch
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field

from langchain.embeddings.base import Embeddings

class AutoGPTQEmbeddings(BaseModel, Embeddings):
    """AutoGPTQ embedding models.

    Example:
        .. code-block:: python

            from langchain.embeddings import AutoGPTQEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = AutoGPTQEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    pipeline: Any  #: :meta private:
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""

    def __init__(self, pipeline, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.pipeline = pipeline

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a AutoGPTQ transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        with torch.no_grad():
            tokenizer = self.pipeline.tokenizer
            # TODO avoroshilov: to CUDA or not to CUDA
            input_ids = tokenizer(texts, return_tensors='pt').input_ids.cuda()
            embeddings = self.pipeline.model.get_input_embeddings()(input_ids).flatten()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a AutoGPTQ transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        with torch.no_grad():
            tokenizer = self.pipeline.tokenizer
            # TODO avoroshilov: to CUDA or not to CUDA
            input_ids = tokenizer(text, return_tensors='pt').input_ids.cuda()
            embeddings = self.pipeline.model.get_input_embeddings()(input_ids).flatten()
        return embeddings.tolist()
