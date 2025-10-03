"""Embeddings generation using sentence-transformers or OpenAI."""
from typing import List, Optional
import os
import time
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers or OpenAI."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: str = "sentence-transformers",
        api_key: Optional[str] = None,
    ):
        """Initialize embedding generator.

        Args:
            model_name: Model name (default: all-MiniLM-L6-v2 for sentence-transformers,
                       text-embedding-3-small for openai)
            provider: "sentence-transformers" or "openai"
            api_key: API key for OpenAI (or set OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.provider = provider
        self.model = None
        self.openai_client = None

        if provider == "openai":
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def load_model(self):
        """Load the sentence transformer model."""
        if self.provider == "sentence-transformers" and self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if self.provider == "openai":
            return self._generate_openai_embeddings(texts, batch_size, show_progress)
        else:
            return self._generate_st_embeddings(texts, batch_size, show_progress)

    def _generate_st_embeddings(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        self.load_model()
        logger = logging.getLogger("lmsys")
        start = time.perf_counter()
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Generating embeddings with {self.model_name}...",
                    total=len(texts)
                )

                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

                progress.update(task, completed=len(texts))
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        elapsed = time.perf_counter() - start
        if len(texts) > 0:
            rate = len(texts) / elapsed if elapsed > 0 else float("inf")
            logger.info(
                "Embeddings(ST): %s texts in %.2fs (%.1f/s) using %s",
                len(texts), elapsed, rate, self.model_name,
            )
        return embeddings

    def _generate_openai_embeddings(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        logger = logging.getLogger("lmsys")
        all_embeddings = []
        start = time.perf_counter()

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Generating embeddings with OpenAI {self.model_name}...",
                    total=len(texts)
                )

                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = self.openai_client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    progress.update(task, advance=len(batch))
        else:
            # Process in batches without progress
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
        elapsed = time.perf_counter() - start
        if len(texts) > 0:
            rate = len(texts) / elapsed if elapsed > 0 else float("inf")
            logger.info(
                "Embeddings(OpenAI): %s texts in %.2fs (%.1f/s) using %s",
                len(texts), elapsed, rate, self.model_name,
            )
        return np.array(all_embeddings)

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings from this model."""
        if self.provider == "openai":
            # OpenAI embedding dimensions
            if "text-embedding-3-small" in self.model_name:
                return 1536
            elif "text-embedding-3-large" in self.model_name:
                return 3072
            elif "text-embedding-ada-002" in self.model_name:
                return 1536
            else:
                return 1536  # Default
        else:
            self.load_model()
            return self.model.get_sentence_embedding_dimension()
