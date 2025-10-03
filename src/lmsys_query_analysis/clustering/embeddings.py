"""Embeddings generation using sentence-transformers."""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding generator.

        Args:
            model_name: SentenceTransformer model name (default: all-MiniLM-L6-v2)
                       This is a fast, efficient model good for clustering
        """
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
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
        self.load_model()

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

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings from this model."""
        self.load_model()
        return self.model.get_sentence_embedding_dimension()
