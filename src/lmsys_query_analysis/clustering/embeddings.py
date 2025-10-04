"""Embeddings generation using sentence-transformers or OpenAI.

Optimizations:
- Default to OpenAI embeddings for quality and consistency
- Async OpenAI requests with anyio + AsyncOpenAI client
- Concurrency-limited batch requests with ordered assembly
"""

from typing import List, Optional
import os
import time
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import anyio
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers or OpenAI."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        provider: str = "openai",
        api_key: Optional[str] = None,
        concurrency: int = 100,
        request_timeout: float = 30.0,
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
        self._async_client = None
        self.concurrency = max(1, int(concurrency))
        self.request_timeout = float(request_timeout)

        if provider == "openai":
            from openai import OpenAI, AsyncOpenAI

            self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self._async_client = AsyncOpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )

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
            # Use async path for OpenAI to parallelize batch requests
            return self._generate_openai_embeddings_async(
                texts, batch_size, show_progress
            )
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
                    total=len(texts),
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
                len(texts),
                elapsed,
                rate,
                self.model_name,
            )
        return embeddings

    async def _async_openai_batches(
        self, texts: List[str], batch_size: int, progress_task, progress
    ) -> List[List[float]]:
        """Run OpenAI embedding requests concurrently preserving order."""
        assert self._async_client is not None

        # Split into batches with original indices
        batches: list[tuple[int, List[str]]] = []
        for i in range(0, len(texts), batch_size):
            batches.append((i, texts[i : i + batch_size]))

        results: list[Optional[List[List[float]]]] = [None] * len(batches)
        semaphore = anyio.Semaphore(self.concurrency)

        async def worker(idx: int, payload: List[str]):
            backoff = 1.0
            for attempt in range(5):
                try:
                    async with semaphore:
                        resp = await self._async_client.embeddings.create(
                            input=payload,
                            model=self.model_name,
                        )
                    embeddings = [item.embedding for item in resp.data]
                    results[idx] = embeddings
                    if progress_task is not None:
                        progress.update(progress_task, advance=len(payload))
                    return
                except Exception:
                    # Simple exponential backoff
                    await anyio.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
            # If all retries fail, raise
            raise

        async with anyio.create_task_group() as tg:
            for j, (start, payload) in enumerate(batches):
                tg.start_soon(worker, j, payload)

        # Flatten preserving original order
        flat: list[List[float]] = []
        for r in results:
            if r is not None:
                flat.extend(r)
        return flat

    def _generate_openai_embeddings_async(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Generate embeddings using OpenAI API with anyio concurrency."""
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
                    f"[cyan]Generating embeddings with OpenAI {self.model_name}...",
                    total=len(texts),
                )
                all_embeddings = anyio.run(
                    self._async_openai_batches, texts, batch_size, task, progress
                )
        else:
            all_embeddings = anyio.run(
                self._async_openai_batches, texts, batch_size, None, None
            )

        elapsed = time.perf_counter() - start
        if len(texts) > 0:
            rate = len(texts) / elapsed if elapsed > 0 else float("inf")
            logger.info(
                "Embeddings(OpenAI): %s texts in %.2fs (%.1f/s) using %s (concurrency=%s)",
                len(texts),
                elapsed,
                rate,
                self.model_name,
                self.concurrency,
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
