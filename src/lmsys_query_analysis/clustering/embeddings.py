"""Embeddings generation using sentence-transformers, OpenAI, or Cohere.

Optimizations:
- Default to OpenAI embeddings for quality and consistency
- Async OpenAI/Cohere requests with anyio + async clients
- Concurrency-limited batch requests with ordered assembly
- Cohere embed-v4.0 supports Matryoshka output dimensions
"""

import logging
import os
import time
from typing import Literal

import anyio
import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from sentence_transformers import SentenceTransformer

CohereModel = Literal["embed-v4.0"]

CohereOutputDimension = Literal[256, 512, 1024, 1536]


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers, OpenAI, or Cohere."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        provider: str = "openai",
        api_key: str | None = None,
        concurrency: int = 50,
        request_timeout: float = 30.0,
        output_dimension: int | None = None,
    ):
        """Initialize embedding generator.

        Args:
            model_name: Model name (default: text-embedding-3-small for openai,
                       all-MiniLM-L6-v2 for sentence-transformers, embed-v4.0 for cohere)
            provider: "sentence-transformers", "openai", or "cohere"
            api_key: API key for OpenAI/Cohere (or set OPENAI_API_KEY/COHERE_API_KEY env var)
            output_dimension: For Cohere v4, use 512 or 1024 (Matryoshka dimensions)
        """
        self.model_name = model_name
        self.provider = provider
        self.model = None
        self.openai_client = None
        self._async_client = None
        self.concurrency = max(1, int(concurrency))
        self.request_timeout = float(request_timeout)
        self.output_dimension = output_dimension

        if provider == "openai":
            from openai import AsyncOpenAI, OpenAI

            self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self._async_client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif provider == "cohere":
            import cohere

            if model_name not in ["embed-v4.0"]:
                raise ValueError("Only Cohere v4 supported: model='embed-v4.0'")
            if output_dimension and output_dimension not in [256, 512, 1024, 1536]:
                raise ValueError(
                    f"Cohere v4 output_dimension must be one of 256, 512, 1024, 1536. Got: {output_dimension}"
                )
            if not output_dimension:
                self.output_dimension = 256

            self._async_client = cohere.AsyncClientV2(
                api_key=api_key or os.getenv("COHERE_API_KEY")
            )

    def load_model(self):
        """Load the sentence transformer model."""
        if self.provider == "sentence-transformers" and self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(
        self,
        texts: list[str],
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
        filtered_texts = []
        original_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                filtered_texts.append(text.strip())
                original_indices.append(i)

        if not filtered_texts:
            logger = logging.getLogger("lmsys")
            logger.warning("All input texts were empty, returning zero embeddings")
            dim = self.get_embedding_dim()
            return np.zeros((len(texts), dim))

        logger = logging.getLogger("lmsys")
        start = time.perf_counter()

        if self.provider == "openai":
            valid_embeddings_list = anyio.run(
                self._async_openai_batches, filtered_texts, batch_size, None, None
            )
            valid_embeddings = np.array(valid_embeddings_list)
            elapsed = time.perf_counter() - start
            if len(filtered_texts) > 0:
                rate = len(filtered_texts) / elapsed if elapsed > 0 else float("inf")
                logger.info(
                    "Embeddings(OpenAI): %s texts in %.2fs (%.1f/s) using %s (concurrency=%s)",
                    len(filtered_texts),
                    elapsed,
                    rate,
                    self.model_name,
                    self.concurrency,
                )
        elif self.provider == "cohere":
            valid_embeddings_list = anyio.run(
                self._async_cohere_batches, filtered_texts, batch_size, None, None
            )
            valid_embeddings = np.array(valid_embeddings_list)
            elapsed = time.perf_counter() - start
            if len(filtered_texts) > 0:
                rate = len(filtered_texts) / elapsed if elapsed > 0 else float("inf")
                logger.info(
                    "Embeddings(Cohere): %s texts in %.2fs (%.1f/s) using %s (concurrency=%s)",
                    len(filtered_texts),
                    elapsed,
                    rate,
                    self.model_name,
                    self.concurrency,
                )
        else:
            valid_embeddings = self._generate_st_embeddings(
                filtered_texts, batch_size, show_progress
            )

        if len(filtered_texts) == len(texts):
            return valid_embeddings

        logger.warning(f"Filtered out {len(texts) - len(filtered_texts)} empty/invalid texts")

        full_embeddings = np.zeros((len(texts), valid_embeddings.shape[1]))
        for i, orig_idx in enumerate(original_indices):
            full_embeddings[orig_idx] = valid_embeddings[i]

        return full_embeddings

    def _generate_st_embeddings(
        self,
        texts: list[str],
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

    async def generate_embeddings_async(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for a list of texts asynchronously.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        filtered_texts = []
        original_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                filtered_texts.append(text.strip())
                original_indices.append(i)

        if not filtered_texts:
            logger = logging.getLogger("lmsys")
            logger.warning("All input texts were empty, returning zero embeddings")
            dim = self.get_embedding_dim()
            return np.zeros((len(texts), dim))

        logger = logging.getLogger("lmsys")
        start = time.perf_counter()

        if self.provider == "openai":
            valid_embeddings_list = await self._async_openai_batches(
                filtered_texts, batch_size, None, None
            )
            valid_embeddings = np.array(valid_embeddings_list)
            elapsed = time.perf_counter() - start
            if len(filtered_texts) > 0:
                rate = len(filtered_texts) / elapsed if elapsed > 0 else float("inf")
                logger.info(
                    "Embeddings(OpenAI): %s texts in %.2fs (%.1f/s) using %s (concurrency=%s)",
                    len(filtered_texts),
                    elapsed,
                    rate,
                    self.model_name,
                    self.concurrency,
                )
        elif self.provider == "cohere":
            valid_embeddings_list = await self._async_cohere_batches(
                filtered_texts, batch_size, None, None
            )
            valid_embeddings = np.array(valid_embeddings_list)
            elapsed = time.perf_counter() - start
            if len(filtered_texts) > 0:
                rate = len(filtered_texts) / elapsed if elapsed > 0 else float("inf")
                logger.info(
                    "Embeddings(Cohere): %s texts in %.2fs (%.1f/s) using %s (concurrency=%s)",
                    len(filtered_texts),
                    elapsed,
                    rate,
                    self.model_name,
                    self.concurrency,
                )
        else:
            valid_embeddings = self._generate_st_embeddings(
                filtered_texts, batch_size, show_progress
            )

        if len(filtered_texts) == len(texts):
            return valid_embeddings

        logger.warning(f"Filtered out {len(texts) - len(filtered_texts)} empty/invalid texts")

        full_embeddings = np.zeros((len(texts), valid_embeddings.shape[1]))
        for i, orig_idx in enumerate(original_indices):
            full_embeddings[orig_idx] = valid_embeddings[i]

        return full_embeddings

    async def _async_openai_batches(
        self, texts: list[str], batch_size: int, progress_task, progress
    ) -> list[list[float]]:
        """Run OpenAI embedding requests concurrently preserving order."""
        assert self._async_client is not None

        batches: list[tuple[int, list[str]]] = []
        for i in range(0, len(texts), batch_size):
            batches.append((i, texts[i : i + batch_size]))

        results: list[list[list[float]] | None] = [None] * len(batches)
        semaphore = anyio.Semaphore(self.concurrency)

        async def worker(idx: int, payload: list[str]):
            backoff = 1.0
            last_exception = None
            for _attempt in range(5):
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
                except Exception as e:
                    last_exception = e
                    await anyio.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("All retries failed but no exception captured")

        async with anyio.create_task_group() as tg:
            for j, (_start, payload) in enumerate(batches):
                tg.start_soon(worker, j, payload)

        flat: list[list[float]] = []
        for r in results:
            if r is not None:
                flat.extend(r)
        return flat

    async def _async_cohere_batches(
        self, texts: list[str], batch_size: int, progress_task, progress
    ) -> list[list[float]]:
        """Run Cohere embedding requests concurrently preserving order."""
        assert self._async_client is not None

        batches: list[tuple[int, list[str]]] = []
        for i in range(0, len(texts), batch_size):
            batches.append((i, texts[i : i + batch_size]))

        results: list[list[list[float]] | None] = [None] * len(batches)
        semaphore = anyio.Semaphore(self.concurrency)

        async def worker(idx: int, payload: list[str]):
            backoff = 1.0
            last_exception = None
            for _attempt in range(5):
                try:
                    async with semaphore:
                        resp = await self._async_client.embed(
                            texts=payload,
                            model=self.model_name,
                            input_type="clustering",
                            embedding_types=["float"],
                            output_dimension=self.output_dimension,
                        )
                    embeddings = resp.embeddings.float
                    results[idx] = embeddings
                    if progress_task is not None:
                        progress.update(progress_task, advance=len(payload))
                    return
                except Exception as e:
                    last_exception = e
                    await anyio.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("All retries failed but no exception captured")

        async with anyio.create_task_group() as tg:
            for j, (_start, payload) in enumerate(batches):
                tg.start_soon(worker, j, payload)

        flat: list[list[float]] = []
        for r in results:
            if r is not None:
                flat.extend(r)
        return flat

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings from this model."""
        if self.provider == "openai":
            if "text-embedding-3-small" in self.model_name:
                return 1536
            elif "text-embedding-3-large" in self.model_name:
                return 3072
            elif "text-embedding-ada-002" in self.model_name:
                return 1536
            else:
                return 1536
        elif self.provider == "cohere":
            return self.output_dimension if self.output_dimension else 256
        else:
            self.load_model()
            return self.model.get_sentence_embedding_dimension()
