"""Embeddings generation using sentence-transformers, OpenAI, or Cohere.

Optimizations:
- Default to OpenAI embeddings for quality and consistency
- Async OpenAI/Cohere requests with anyio + async clients
- Concurrency-limited batch requests with ordered assembly
- Cohere embed-v4.0 supports Matryoshka output dimensions
"""

from typing import List, Optional, Literal
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

# Type definitions for Cohere models
CohereModel = Literal["embed-v4.0"]

# Matryoshka embedding dimensions for Cohere v4
CohereOutputDimension = Literal[256, 512, 1024, 1536]


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers, OpenAI, or Cohere."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        provider: str = "openai",
        api_key: Optional[str] = None,
        concurrency: int = 100,
        request_timeout: float = 30.0,
        output_dimension: Optional[int] = None,
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
            from openai import OpenAI, AsyncOpenAI

            self.openai_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self._async_client = AsyncOpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
        elif provider == "cohere":
            import cohere

            # Validate Cohere v4 model and dimensions
            if model_name not in ["embed-v4.0"]:
                raise ValueError("Only Cohere v4 supported: model='embed-v4.0'")
            if output_dimension and output_dimension not in [256, 512, 1024, 1536]:
                raise ValueError(
                    f"Cohere v4 output_dimension must be one of 256, 512, 1024, 1536. Got: {output_dimension}"
                )
            # Default to 256 if not specified (Matryoshka)
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
        # Filter out empty/invalid texts and track original indices
        filtered_texts = []
        original_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():  # Skip empty or whitespace-only strings
                filtered_texts.append(text.strip())
                original_indices.append(i)
        
        if not filtered_texts:
            # All texts were empty, return zero embeddings
            logger = logging.getLogger("lmsys")
            logger.warning("All input texts were empty, returning zero embeddings")
            dim = self.get_embedding_dim()
            return np.zeros((len(texts), dim))
        
        # Generate embeddings for valid texts
        if self.provider == "openai":
            # Use async path for OpenAI to parallelize batch requests
            valid_embeddings = self._generate_openai_embeddings_async(
                filtered_texts, batch_size, show_progress
            )
        elif self.provider == "cohere":
            # Use async path for Cohere to parallelize batch requests
            valid_embeddings = self._generate_cohere_embeddings_async(
                filtered_texts, batch_size, show_progress
            )
        else:
            valid_embeddings = self._generate_st_embeddings(filtered_texts, batch_size, show_progress)
        
        # If all texts were valid, return as-is
        if len(filtered_texts) == len(texts):
            return valid_embeddings
        
        # Otherwise, create full array with zero embeddings for invalid texts
        logger = logging.getLogger("lmsys")
        logger.warning(f"Filtered out {len(texts) - len(filtered_texts)} empty/invalid texts")
        
        full_embeddings = np.zeros((len(texts), valid_embeddings.shape[1]))
        for i, orig_idx in enumerate(original_indices):
            full_embeddings[orig_idx] = valid_embeddings[i]
        
        return full_embeddings

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
            last_exception = None
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
                except Exception as e:
                    last_exception = e
                    # Simple exponential backoff
                    await anyio.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
            # If all retries fail, raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("All retries failed but no exception captured")

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
        import asyncio

        logger = logging.getLogger("lmsys")
        start = time.perf_counter()

        async def _run():
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
                    return await self._async_openai_batches(texts, batch_size, task, progress)
            else:
                return await self._async_openai_batches(texts, batch_size, None, None)

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - run in a separate thread with its own event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run())
                all_embeddings = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            all_embeddings = asyncio.run(_run())

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

    async def _async_cohere_batches(
        self, texts: List[str], batch_size: int, progress_task, progress
    ) -> List[List[float]]:
        """Run Cohere embedding requests concurrently preserving order."""
        assert self._async_client is not None

        # Split into batches with original indices
        batches: list[tuple[int, List[str]]] = []
        for i in range(0, len(texts), batch_size):
            batches.append((i, texts[i : i + batch_size]))

        results: list[Optional[List[List[float]]]] = [None] * len(batches)
        semaphore = anyio.Semaphore(self.concurrency)

        async def worker(idx: int, payload: List[str]):
            backoff = 1.0
            last_exception = None
            for attempt in range(5):
                try:
                    async with semaphore:
                        resp = await self._async_client.embed(
                            texts=payload,
                            model=self.model_name,
                            input_type="clustering",  # Optimize for clustering
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
                    # Simple exponential backoff
                    await anyio.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
            # If all retries fail, raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("All retries failed but no exception captured")

        async with anyio.create_task_group() as tg:
            for j, (start, payload) in enumerate(batches):
                tg.start_soon(worker, j, payload)

        # Flatten preserving original order
        flat: list[List[float]] = []
        for r in results:
            if r is not None:
                flat.extend(r)
        return flat

    def _generate_cohere_embeddings_async(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """Generate embeddings using Cohere API with anyio concurrency."""
        import asyncio

        logger = logging.getLogger("lmsys")
        start = time.perf_counter()

        async def _run():
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                ) as progress:
                    task = progress.add_task(
                        f"[cyan]Generating embeddings with Cohere {self.model_name}...",
                        total=len(texts),
                    )
                    return await self._async_cohere_batches(texts, batch_size, task, progress)
            else:
                return await self._async_cohere_batches(texts, batch_size, None, None)

        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - run in a separate thread with its own event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run())
                all_embeddings = future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            all_embeddings = asyncio.run(_run())

        elapsed = time.perf_counter() - start
        if len(texts) > 0:
            rate = len(texts) / elapsed if elapsed > 0 else float("inf")
            logger.info(
                "Embeddings(Cohere): %s texts in %.2fs (%.1f/s) using %s (concurrency=%s)",
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
        elif self.provider == "cohere":
            # Return the configured output dimension for Cohere v4
            return self.output_dimension if self.output_dimension else 256
        else:
            self.load_model()
            return self.model.get_sentence_embedding_dimension()
