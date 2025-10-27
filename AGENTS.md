# Repository Guidelines

## Purpose & Goals

This repository provides agents with the ability to perform comprehensive data analysis on LLM Sys queries. The primary goal is to enable systematic investigation of how people are using LLM Sys through:

- **Data Loading**: Download and process the LMSYS-1M dataset from Hugging Face
- **Clustering Analysis**: Group similar queries using embeddings and clustering algorithms (KMeans, HDBSCAN)
- **Hierarchical Organization**: Create multi-level cluster hierarchies using LLM-driven merging (Anthropic Clio methodology)
- **Cluster Investigation**: Explore and summarize query clusters to identify patterns and usage trends
- **Hypothesis Generation**: Discover interesting insights about user behavior, query patterns, and system interactions

The analysis workflow supports agents in forming data-driven hypotheses about LLM usage patterns, query complexity, user intent, and system performance across different query types.

All of these capabilities are accessible through the command-line interface (CLI) using the `lmsys` command, making it easy for agents to perform comprehensive data analysis workflows.

If agents identify gaps in functionality or need additional tools to enhance their analysis capabilities, they should suggest these improvements to the user for potential implementation.

## Project Structure & Modules

- `src/lmsys_query_analysis/`: Core Python package.
  - `cli/`: Typer CLI (`main.py`) with commands: `load`, `cluster kmeans`, `cluster hdbscan`, `merge-clusters`, `runs`, `list`, `list-clusters`, `summarize`, `inspect`, `export`, `search`, `clear`, `backfill-chroma`.
  - `db/`: Persistence layer (`models.py`, `connection.py`, `loader.py`, `chroma.py`).
  - `clustering/`: Embeddings, KMeans/HDBSCAN clustering, hierarchical merging (`hierarchy.py`), and LLM summarization.
- `tests/`: Pytest suite covering CLI and data layer.
- `pyproject.toml`: Project metadata, dependencies, and console script (`lmsys`).

## Build, Test, and Dev Commands

- Install deps: `uv sync` — resolve and create the virtual env (uses `uv.lock`).
- Run CLI: `uv run lmsys --help` — invoke commands in the venv.
- Load data: `uv run lmsys load --limit 10000 --use-chroma` — download HF dataset, write SQLite, add embeddings to Chroma.
- Cluster: `uv run lmsys cluster kmeans --n-clusters 200 --use-chroma` — embed + KMeans + optional Chroma summaries.
- View runs: `uv run lmsys runs --latest` — show the most recent clustering run.
- Lint: `uv run ruff check .` — check code style; add `--fix` to auto-fix issues.
- Format: `uv run ruff format .` — format code with Ruff formatter.
- Tests: `uv run pytest -v` — run all tests; add `--cov=src/lmsys_query_analysis` for coverage.
- Single test: `uv run pytest tests/path/test_file.py::test_function -v` — run one specific test.
- Smoke tests: `uv run pytest -m smoke -v` — run smoke tests (requires API keys).

## Coding Style & Naming

- Python 3.10+, 4‑space indentation, type hints throughout (`py.typed`).
- Modules/packages: `snake_case`; classes: `CapWords`; functions/vars: `snake_case`.
- Prefer small, focused functions; docstrings on public functions/classes.
- Rich terminal UX for CLI output; keep messages actionable and compact.

## Testing Guidelines

- Framework: `pytest`. Place tests under `tests/` with files named `test_*.py` and functions `test_*`.
- Use in‑memory SQLite for model tests; avoid network in unit tests.
- Add tests for new CLI flags and database behaviors; aim to keep existing coverage passing.

## Commit & Pull Requests

- Commits: imperative, concise, scoped (e.g., "Add kmeans run summary table"). Group related changes.
- PRs: include purpose, key changes, test plan (commands run and outputs), and screenshots/snippets for CLI tables where useful. Link issues when applicable and update `README.md` if user‑facing behavior changes.

## Security & Config Tips

- Hugging Face: `huggingface-cli login` and accept LMSYS-1M terms before `load`.
- API keys (LLM): set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `COHERE_API_KEY`, or `GROQ_API_KEY` in your shell, not in code.
- On this machine, API keys are exported in `~/.zshrc`. The CLI inherits them if you run via your shell (e.g., `uv run lmsys ...`). If running via another environment that doesn’t source `~/.zshrc`, explicitly export the keys in that session.
- Defaults: SQLite at `~/.lmsys-query-analysis/queries.db`; Chroma at `~/.lmsys-query-analysis/chroma`.
- **ALWAYS clear database before loading new data**: Use `uv run lmsys clear --yes` to avoid duplicates and source conflicts. Use `--db-path` and `--chroma-path` flags for custom database locations or testing with temp databases.

## Command Naming

- Cluster summarization command is `summarize` (not `describe-clusters`). Example:
  - `lmsys summarize <RUN_ID> --alias v1 --use-chroma`
