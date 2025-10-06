# Repository Guidelines

## Purpose & Goals

This repository provides **terminal-based CLI tools for agents to perform comprehensive data analysis on LMSYS queries**. The primary goal is to enable systematic investigation of how people use LLM systems through data-driven analysis workflows.

### Core Capabilities

- **Data Loading**: Download and process the LMSYS-1M dataset from Hugging Face
- **Clustering Analysis**: Group similar queries using embeddings and clustering algorithms (KMeans, HDBSCAN)
- **Hierarchical Organization**: Create multi-level topic hierarchies using LLM-driven merging (Anthropic Clio methodology)
- **Cluster Investigation**: Explore and summarize query clusters to identify patterns and usage trends
- **Contrastive Analysis**: Highlight what makes each cluster unique compared to neighbors
- **Semantic Search**: Navigate queries and clusters using natural language
- **Hypothesis Generation**: Discover insights about user behavior, query patterns, and LLM interactions

### Agent Workflow

Agents can use this tool to:

1. **Investigate LLM Usage Patterns**: Discover how users interact with different LLM systems
2. **Identify Common Use Cases**: Group similar queries to find patterns
3. **Generate Insights**: Use LLM-powered summarization to understand cluster characteristics
4. **Form Hypotheses**: Systematically explore data to develop testable hypotheses
5. **Export Findings**: Save analysis results for further investigation

All capabilities are accessible through the `lmsys` CLI command in a composable workflow: `load → cluster → summarize → merge-clusters → search → export`

### Extensibility

If agents identify gaps in functionality or need additional tools to enhance their analysis capabilities, they should suggest these improvements to the user for potential implementation.

## Project Structure & Modules

- `src/lmsys_query_analysis/`: Core Python package.
  - `cli/`: Typer CLI (`main.py`) with commands: `load`, `cluster kmeans`, `cluster hdbscan`, `merge-clusters`, `runs`, `list`, `list-clusters`, `summarize`, `inspect`, `export`, `search`, `clear`, `backfill-chroma`.
  - `db/`: Persistence layer (`models.py`, `connection.py`, `loader.py`, `chroma.py`).
  - `clustering/`: Embeddings, KMeans/HDBSCAN clustering, hierarchical merging (`hierarchy.py`), and LLM summarization.
- `tests/`: Pytest suite covering CLI and data layer.
- `pyproject.toml`: Project metadata, dependencies, and console script (`lmsys`).
- `docs/experiments/`: Documentation of clustering experiments and findings.

## Build, Test, and Dev Commands

- Install deps: `uv sync` — resolve and create the virtual env (uses `uv.lock`).
- Run CLI: `uv run lmsys --help` — invoke commands in the venv.
- Load data: `uv run lmsys load --limit 10000 --use-chroma` — download HF dataset, write SQLite, add embeddings to Chroma.
- Cluster: `uv run lmsys cluster kmeans --n-clusters 200 --use-chroma` — embed + KMeans + optional Chroma summaries.
- View runs: `uv run lmsys runs --latest` — show the most recent clustering run.
- Tests: `uv run pytest -v` — run all tests; add `--cov=src/lmsys_query_analysis` for coverage.

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
- API keys (LLM): set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY` in your shell, not in code.
- Defaults: SQLite at `~/.lmsys-query-analysis/queries.db`; Chroma at `~/.lmsys-query-analysis/chroma`.
