# Installation

## Requirements

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Install Dependencies

```bash
# Clone the repository
git clone https://github.com/jxnl/lmsys-query-analysis.git
cd lmsys-query-analysis

# Install dependencies with uv
uv sync
```

## Setup

### 1. HuggingFace Authentication

The LMSYS-1M dataset is gated and requires authentication:

```bash
huggingface-cli login
```

Accept the terms at: [https://huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)

### 2. LLM API Key (Optional)

For cluster summarization, set an environment variable for your chosen provider:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI GPT
export OPENAI_API_KEY="sk-..."

# Groq
export GROQ_API_KEY="gsk_..."
```

## Verify Installation

```bash
# Check CLI is available
uv run lmsys --help

# Run tests
uv run pytest -v
```

## Default Paths

- SQLite Database: `~/.lmsys-query-analysis/queries.db`
- ChromaDB: `~/.lmsys-query-analysis/chroma/`

You can override these with `--db-path` and `--chroma-path` flags.
