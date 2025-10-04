# CLI Implementation

The CLI is built using [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/) for a modern terminal experience.

## Main Application

### Usage Example

```python
from lmsys_query_analysis.cli.main import app
import typer

if __name__ == "__main__":
    app()
```

---

## Command Reference

### Load Command

Loads queries from the LMSYS-1M dataset.

```python
@app.command()
def load(
    db_path: str = DEFAULT_DB_PATH,
    limit: int | None = None,
    use_chroma: bool = False,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_provider: str = "sentence-transformers",
    batch_size: int = 32,
):
    """Load queries from LMSYS-1M dataset."""
```

### Cluster Commands

#### KMeans

```python
@cluster_app.command("kmeans")
def cluster_kmeans(
    db_path: str = DEFAULT_DB_PATH,
    n_clusters: int = 100,
    use_chroma: bool = False,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_provider: str = "sentence-transformers",
    embed_batch_size: int = 32,
    mb_batch_size: int = 4096,
    chunk_size: int = 5000,
    description: str | None = None,
):
    """Run MiniBatchKMeans clustering."""
```

#### HDBSCAN

```python
@cluster_app.command("hdbscan")
def cluster_hdbscan(
    db_path: str = DEFAULT_DB_PATH,
    use_chroma: bool = False,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_provider: str = "sentence-transformers",
    embed_batch_size: int = 32,
    chunk_size: int = 5000,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    description: str | None = None,
):
    """Run HDBSCAN density-based clustering."""
```

### Summarize Command

```python
@app.command()
def summarize(
    run_id: str,
    db_path: str = DEFAULT_DB_PATH,
    cluster_id: int | None = None,
    max_queries: int = 50,
    model: str = "anthropic/claude-3-haiku-20240307",
    use_chroma: bool = False,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    concurrency: int = 4,
    rpm: int | None = None,
    contrast_mode: str = "none",
):
    """Generate LLM summaries for clusters."""
```

### Analysis Commands

```python
@app.command()
def runs(
    db_path: str = DEFAULT_DB_PATH,
    latest: bool = False,
):
    """List all clustering runs."""

@app.command("list-clusters")
def list_clusters(
    run_id: str,
    db_path: str = DEFAULT_DB_PATH,
    limit: int | None = None,
    show_examples: int = 0,
):
    """List clusters from a run."""

@app.command()
def inspect(
    run_id: str,
    cluster_id: int,
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 20,
):
    """Inspect a specific cluster."""

@app.command()
def search(
    query: str,
    db_path: str = DEFAULT_DB_PATH,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    search_type: str = "queries",
    run_id: str | None = None,
    n_results: int = 10,
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """Semantic search for queries or clusters."""

@app.command()
def export(
    run_id: str,
    output: str,
    db_path: str = DEFAULT_DB_PATH,
    format: str = "csv",
    include_queries: bool = False,
):
    """Export cluster data to CSV or JSON."""
```

---

## Rich Console Output

The CLI uses Rich for enhanced terminal output:

### Tables

```python
from rich.table import Table
from rich.console import Console

console = Console()

table = Table(title="Clustering Runs")
table.add_column("Run ID", style="cyan")
table.add_column("Algorithm", style="magenta")
table.add_column("Clusters", justify="right", style="green")

table.add_row("kmeans-100-20251003-123456", "kmeans", "100")
console.print(table)
```

### Progress Bars

```python
from rich.progress import track

for query in track(queries, description="Processing queries..."):
    # Process query
    pass
```

### Panels

```python
from rich.panel import Panel

console.print(Panel(
    "Clustering complete!",
    title="Success",
    border_style="green"
))
```

---

## Logging

The CLI uses a custom logging setup with Rich handlers:

```python
from lmsys_query_analysis.utils.logging import setup_logging

# Enable verbose logging
logger = setup_logging(verbose=True)

logger.info("Starting clustering...")
logger.debug("Processing batch 1/10")
logger.warning("Low memory detected")
logger.error("Failed to connect to database")
```

---

## Error Handling

```python
import typer
from rich.console import Console

console = Console()

try:
    # CLI operation
    result = perform_clustering()
except FileNotFoundError as e:
    console.print(f"[red]Error:[/red] Database not found: {e}")
    raise typer.Exit(code=1)
except Exception as e:
    console.print(f"[red]Unexpected error:[/red] {e}")
    raise typer.Exit(code=1)
```

---

## Extending the CLI

### Adding a New Command

```python
import typer
from rich.console import Console

console = Console()

@app.command()
def my_command(
    db_path: str = DEFAULT_DB_PATH,
    my_option: int = 10,
):
    """
    My custom command description.
    """
    console.print(f"Running my command with option: {my_option}")

    # Your implementation here
    db = DatabaseManager(db_path)
    # ...

    console.print("[green]Success![/green]")
```

### Adding a Command Group

```python
custom_app = typer.Typer(help="Custom commands")

@custom_app.command("subcommand")
def my_subcommand(option: str):
    """Subcommand description."""
    console.print(f"Running subcommand: {option}")

# Register with main app
app.add_typer(custom_app, name="custom")
```

Now you can run: `lmsys custom subcommand --option value`

---

## Testing CLI Commands

```python
from typer.testing import CliRunner
from lmsys_query_analysis.cli.main import app

runner = CliRunner()

def test_load_command():
    result = runner.invoke(app, ["load", "--limit", "100"])
    assert result.exit_code == 0
    assert "Loaded 100 queries" in result.output

def test_cluster_command():
    result = runner.invoke(app, ["cluster", "kmeans", "--n-clusters", "10"])
    assert result.exit_code == 0
    assert "Clustering complete" in result.output
```

---

## Next Steps

- [Database Models](models.md)
- [Clustering API](clustering.md)
- [CLI Command Reference](../cli/overview.md)
