# Examples

This directory contains example scripts demonstrating how to use the LMSYS Query Analysis package.

## example_runner.py

**Complete end-to-end pipeline example**

Demonstrates:
- Loading 1,000 queries from LMSYS-1M
- Running KMeans clustering with 100 clusters
- Configuration management (create, save, load)
- Error handling and user feedback
- Next steps for exploring results

### Prerequisites

```bash
# Install dependencies
uv sync

# Set API keys
export COHERE_API_KEY="your-cohere-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Login to Hugging Face and accept LMSYS-1M terms
huggingface-cli login
```

### Running

```bash
# Run the complete example
uv run python examples/example_runner.py

# Or make it executable and run directly
chmod +x examples/example_runner.py
./examples/example_runner.py
```

### What It Does

1. **Creates Configuration**
   - 1,000 queries
   - 100 clusters
   - Cohere embeddings
   - Anthropic Claude for summaries
   - Hierarchical organization (3 levels)
   - Saves config to `examples/example_config.yaml`

2. **Runs Analysis**
   - Downloads and processes LMSYS data
   - Generates embeddings
   - Performs clustering
   - Creates hierarchical organization
   - Saves to `examples/example_analysis.db`

3. **Demonstrates Search**
   - Searches clusters by topic ("programming languages")
   - Searches clusters by topic ("artificial intelligence")
   - Searches queries by content ("python tutorial")
   - Displays hierarchical structure

4. **Shows Next Steps**
   - Commands to explore results
   - How to generate summaries
   - How to search and export

### Customization

Edit the `RunnerConfig` in `example_runner.py`:

```python
config = RunnerConfig(
    query_limit=5000,        # Process more queries
    n_clusters=200,          # Create more clusters
    enable_hierarchy=True,   # Enable hierarchy creation
    hierarchy_levels=4,      # 4 levels deep
    llm_provider="openai",   # Use OpenAI instead
    llm_model="gpt-4",       # Specify model
)
```

### Expected Output

```
====================================================
                 STEP 1: Configuration
====================================================

✓ Configuration created
  • Queries: 1000
  • Clusters: 100
  • Embedding: cohere/embed-v4.0
  • LLM: anthropic/claude-3-5-sonnet-20241022
  • Database: ./examples/example_analysis.db
  • Hierarchy: Enabled

✓ Configuration saved to ./examples/example_config.yaml

====================================================
              STEP 2: Running Analysis Pipeline
====================================================

⚠  This will take several minutes...
⚠  Press Ctrl+C to cancel

[... data loading ...]
[... clustering ...]
[... hierarchy creation ...]

====================================================
                    STEP 3: Results
====================================================

                   Analysis Results
┌─────────────────────────┬──────────────────────────────┐
│ Metric                  │ Value                        │
├─────────────────────────┼──────────────────────────────┤
│ Run ID                  │ kmeans-100-20251008-120000   │
│ Hierarchy Run ID        │ hier-kmeans-100-...          │
│ Total Queries Processed │ 1000                         │
│ Execution Time          │ 456.78s                      │
│ Database Path           │ ./examples/example_analysis… │
└─────────────────────────┴──────────────────────────────┘

====================================================
              STEP 4: Search Demonstrations
====================================================

Search 1: Clusters about 'programming languages'
     Top 3 Programming Clusters
┌──────┬────────────┬────────────────────┬────────┐
│ Rank │ Cluster ID │ Title              │ Score  │
├──────┼────────────┼────────────────────┼────────┤
│ 1    │ 42         │ Python Programming │ 0.856  │
│ 2    │ 15         │ JavaScript Web Dev │ 0.823  │
│ 3    │ 67         │ Code Debugging     │ 0.798  │
└──────┴────────────┴────────────────────┴────────┘

Search 2: Clusters about 'artificial intelligence'
     Top 3 AI/ML Clusters
┌──────┬────────────┬────────────────────┬────────┐
│ Rank │ Cluster ID │ Title              │ Score  │
├──────┼────────────┼────────────────────┼────────┤
│ 1    │ 23         │ Machine Learning   │ 0.912  │
│ 2    │ 8          │ Neural Networks    │ 0.887  │
│ 3    │ 55         │ Data Science       │ 0.845  │
└──────┴────────────┴────────────────────┴────────┘

Search 3: Queries about 'python tutorial'
     Top 5 Python Queries
┌──────┬────────────────────────────────┬────────┐
│ Rank │ Query                          │ Score  │
├──────┼────────────────────────────────┼────────┤
│ 1    │ How do I learn Python fast?    │ 0.923  │
│ 2    │ Best Python tutorial for be... │ 0.901  │
│ 3    │ Python basics course recommend │ 0.878  │
└──────┴────────────────────────────────┴────────┘

Hierarchy Structure:
   Cluster Hierarchy (hier-kmeans-100-...)
┌───────┬────────┬────────────────────────────────┐
│ Level │ Count  │ Sample Titles                  │
├───────┼────────┼────────────────────────────────┤
│ 0     │ 100    │ Python Programming, Web De...  │
│ 1     │ 30     │ Programming Languages, Dat...  │
│ 2     │ 10     │ Technical Topics, Creative...  │
└───────┴────────┴────────────────────────────────┘

====================================================
              STEP 5: Explore Your Results
====================================================

[Next steps with actual commands...]
```

### Duration

- **1,000 queries + 100 clusters + hierarchy**: ~15-25 minutes total
  - Data loading: ~2-3 minutes
  - Clustering: ~1-2 minutes
  - Hierarchy creation: ~10-20 minutes (depends on LLM API speed)
  - Search demonstrations: ~10 seconds

### Output Files

```
examples/
├── example_runner.py         # This script
├── example_config.yaml       # Saved configuration
└── example_analysis.db       # SQLite database with results
```

## Tips

1. **Start Small**: Run with 100 queries first to test your setup
2. **Enable Hierarchy Later**: Hierarchy creation is time-consuming, do it after verifying clustering works
3. **Monitor Resources**: Large datasets require significant memory and API quota
4. **Save Your Config**: YAML configs make it easy to reproduce analyses

## Troubleshooting

### "Already running asyncio in this thread"
The runner has an asyncio nesting issue. Use the CLI workflow instead:

```bash
uv run lmsys load --limit 1000 --use-chroma --db-path examples/example.db
uv run lmsys cluster kmeans --n-clusters 100 --use-chroma --db-path examples/example.db
```

### "Missing API keys"
Set required environment variables:

```bash
export COHERE_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### "Access denied to LMSYS-1M"
Login to Hugging Face and accept dataset terms:

```bash
huggingface-cli login
# Then visit: https://huggingface.co/datasets/lmsys/lmsys-chat-1m
```

## More Examples

For more usage patterns, see:
- [RUNNER_README.md](../RUNNER_README.md) - Complete API reference
- [README.md](../README.md) - CLI documentation
- [CLAUDE.md](../CLAUDE.md) - Architecture overview
