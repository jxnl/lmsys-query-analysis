# Data Plan: Multi-Source Data Ingestion

**Status**: Draft  
**Created**: October 15, 2025  
**Author**: Vignesh

## Overview

Refactor the data loading pipeline to support multiple data sources (HuggingFace datasets, CSV files, etc.) through an adapter pattern. This plan focuses on **backward compatibility first** - we'll extract the adapter infrastructure without breaking existing functionality, then add new features incrementally.

## Goals

✅ Extract adapter pattern while maintaining backward compatibility  
✅ Keep all existing tests passing  
✅ Enable future support for multiple data sources (HuggingFace, CSV)  
✅ Test incrementally at each phase

## Non-Goals

❌ Multi-dataset persistence in single database  
❌ Dynamic schema mapping  
❌ Source conflict detection (assume users clear DB between sources)

---

## Phase 1: Create Adapter Infrastructure

**Goal**: Create the adapter interface, test it with mocks, then implement the HuggingFace adapter.

### Phase 1a: Create Protocol and Mock Tests

**Goal**: Define the adapter interface and test it with mock implementations before building the real adapter.

#### Files to Create

- `src/lmsys_query_analysis/db/adapters.py`
  - `DataSourceAdapter` Protocol (defines interface)
  - Helper functions (`extract_first_query` moved here)

- `tests/unit/db/test_adapters.py`
  - Mock adapter implementation for testing
  - Tests for the protocol interface
  - Tests for `extract_first_query` helper

#### Files to Read/Reference

- `src/lmsys_query_analysis/db/loader.py` (lines 39-56: `extract_first_query`)
- `tests/unit/db/test_loader.py` (lines 11-70: existing `extract_first_query` tests)

#### What to Build

```python
# adapters.py structure (Phase 1a):

from typing import Protocol, Iterator, runtime_checkable

@runtime_checkable
class DataSourceAdapter(Protocol):
    """Protocol defining interface for data source adapters.
    
    All adapters must yield normalized records with this schema:
    {
        "conversation_id": str,
        "query_text": str,
        "model": str,
        "language": str | None,
        "timestamp": datetime | None,
        "extra_metadata": dict,
    }
    """
    def __iter__(self) -> Iterator[dict]:
        """Yield normalized records with standard schema."""
        ...
    
    def __len__(self) -> int | None:
        """Return total count if known, None for streaming sources."""
        ...


def extract_first_query(conversation: list[dict] | None) -> str | None:
    """Extract the first user query from a conversation.
    
    Args:
        conversation: List of conversation turns in OpenAI format
    
    Returns:
        The first user message content, or None if not found
    """
    # Move implementation from loader.py
    ...
```

#### Tests to Write (Phase 1a)

- `tests/unit/db/test_adapters.py`
  - `MockDataSourceAdapter` class for testing
  - `test_extract_first_query_*` (move from `test_loader.py`)
  - `test_adapter_protocol_interface`
  - `test_mock_adapter_iteration`
  - `test_mock_adapter_length`
  - `test_adapter_normalized_output_schema`

#### Success Criteria (Phase 1a)

- [x] `adapters.py` created with `DataSourceAdapter` protocol
- [x] `extract_first_query` moved from `loader.py` to `adapters.py`
- [x] `test_adapters.py` created with mock adapter
- [x] Protocol tests pass with mock implementation
- [x] Unit tests pass: `uv run pytest tests/unit/db/test_adapters.py -v`

---

### Phase 1b: Implement HuggingFace Adapter

**Goal**: Implement the real HuggingFace adapter using the tested protocol interface.

#### Files to Modify

- `src/lmsys_query_analysis/db/adapters.py`
  - Add `HuggingFaceAdapter` class (implements `DataSourceAdapter`)
  - Extract HF-specific logic from `loader.py`

#### Files to Read/Reference

- `src/lmsys_query_analysis/db/loader.py` (lines 92-103: HF dataset loading logic)
- `src/lmsys_query_analysis/db/loader.py` (lines 164-223: batch iteration and data extraction)

#### What to Build

```python
# Add to adapters.py:

class HuggingFaceAdapter:
    """Adapter for HuggingFace datasets (lmsys/lmsys-chat-1m)."""
    
    def __init__(
        self,
        dataset_name: str = "lmsys/lmsys-chat-1m",
        split: str = "train",
        limit: int | None = None,
        use_streaming: bool = False,
    ):
        # Load dataset using datasets.load_dataset()
        # Apply limit if specified
        # Store dataset and configuration
        ...
    
    def __iter__(self) -> Iterator[dict]:
        # Iterate over HF dataset
        # Parse conversation field (handle JSON strings)
        # Extract first query using extract_first_query()
        # Build extra_metadata from HF fields
        # Yield normalized dicts matching protocol schema
        ...
    
    def __len__(self) -> int | None:
        # Return dataset length or None if streaming
        ...
```

#### Tests to Write (Phase 1b)

- Add to `tests/unit/db/test_adapters.py`
  - `test_hf_adapter_initialization`
  - `test_hf_adapter_iteration_basic`
  - `test_hf_adapter_with_limit`
  - `test_hf_adapter_normalized_output_schema`
  - `test_hf_adapter_handles_json_conversations`
  - `test_hf_adapter_handles_missing_fields`
  - `test_hf_adapter_streaming_mode`
  - `test_hf_adapter_conforms_to_protocol`

#### Success Criteria (Phase 1b)

- [x] `HuggingFaceAdapter` class extracts all HF-specific logic
- [x] Adapter conforms to `DataSourceAdapter` protocol
- [x] All HF adapter tests pass
- [x] Unit tests pass: `uv run pytest tests/unit/db/test_adapters.py -v`
- [x] Adapter can be instantiated and iterated independently
- [x] Mock HF dataset used in tests (no real HF downloads)

---

## Phase 2: Refactor Loader to Use Adapter

**Goal**: Update `load_lmsys_dataset()` to use the adapter internally while maintaining the exact same function signature and behavior.

### Files to Modify

- `src/lmsys_query_analysis/db/loader.py`
  - Refactor `load_lmsys_dataset()` to:
    - Create `HuggingFaceAdapter` instance internally
    - Replace direct HF dataset iteration with adapter iteration
    - Keep all batching, deduplication, and Chroma logic unchanged

### Key Changes

```python
# loader.py before:
dataset = load_dataset("lmsys/lmsys-chat-1m", split="train")
for batch in chunk_iter(dataset, batch_size):
    for row in batch:
        conversation_id = row.get("conversation_id")
        query_text = extract_first_query(row.get("conversation"))
        ...

# loader.py after:
from .adapters import HuggingFaceAdapter

adapter = HuggingFaceAdapter(
    dataset_name="lmsys/lmsys-chat-1m",
    split="train",
    limit=limit,
    use_streaming=use_streaming,
)

for batch in chunk_iter(adapter, batch_size):
    for normalized_record in batch:
        # normalized_record already has conversation_id, query_text, etc.
        conversation_id = normalized_record["conversation_id"]
        query_text = normalized_record["query_text"]
        ...
```

### Imports to Update

- Import `HuggingFaceAdapter` from `.adapters`
- Import `extract_first_query` from `.adapters` (for backward compat if needed)
- Remove direct `load_dataset` call from main function body

### Tests to Update

- `tests/unit/db/test_loader.py`
  - Update mocks to patch `HuggingFaceAdapter` instead of `load_dataset`
  - All existing tests should pass without logic changes
  - Tests:
    - `test_load_lmsys_dataset_basic`
    - `test_load_lmsys_dataset_skip_existing`
    - `test_load_lmsys_dataset_handles_errors`
    - `test_load_lmsys_dataset_with_limit`
    - `test_load_lmsys_dataset_deduplicates_within_batch`
    - `test_load_lmsys_dataset_handles_json_conversation`
    - `test_load_lmsys_dataset_stores_metadata`
    - `test_load_lmsys_dataset_with_chroma`
    - `test_load_lmsys_dataset_large_batch`
    - `test_load_lmsys_dataset_missing_language`

### Success Criteria

- [x] `load_lmsys_dataset()` uses `HuggingFaceAdapter` internally
- [x] Function signature unchanged (fully backward compatible)
- [x] All batching, dedup, Chroma logic unchanged
- [x] All existing tests pass: `uv run pytest tests/unit/db/test_loader.py -v`
- [x] No functional changes observable from outside

---

## Phase 3: CLI Verification & Integration Tests

**Goal**: Verify the refactored code works end-to-end through the CLI and add integration tests.

### Files to Test

- `src/lmsys_query_analysis/cli/commands/data.py` (no changes needed)
  - Should work unchanged since `load_lmsys_dataset()` signature is the same

### Tests to Run

#### Existing Unit Tests
```bash
# All unit tests should pass
uv run pytest tests/unit/db/ -v
```

#### Existing Integration Tests
```bash
# CLI integration tests
uv run pytest tests/integration/test_cli.py -v
```

#### Manual CLI Testing
```bash
# Test with small limit (uses temp DB)
uv run lmsys load --limit 10 --db-path /tmp/test-adapter.db

# Verify data loaded
uv run lmsys list --db-path /tmp/test-adapter.db --limit 5

# Test with Chroma (small limit)
uv run lmsys load --limit 50 --use-chroma --db-path /tmp/test-adapter-chroma.db --chroma-path /tmp/test-chroma

# Clean up
uv run lmsys clear --db-path /tmp/test-adapter.db --chroma-path /tmp/test-chroma --yes
```

### New Integration Test

Add to `tests/integration/test_cli.py`:

```python
def test_load_command_with_adapter(tmp_path):
    """Test that load command works with adapter refactor."""
    db_path = tmp_path / "adapter-test.db"
    
    # Mock the HuggingFaceAdapter to avoid actual HF download
    with patch('lmsys_query_analysis.db.adapters.HuggingFaceAdapter') as mock_adapter:
        mock_data = [
            {
                "conversation_id": "test1",
                "query_text": "What is Python?",
                "model": "gpt-4",
                "language": "en",
                "timestamp": None,
                "extra_metadata": {},
            }
        ]
        mock_adapter.return_value.__iter__ = Mock(return_value=iter(mock_data))
        mock_adapter.return_value.__len__ = Mock(return_value=1)
        
        result = runner.invoke(
            app,
            ["load", "--limit", "10", "--db-path", str(db_path)]
        )
        
        assert result.exit_code == 0
        assert "loaded" in result.stdout.lower()
```

### Success Criteria

- [x] All unit tests pass: `uv run pytest tests/unit/db/ -v`
- [x] All integration tests pass: `uv run pytest tests/integration/test_cli.py -v`
- [x] Manual CLI test loads data successfully
- [x] Manual CLI test with Chroma works
- [x] New integration test added and passing
- [x] No errors or warnings in CLI output

---

## Phase 4: Documentation & Code Cleanup

**Goal**: Clean up imports, update docstrings, and document the new architecture.

### Files to Update

- `src/lmsys_query_analysis/db/adapters.py`
  - Add comprehensive module docstring
  - Document all classes and methods
  - Add usage examples in docstrings

- `src/lmsys_query_analysis/db/loader.py`
  - Update docstring to mention adapter usage
  - Clean up any unused imports
  - Add comment explaining adapter pattern

- `src/lmsys_query_analysis/db/__init__.py`
  - Export `HuggingFaceAdapter` if useful for external usage
  - Export `DataSourceAdapter` protocol

### Documentation to Add

Add to `adapters.py`:

```python
"""Data source adapters for multi-source ingestion.

This module provides adapters that normalize different data sources
(HuggingFace datasets, CSV files, etc.) into a common format for ingestion.

The adapter pattern allows the loader to remain source-agnostic while
supporting multiple input formats.

Example usage:
    >>> from lmsys_query_analysis.db.adapters import HuggingFaceAdapter
    >>> adapter = HuggingFaceAdapter(limit=100)
    >>> for record in adapter:
    ...     print(record["query_text"])

Architecture:
    - DataSourceAdapter: Protocol defining the adapter interface
    - HuggingFaceAdapter: Adapter for HuggingFace datasets
    - CSVAdapter: (Future) Adapter for CSV files
"""
```

### Success Criteria

- [ ] All docstrings complete and accurate
- [ ] Module-level documentation explains architecture
- [ ] Code is clean with no unused imports
- [ ] Type hints complete and correct
- [ ] Comments explain non-obvious design decisions

---

## Testing Strategy Summary

### Unit Tests (`tests/unit/db/`)
- Fast, no external dependencies
- Use in-memory SQLite (`temp_db` fixture)
- Mock external data sources
- Files:
  - `test_adapters.py` (new) - Adapter logic
  - `test_loader.py` (updated) - Loader logic with adapters

### Integration Tests (`tests/integration/`)
- Test services working together
- Use temporary directories and databases
- Still mock external APIs to avoid costs
- Files:
  - `test_cli.py` (updated) - CLI with adapter

### Manual Testing
- Small limit tests with real HF dataset
- Verify CLI output and behavior
- Test with and without Chroma

---

## Phase 5: Rename Function and Add --hf Flag

**Goal**: Rename `load_lmsys_dataset` to `load_dataset` (since it's now generic) and add a `--hf` flag to the CLI to allow users to specify which HuggingFace dataset to load. Maintain backward compatibility via function alias.

### Rationale

Since the function now supports any HuggingFace dataset (not just LMSYS), the name should reflect its generic purpose. Clean rename with no deprecated alias.

### Files to Modify

- `src/lmsys_query_analysis/db/loader.py`
  - Rename `load_lmsys_dataset` to `load_dataset`
  - Add optional `dataset_name` parameter with default `"lmsys/lmsys-chat-1m"`
  - Pass `dataset_name` to `HuggingFaceAdapter`

- `src/lmsys_query_analysis/cli/commands/data.py`
  - Update import to use `load_dataset` (new name)
  - Add `--hf` option to `load` command
  - Default to `lmsys/lmsys-chat-1m` if not specified
  - Pass dataset name to `load_dataset()`

- `src/lmsys_query_analysis/runner.py`
  - Update import to use `load_dataset` (preferred, but old name still works)

- All test files referencing `load_lmsys_dataset`
  - Update to use new function name `load_dataset`

### What to Build

```python
# loader.py changes:

def load_dataset(
    db: Database,
    limit: int | None = None,
    skip_existing: bool = True,
    chroma: Optional[ChromaManager] = None,
    embedding_model: str = "embed-v4.0",
    embedding_provider: str = "cohere",
    batch_size: int = 5000,
    use_streaming: bool = False,
    apply_pragmas: bool = True,
    dataset_name: str = "lmsys/lmsys-chat-1m",  # NEW parameter with default
) -> dict:
    """Load dataset from HuggingFace (or other sources in future).
    
    Args:
        db: Database instance
        limit: Maximum records to load
        skip_existing: Skip conversations that already exist in DB
        chroma: Optional ChromaDB manager for vector storage
        embedding_model: Model for generating embeddings
        embedding_provider: Provider for embeddings
        batch_size: Number of records per batch for DB inserts
        use_streaming: Use streaming dataset iteration
        apply_pragmas: Apply SQLite PRAGMA speedups during load
        dataset_name: HuggingFace dataset identifier (default: lmsys/lmsys-chat-1m)
        
    Returns:
        Dictionary with loading statistics
    """
    # ... existing implementation, but pass dataset_name to adapter ...
    adapter = HuggingFaceAdapter(
        dataset_name=dataset_name,  # Use the parameter
        split="train",
        limit=limit,
        use_streaming=use_streaming,
    )
    # ... rest of function unchanged ...


```

```python
# data.py changes:

from ...db.loader import load_dataset  # Updated import

@with_error_handling
def load(
    limit: int = typer.Option(None, help="Limit number of records to load"),
    db_path: str = db_path_option,
    use_chroma: bool = typer.Option(False, help="Enable ChromaDB for semantic search"),
    hf: str = typer.Option(
        "lmsys/lmsys-chat-1m",
        "--hf",
        help="HuggingFace dataset to load"
    ),  # NEW option
    chroma_path: str = chroma_path_option,
    embedding_model: str = embedding_model_option,
    # ... other options
):
    """Download and load HuggingFace dataset into SQLite."""
    # ... setup code ...
    
    stats = load_dataset(  # Updated function name
        db,
        limit=limit,
        skip_existing=not force_reload,
        chroma=chroma,
        embedding_model=model,
        embedding_provider=provider,
        batch_size=db_batch_size,
        use_streaming=streaming,
        apply_pragmas=not no_pragmas,
        dataset_name=hf,  # Pass the dataset name
    )
    # ... rest unchanged ...
```

### Tests to Write

Add to `tests/unit/cli/test_data_cli.py` (or create if needed):

```python
def test_load_with_custom_hf_dataset(tmp_path):
    """Test load command with custom HuggingFace dataset."""
    db_path = tmp_path / "custom-hf.db"
    
    with patch('lmsys_query_analysis.db.loader.HuggingFaceAdapter') as mock_adapter:
        mock_adapter.return_value.__iter__ = Mock(return_value=iter([]))
        mock_adapter.return_value.__len__ = Mock(return_value=0)
        
        result = runner.invoke(
            app,
            ["load", "--hf", "custom/dataset", "--limit", "10", "--db-path", str(db_path)]
        )
        
        # Verify HuggingFaceAdapter was called with custom dataset
        mock_adapter.assert_called_once()
        call_kwargs = mock_adapter.call_args[1]
        assert call_kwargs["dataset_name"] == "custom/dataset"

def test_load_defaults_to_lmsys_dataset(tmp_path):
    """Test load command defaults to lmsys/lmsys-chat-1m."""
    db_path = tmp_path / "default-hf.db"
    
    with patch('lmsys_query_analysis.db.loader.HuggingFaceAdapter') as mock_adapter:
        mock_adapter.return_value.__iter__ = Mock(return_value=iter([]))
        mock_adapter.return_value.__len__ = Mock(return_value=0)
        
        result = runner.invoke(
            app,
            ["load", "--limit", "10", "--db-path", str(db_path)]
        )
        
        # Verify default dataset was used
        mock_adapter.assert_called_once()
        call_kwargs = mock_adapter.call_args[1]
        assert call_kwargs["dataset_name"] == "lmsys/lmsys-chat-1m"
```

Add to `tests/unit/db/test_loader.py`:

```python
def test_load_with_custom_dataset_name(temp_db):
    """Test loader accepts custom dataset name."""
    with patch('lmsys_query_analysis.db.adapters.HuggingFaceAdapter') as mock_adapter:
        mock_adapter.return_value.__iter__ = Mock(return_value=iter([]))
        mock_adapter.return_value.__len__ = Mock(return_value=0)
        
        session = Session(temp_db)
        load_lmsys_dataset(
            session,
            dataset_name="custom/dataset",
            limit=10
        )
        
        # Verify adapter was initialized with custom dataset
        mock_adapter.assert_called_once()
        assert mock_adapter.call_args[1]["dataset_name"] == "custom/dataset"
```

### Manual CLI Testing

```bash
# Test with default dataset (should work as before)
uv run lmsys load --limit 10 --db-path /tmp/test-default.db

# Test with custom dataset (using another LMSYS dataset)
uv run lmsys load --hf lmsys/chatbot-arena-conversations --limit 10 --db-path /tmp/test-custom.db

# Test help text shows the new flag
uv run lmsys load --help | grep -A2 "\-\-hf"

# Clean up
rm /tmp/test-default.db /tmp/test-custom.db
```

### Success Criteria

- [x] Function renamed from `load_lmsys_dataset` to `load_dataset` in `loader.py`
- [x] `dataset_name` parameter added to `load_dataset()` with default value
- [x] Dataset name passed to `HuggingFaceAdapter` correctly
- [x] CLI imports updated to use `load_dataset` (new name)
- [ ] `--hf` flag added to CLI load command
- [ ] Default value for `--hf` is `lmsys/lmsys-chat-1m` (backward compatible)
- [ ] Runner imports updated to use `load_dataset` (new name)
- [ ] All test files updated to use new function name
- [ ] All existing tests still pass
- [ ] New unit tests for custom dataset name pass
- [ ] New CLI tests for `--hf` flag pass
- [ ] Help text displays the new option clearly
- [ ] Manual testing with different datasets works

---

## Future Phases (Not in This Plan)

After Phase 5 is complete, future work includes:

### Phase 6: CSV Adapter
- Create `CSVAdapter` class in `adapters.py`
- Define required CSV schema
- Add `--csv <path>` flag to CLI

### Phase 7: Flag Mutual Exclusivity
- Add validation: exactly one of `--hf` or `--csv` required
- Add database empty check/warning
- Error messages for invalid combinations

---

## Risk Mitigation

### Risk: Breaking Existing Functionality
**Mitigation**: 
- Maintain exact same function signatures
- Run full test suite after each phase
- Test manually with CLI before merging

### Risk: Performance Regression
**Mitigation**:
- Adapters should be zero-copy where possible
- Keep batching and deduplication logic unchanged
- Profile if needed: `uv run python -m cProfile ...`

### Risk: Test Complexity
**Mitigation**:
- Use fixtures from `conftest.py` (`temp_db`, `sample_queries`)
- Mock adapters in unit tests, not full HF dataset
- Keep integration tests simple and focused

---

## Timeline Estimate

- **Phase 1** (Create Adapters): 2-3 hours
- **Phase 2** (Refactor Loader): 2-3 hours
- **Phase 3** (CLI Testing): 1-2 hours
- **Phase 4** (Documentation): 1 hour
- **Phase 5** (Add --hf Flag): 1-2 hours

**Total**: 7-11 hours for complete backward-compatible refactor with HF dataset selection

---

## Approval Checklist

Before starting implementation:

- [ ] Plan reviewed and approved
- [ ] Test strategy understood
- [ ] File structure makes sense
- [ ] Success criteria clear for each phase
- [ ] Ready to implement incrementally

---

## References

### Key Files
- Current loader: `src/lmsys_query_analysis/db/loader.py`
- Current tests: `tests/unit/db/test_loader.py`
- CLI command: `src/lmsys_query_analysis/cli/commands/data.py`
- Test fixtures: `tests/conftest.py`

### Testing Docs
- Test README: `tests/README.md`
- Test organization: Unit → Integration → Smoke
- Fixtures: `temp_db`, `db_session`, `sample_queries`

