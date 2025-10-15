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

**Goal**: Create the adapter interface and extract HuggingFace-specific logic into a dedicated adapter class.

### Files to Create

- `src/lmsys_query_analysis/db/adapters.py`
  - `DataSourceAdapter` Protocol (defines interface)
  - `HuggingFaceAdapter` class (hardcoded to `lmsys/lmsys-chat-1m`)
  - Helper functions (`extract_first_query` moved here)

### Files to Read/Reference

- `src/lmsys_query_analysis/db/loader.py` (lines 39-56: `extract_first_query`)
- `src/lmsys_query_analysis/db/loader.py` (lines 92-103: HF dataset loading logic)
- `src/lmsys_query_analysis/db/loader.py` (lines 164-223: batch iteration and data extraction)

### What to Build

```python
# adapters.py structure:

class DataSourceAdapter(Protocol):
    """Protocol defining interface for data source adapters."""
    def __iter__(self) -> Iterator[dict]:
        """Yield normalized records with standard schema."""
        ...
    
    def __len__(self) -> int | None:
        """Return total count if known, None for streaming sources."""
        ...


class HuggingFaceAdapter:
    """Adapter for HuggingFace datasets (lmsys/lmsys-chat-1m)."""
    
    def __init__(
        self,
        dataset_name: str = "lmsys/lmsys-chat-1m",
        split: str = "train",
        limit: int | None = None,
        use_streaming: bool = False,
    ):
        # Load dataset
        # Store limit for iteration
        ...
    
    def __iter__(self) -> Iterator[dict]:
        # Yield normalized dicts:
        # {
        #   "conversation_id": str,
        #   "query_text": str,
        #   "model": str,
        #   "language": str | None,
        #   "timestamp": datetime | None,
        #   "extra_metadata": dict,
        # }
        ...
    
    def __len__(self) -> int | None:
        # Return dataset length or None if streaming
        ...
```

### Tests to Write

- `tests/unit/db/test_adapters.py`
  - `test_extract_first_query_*` (move from `test_loader.py`)
  - `test_hf_adapter_initialization`
  - `test_hf_adapter_iteration_basic`
  - `test_hf_adapter_with_limit`
  - `test_hf_adapter_normalized_output_schema`
  - `test_hf_adapter_handles_json_conversations`
  - `test_hf_adapter_handles_missing_fields`
  - `test_hf_adapter_streaming_mode`

### Success Criteria

- [ ] `adapters.py` created with `DataSourceAdapter` protocol
- [ ] `HuggingFaceAdapter` class extracts all HF-specific logic
- [ ] `extract_first_query` moved to `adapters.py`
- [ ] Unit tests pass: `uv run pytest tests/unit/db/test_adapters.py -v`
- [ ] Adapter can be instantiated and iterated independently

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

- [ ] `load_lmsys_dataset()` uses `HuggingFaceAdapter` internally
- [ ] Function signature unchanged (fully backward compatible)
- [ ] All batching, dedup, Chroma logic unchanged
- [ ] All existing tests pass: `uv run pytest tests/unit/db/test_loader.py -v`
- [ ] No functional changes observable from outside

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

- [ ] All unit tests pass: `uv run pytest tests/unit/db/ -v`
- [ ] All integration tests pass: `uv run pytest tests/integration/test_cli.py -v`
- [ ] Manual CLI test loads data successfully
- [ ] Manual CLI test with Chroma works
- [ ] New integration test added and passing
- [ ] No errors or warnings in CLI output

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

## Future Phases (Not in This Plan)

After this refactor is complete and tested, future work includes:

### Phase 5: Add --hf Flag (Optional Dataset Selection)
- Add `--hf <dataset_name>` flag to CLI
- Default to `lmsys/lmsys-chat-1m` if not specified
- Pass dataset name to `HuggingFaceAdapter`

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

**Total**: 6-9 hours for complete backward-compatible refactor

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

