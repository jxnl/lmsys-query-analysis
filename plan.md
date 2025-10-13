# Dataset-Aware Ingestion (Phase 1)

## Overview

Two-phase implementation with **tests written alongside code** at each step:

- **Phase 1a**: Refactor existing code to source pattern (HuggingFace only, no behavior change)
- **Phase 1b**: Add CSV support with simple text format

## Key Principles

1. **Test-Driven**: Write tests immediately after (or before) implementation
2. **No Wrappers**: Remove `load_lmsys_dataset()` entirely - all callers use new `load_from_source()` directly
3. **Run Tests Frequently**: After each section, run relevant tests to catch regressions early
4. **Incremental**: Each section should leave the codebase in a working state

---

## Phase 1a: Refactor to Source Pattern

### Section 1a.1: Create Source Infrastructure

**Objective**: Create abstract base class and HuggingFace source implementation

#### Files to Create

- `src/lmsys_query_analysis/db/sources.py` (~150 lines initially)

#### Files to Reference

- `src/lmsys_query_analysis/db/models.py` (lines 10-26) - `Query` model schema
- `src/lmsys_query_analysis/db/loader.py` (lines 39-55) - `extract_first_query()`
- `src/lmsys_query_analysis/db/loader.py` (lines 92-103) - HF dataset loading code to move

#### Implementation Checklist

- [ ] Create `BaseSource` abstract class:
  - [ ] `validate_source()` - verify source accessible
  - [ ] `iter_records()` - yield normalized dicts
  - [ ] `get_source_label()` - return string like "hf:dataset"
- [ ] Implement `HuggingFaceSource`:
  - [ ] Accept `dataset_id` (e.g., "lmsys/lmsys-chat-1m")
  - [ ] Accept `limit`, `streaming` parameters
  - [ ] Import `extract_first_query()` from loader.py
  - [ ] Wrap `datasets.load_dataset()` call
  - [ ] Parse conversation field using `extract_first_query()`
  - [ ] Return label like `"hf:lmsys/lmsys-chat-1m"`
- [ ] Define normalized record format:
  ```python
  {
      "conversation_id": str,
      "query_text": str,
      "model": str,
      "language": Optional[str],
      "timestamp": Optional[datetime],
      "extra_metadata": Optional[dict],
  }
  ```


---

### Section 1a.2: Test HuggingFaceSource

**Objective**: Write comprehensive tests for HuggingFaceSource before refactoring loader

⚠️ **IMPORTANT**: Complete this section before moving to 1a.3. Tests must pass before refactoring.

#### Files to Create

- `tests/unit/db/test_sources.py` (start with ~100 lines for HF tests)

#### Files to Reference

- `tests/conftest.py` - fixtures
- `tests/unit/db/test_loader.py` (lines 135-177) - mock dataset patterns

#### Testing Checklist

- [ ] Test `HuggingFaceSource`:
  - [ ] Mock `load_dataset()`, verify params passed correctly
  - [ ] Test streaming mode → `streaming=True` passed
  - [ ] Test non-streaming mode → limit applied via `.select()`
  - [ ] Test `extract_first_query()` called on conversation field
  - [ ] Test `get_source_label()` returns `"hf:dataset_id"`
  - [ ] Test record format matches normalized schema

---

### Section 1a.3: Refactor Loader

**Objective**: Replace `load_lmsys_dataset()` with generic `load_from_source()`

⚠️ **CRITICAL**: This is a major refactor. Do NOT create a wrapper - delete `load_lmsys_dataset()` entirely.

#### Files to Modify

- `src/lmsys_query_analysis/db/loader.py` (major refactor - net reduction ~50 lines)

#### Current Code to Extract

From `load_lmsys_dataset()` (lines 58-351):

- ✂️ **DELETE**: HF dataset loading (lines 92-103) → already in HuggingFaceSource
- ✂️ **DELETE**: Function definition (lines 58-68) → no wrapper needed
- ✅ **KEEP**: PRAGMA setup (lines 120-126) → move into `load_from_source`
- ✅ **KEEP**: Batch loop (lines 165-277) → adapt to use `source.iter_records()`
- ✅ **KEEP**: ChromaDB sync (lines 282-349) → move into `load_from_source`
- ✅ **KEEP**: `extract_first_query()` (lines 39-55) → still used by HuggingFaceSource

#### Implementation Checklist

- [ ] Create `load_queries(db, source, chroma, embedding_model, embedding_provider, batch_size, skip_existing, apply_pragmas)`:
  - [ ] Accept any BaseSource instance
  - [ ] Get source label via `source.get_source_label()` for stats
  - [ ] Apply PRAGMA optimizations if `apply_pragmas=True`
  - [ ] Iterate records from `source.iter_records()` (yields normalized dicts)
  - [ ] Batch processing with chunking (reuse existing logic)
  - [ ] Dedupe on conversation_id (existing logic)
  - [ ] ChromaDB embedding sync (existing logic)
  - [ ] Return stats: `{"source": str, "total_processed": int, "loaded": int, "skipped": int, "errors": int}`
- [ ] **DELETE `load_lmsys_dataset()` function entirely** (lines 58-351)
- [ ] Keep `extract_first_query()` helper (used by HuggingFaceSource)
- [ ] Update module imports if needed

---

### Section 1a.4: Test Loader Refactoring

**Objective**: Update all existing loader tests + add new tests for `load_queries()`

⚠️ **IMPORTANT**: Many existing tests will break because `load_lmsys_dataset()` is deleted.

#### Files to Modify

- `tests/unit/db/test_loader.py` (rewrite ~200 lines)

#### Files to Reference

- `tests/conftest.py` - DB fixtures (temp_db fixture)
- `tests/unit/db/test_loader.py` (lines 135-485) - all tests that call `load_lmsys_dataset()`

#### Testing Checklist

- [ ] **Rewrite existing tests** to use `load_queries()` + mock sources:
  - [ ] `test_load_lmsys_dataset_basic` → `test_load_queries_basic`
  - [ ] `test_load_lmsys_dataset_skip_existing` → `test_load_queries_skip_existing`
  - [ ] `test_load_lmsys_dataset_handles_errors` → `test_load_queries_handles_errors`
  - [ ] `test_load_lmsys_dataset_with_limit` → `test_load_queries_with_limit`
  - [ ] `test_load_lmsys_dataset_deduplicates_within_batch` → `test_load_queries_deduplicates_within_batch`
  - [ ] `test_load_lmsys_dataset_handles_json_conversation` → (delete, now in source tests)
  - [ ] `test_load_lmsys_dataset_stores_metadata` → `test_load_queries_stores_metadata`
  - [ ] `test_load_lmsys_dataset_with_chroma` → `test_load_queries_with_chroma`
  - [ ] `test_load_lmsys_dataset_large_batch` → `test_load_queries_large_batch`
  - [ ] `test_load_lmsys_dataset_missing_language` → `test_load_queries_missing_language`
- [ ] **Add new tests**:
  - [ ] `test_load_queries_with_mock_base_source` - use pure mock BaseSource
  - [ ] `test_load_queries_stats_include_source_label` - verify source label in stats
  - [ ] `test_load_queries_pragmas_applied` - verify PRAGMA statements executed
- [ ] Run `uv run pytest tests/unit/db/test_loader.py -v` → all pass
- [ ] Run full test suite → verify no regressions in other modules

### Section 1a.5: Update Runner

**Objective**: Update runner.py to use new source pattern

#### Files to Modify

- `src/lmsys_query_analysis/runner.py` (lines 171-205 - `load_data()` function)

#### Files to Reference

- `src/lmsys_query_analysis/db/sources.py` - import HuggingFaceSource
- `src/lmsys_query_analysis/db/loader.py` - import load_queries

#### Implementation Checklist

- [ ] Update imports:
  - [ ] Add `from .db.sources import HuggingFaceSource`
  - [ ] Change `from .db.loader import load_lmsys_dataset` → `from .db.loader import load_queries`
- [ ] Refactor `load_data()` function (lines 171-205):
  - [ ] Create `HuggingFaceSource` instance with dataset_id="lmsys/lmsys-chat-1m"
  - [ ] Pass `limit=config.query_limit` and `streaming=config.use_streaming` to source
  - [ ] Call `load_queries(db, source, chroma, ...)` instead of `load_lmsys_dataset()`
  - [ ] Keep all existing parameters and logging

---

### Section 1a.6: Update CLI Data Command

**Objective**: Update CLI load command to use new source pattern

#### Files to Modify

- `src/lmsys_query_analysis/cli/commands/data.py` (lines 1-69 - imports and `load()` function)

#### Files to Reference

- `src/lmsys_query_analysis/db/sources.py` - import HuggingFaceSource
- `src/lmsys_query_analysis/db/loader.py` - import load_queries

#### Implementation Checklist

- [ ] Update imports (around line 17):
  - [ ] Add `from ...db.sources import HuggingFaceSource`
  - [ ] Change `from ...db.loader import load_lmsys_dataset` → `from ...db.loader import load_queries`
- [ ] Refactor `load()` function (lines 23-68):
  - [ ] Create `HuggingFaceSource` instance with dataset_id="lmsys/lmsys-chat-1m"
  - [ ] Pass `limit=limit` and `streaming=streaming` to source
  - [ ] Call `load_queries(db, source, chroma, ...)` instead of `load_lmsys_dataset()`
  - [ ] Keep all existing parameters and output formatting

---

### Section 1a.7: Update CLI Tests

**Objective**: Update test_data.py to mock new functions

#### Files to Modify

- `tests/unit/cli/commands/test_data.py` (update mocks)

#### Files to Reference

- `tests/unit/cli/commands/test_data.py` - existing test patterns

#### Implementation Checklist

- [ ] Update mock patches to use `load_queries` instead of `load_lmsys_dataset`
- [ ] Add mocks for `HuggingFaceSource` if needed
- [ ] Verify all CLI tests still pass

---

### Section 1a.8: Run Full Test Suite

**Objective**: Verify no regressions across entire codebase

#### Testing Checklist

- [ ] Run unit tests: `uv run pytest tests/unit/ -v`
- [ ] Run integration tests: `uv run pytest tests/integration/ -v`
- [ ] Run smoke tests: `uv run pytest tests/smoke/ -v`
- [ ] Manual smoke test: `uv run lmsys load --limit 100`
  - [ ] Verify data loads successfully
  - [ ] Verify stats table displays correctly
  - [ ] Verify queries in database
- [ ] Check for any deprecation warnings or errors
- [ ] Verify backward compatibility: existing workflows should work identically

---

## Phase 1b: Add CSV Support

### Section 1b.1: Implement CSVSource

**Objective**: Add CSV source implementation to sources.py

#### Files to Modify

- `src/lmsys_query_analysis/db/sources.py` (add ~150 lines)

#### Files to Reference

- `src/lmsys_query_analysis/db/sources.py` - existing BaseSource class
- Python csv module documentation

#### Implementation Checklist

- [ ] Implement `CSVSource` class:
  - [ ] Accept `file_path` in constructor
  - [ ] Validate file exists in `validate_source()`
  - [ ] Required columns: `conversation_id`, `query_text` (exact, case-sensitive)
  - [ ] Optional columns: `model`, `language`, `timestamp`
  - [ ] Use `csv.DictReader` for parsing
  - [ ] In `validate_source()`: open file, check headers, raise ValueError if missing required
  - [ ] In `iter_records()`:
    - [ ] Yield records with `query_text` directly (no conversation parsing)
    - [ ] Skip rows with empty `conversation_id` or `query_text`
    - [ ] Default `model` to "unknown" if not provided
    - [ ] Parse timestamp with `datetime.fromisoformat()`, set None on error
    - [ ] Log warnings for skipped/invalid rows
  - [ ] Return label like `"csv:path/to/file.csv"`

---

### Section 1b.2: Test CSVSource

**Objective**: Write comprehensive tests for CSVSource before using it

⚠️ **IMPORTANT**: Write tests + fixtures before integrating CSV into CLI

#### Files to Modify

- `tests/unit/db/test_sources.py` (add ~200 lines for CSV tests)

#### Files to Create

- `tests/fixtures/valid_queries.csv`
- `tests/fixtures/invalid_headers.csv` (missing query_text)
- `tests/fixtures/empty_fields.csv` (has empty conversation_id/query_text)

#### Testing Checklist

- [ ] Create test fixtures:
  - [ ] `valid_queries.csv`: all columns, 5 valid rows
  - [ ] `invalid_headers.csv`: missing `query_text` column
  - [ ] `empty_fields.csv`: some rows with empty required fields
- [ ] Test `CSVSource`:
  - [ ] Valid CSV → yields correct records
  - [ ] Missing `conversation_id` column → ValueError with message
  - [ ] Missing `query_text` column → ValueError with message
  - [ ] Empty `conversation_id` → skip row, count in stats
  - [ ] Empty `query_text` → skip row, count in stats
  - [ ] Invalid timestamp → set None, continue
  - [ ] Missing optional columns → use defaults
  - [ ] Extra columns → ignored
  - [ ] `get_source_label()` returns correct path

---

### Section 1b.3: Multi-Source Loading

**Objective**: Add support for loading from multiple sources in one session

#### Files to Modify

- `src/lmsys_query_analysis/db/loader.py` (add ~50 lines)

#### Files to Reference

- `src/lmsys_query_analysis/db/loader.py` - existing `load_queries()` implementation

#### Implementation Checklist

- [ ] Create `load_queries_from_multiple(db, sources, chroma, ...)`:
  - [ ] Accept list of BaseSource instances
  - [ ] Validate sources is non-empty list
  - [ ] Maintain single session with PRAGMAs applied once
  - [ ] Maintain global `seen_conv_ids` set across sources (dedupes across all sources)
  - [ ] Loop through sources, call `load_queries()` for each
  - [ ] Pass shared `seen_conv_ids` to each call
  - [ ] Track per-source stats in a list
  - [ ] Return list of stats dicts (one per source)

---

### Section 1b.4: Test Multi-Source Loading

**Objective**: Test multi-source loading with CSV fixtures

#### Files to Create

- `tests/integration/test_multi_source.py` (~200 lines)
- `tests/fixtures/dataset1.csv` (3 rows: conv_ids a, b, c)
- `tests/fixtures/dataset2.csv` (3 rows: conv_ids c, d, e - c overlaps with dataset1)

#### Files to Reference

- `tests/conftest.py` - DB fixtures (temp_db)
- `tests/unit/db/test_loader.py` - similar test patterns

#### Testing Checklist

- [ ] Create fixtures:
  - [ ] `dataset1.csv`: conv_ids "a", "b", "c"
  - [ ] `dataset2.csv`: conv_ids "c", "d", "e" (c is duplicate)
- [ ] Test multi-source loading:
  - [ ] Load dataset1.csv → 3 loaded
  - [ ] Load dataset1.csv again → 0 loaded, 3 skipped
  - [ ] Load both dataset1 + dataset2 → 5 loaded (c skipped in dataset2)
  - [ ] Verify per-source stats correct
  - [ ] Verify all expected rows in DB

---

### Section 1b.5: Update CLI for CSV Support

**Objective**: Add CLI options for CSV loading with proper validation

#### Files to Modify

- `src/lmsys_query_analysis/cli/commands/data.py` (lines 23-68 - `load()` function)
- `src/lmsys_query_analysis/cli/formatters/tables.py` (add `format_multi_source_stats_table()`)

#### Files to Reference

- `src/lmsys_query_analysis/cli/common.py` - option patterns
- `src/lmsys_query_analysis/cli/formatters/tables.py` (lines 9-28) - existing `format_loading_stats_table()`
- `src/lmsys_query_analysis/db/sources.py` - import CSVSource
- `src/lmsys_query_analysis/db/loader.py` - import load_queries_from_multiple

#### Implementation Checklist

- [ ] Update `load()` function signature:
  - [ ] Add `csv: List[str] = typer.Option(None, "--csv", help="CSV file path(s)")`
  - [ ] Add `hf_dataset: str = typer.Option(None, "--hf-dataset", help="HuggingFace dataset ID")`
  - [ ] Keep existing options unchanged
- [ ] Add validation logic:
  - [ ] If both `--csv` and `--hf-dataset` → error: "Cannot specify both"
  - [ ] If neither → default HuggingFaceSource("lmsys/lmsys-chat-1m")
- [ ] Build source list:
  - [ ] If `csv`: create CSVSource for each path
  - [ ] If `hf_dataset`: create single HuggingFaceSource
  - [ ] Pass limit/streaming to HuggingFaceSource only
- [ ] Call appropriate loader:
  - [ ] Single source → call `load_queries()` directly
  - [ ] Multiple CSVs → call `load_queries_from_multiple()`
- [ ] Display stats:
  - [ ] Single source → use existing `format_loading_stats_table()`
  - [ ] Multiple sources → use new `format_multi_source_stats_table()`
- [ ] In `tables.py`, add `format_multi_source_stats_table(stats_list)`:
  - [ ] Per-source rows: Source | Processed | Loaded | Skipped | Errors
  - [ ] Separator line
  - [ ] Total summary row

---

### Section 1b.6: Test CLI Integration

**Objective**: Test CLI CSV functionality end-to-end

#### Files to Create/Modify

- `tests/integration/test_cli_csv.py` (~150 lines)

#### Files to Reference

- `tests/integration/test_cli_load.py` - CLI test patterns (if exists)
- `tests/unit/cli/commands/test_data.py` - existing CLI test patterns
- `tests/fixtures/valid_queries.csv` - use existing fixture

#### Testing Checklist

- [ ] Test CLI with single CSV:
  - [ ] `lmsys load --csv valid_queries.csv`
  - [ ] Verify rows in DB
  - [ ] Verify stats table output
- [ ] Test CLI with multiple CSVs:
  - [ ] `lmsys load --csv dataset1.csv --csv dataset2.csv`
  - [ ] Verify multi-source stats table
  - [ ] Verify correct row counts
- [ ] Test mutual exclusivity:
  - [ ] `lmsys load --csv file.csv --hf-dataset org/data` → error
  - [ ] Verify clear error message
- [ ] Test backward compatibility:
  - [ ] `lmsys load --limit 100` (no source args)
  - [ ] Verify loads LMSYS as before
- [ ] Test CSV with ChromaDB:
  - [ ] `lmsys load --csv dataset1.csv --use-chroma`
  - [ ] Verify embeddings created
  - [ ] Verify count matches loaded rows
- [ ] Test validation errors:
  - [ ] Load CSV with missing columns → clear error
  - [ ] Verify error message helpful

---

### Section 1b.7: Documentation

**Objective**: Document CSV loading feature for agents and users

#### Files to Modify

- `CLAUDE.md` (add new section after existing content)
- `README.md` (update with CSV examples if needed)

#### Implementation Checklist

- [ ] Add "CSV Data Loading" section:
  - [ ] Required columns: `conversation_id`, `query_text`
  - [ ] Optional columns: `model`, `language`, `timestamp`
  - [ ] Timestamp format: ISO-8601 (e.g., "2024-10-13T12:34:56Z")
  - [ ] File format: UTF-8 encoded CSV
  - [ ] Example CSV snippet
- [ ] Add "Usage Examples":
  - [ ] Load single CSV: `lmsys load --csv data.csv`
  - [ ] Load multiple CSVs: `lmsys load --csv data1.csv --csv data2.csv`
  - [ ] Load HF dataset: `lmsys load --hf-dataset org/dataset`
  - [ ] Default (LMSYS): `lmsys load --limit 1000`
  - [ ] With ChromaDB: `lmsys load --csv data.csv --use-chroma`
- [ ] Add "Validation & Errors":
  - [ ] Missing required columns → clear error
  - [ ] Empty fields → row skipped with warning
  - [ ] Invalid timestamp → warning, continues
  - [ ] Cannot mix CSV and HF in one command
- [ ] Add "Phase 1 Limitations":
  - [ ] No dataset_id in schema yet
  - [ ] conversation_id must be unique across all sources
  - [ ] CSV must use exact column names (no mapping)

---

## Files Summary

**Phase 1a - Create:**

- `src/lmsys_query_analysis/db/sources.py` (BaseSource + HuggingFaceSource)
- `tests/unit/db/test_sources.py` (HuggingFaceSource tests)

**Phase 1a - Modify:**

- `src/lmsys_query_analysis/db/loader.py` (refactor to use sources, **DELETE** load_lmsys_dataset)
- `tests/unit/db/test_loader.py` (rewrite all tests to use load_queries, add new tests)
- `src/lmsys_query_analysis/runner.py` (update load_data() to use HuggingFaceSource + load_queries)
- `src/lmsys_query_analysis/cli/commands/data.py` (update load() to use HuggingFaceSource + load_queries)
- `tests/unit/cli/commands/test_data.py` (update mocks to use load_queries)

**Phase 1b - Modify:**

- `src/lmsys_query_analysis/db/sources.py` (add CSVSource)
- `tests/unit/db/test_sources.py` (add CSVSource tests)
- `src/lmsys_query_analysis/db/loader.py` (add load_queries_from_multiple function)
- `src/lmsys_query_analysis/cli/commands/data.py` (add CSV options)
- `src/lmsys_query_analysis/cli/formatters/tables.py` (add format_multi_source_stats_table)
- `CLAUDE.md` (documentation)
- `README.md` (documentation - optional)

**Phase 1b - Create:**

- `tests/integration/test_multi_source.py` (multi-source loader tests)
- `tests/integration/test_cli_csv.py` (CLI CSV tests)
- `tests/fixtures/valid_queries.csv`
- `tests/fixtures/invalid_headers.csv`
- `tests/fixtures/empty_fields.csv`
- `tests/fixtures/dataset1.csv`
- `tests/fixtures/dataset2.csv`