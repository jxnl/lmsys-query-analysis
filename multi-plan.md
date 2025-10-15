# Multi-Source Data Ingestion Implementation Plan

**Goal**: Add support for loading data from both Hugging Face datasets and local CSV files through the `lmsys load` command.

**Status**: Phase 1 Complete ✅ | Phase 2 Complete ✅ | Phase 3 Ready 🎯

**Testing Philosophy**: ⚠️ **TEST AS WE GO** - Write tests immediately after implementing each component. Don't save testing for the end!

## Recent Progress

### Phase 1 Completed ✅
- ✅ Created adapter pattern infrastructure (`BaseAdapter` protocol, `RecordDict` format)
- ✅ Implemented `HuggingFaceAdapter` with full test coverage
- ✅ Refactored `load_lmsys_dataset()` → `load_dataset()` to use adapters
- ✅ Updated all loader tests (19/19 passing) to use new adapter-based API
- ✅ Updated `runner.py` and CLI `data.py` to use new function
- ✅ All existing tests passing - zero regressions

### Phase 2 Completed ✅
- ✅ Added `--hf` CLI option with default value "lmsys/lmsys-chat-1m"
- ✅ Updated load command to accept and use `--hf` parameter
- ✅ Wired up HuggingFaceAdapter creation with dataset parameter
- ✅ Added comprehensive unit tests for CLI flag (4 tests in test_data.py)
- ✅ Added integration tests for CLI (4 tests in test_cli_hf.py)
- ✅ Verified backwards compatibility (no flag = default dataset)
- ✅ Updated help text and docstrings with examples
- ✅ Manual testing: `lmsys load --help` shows correct output
- ✅ All 406 tests passing (402 + 4 new integration tests) - zero regressions

### Next: Phase 3 - Documentation & Examples (HF Only) 🎯
Update documentation with `--hf` flag usage examples in README.md, docs, and CLAUDE.md.

---

## Progress Overview

| Phase | Status | Description | Tests |
|-------|--------|-------------|-------|
| **Phase 1** | ✅ Complete | Core Infrastructure & HuggingFaceAdapter | 19 loader + adapter tests |
| **Phase 2** | ✅ Complete | CLI Integration with `--hf` flag | 4 unit + 4 integration tests |
| **Phase 3** | 🎯 Ready | Documentation & Examples | N/A |
| **Phase 4** | ⏳ Pending | Error Handling & Edge Cases | TBD |
| **Phase 5** | ⏳ Pending | Final Validation & Cleanup | TBD |

**Current Status:** 406 tests passing | Zero regressions | Ready for Phase 3

---

## Implementation Strategy

**🎯 PHASED APPROACH**: We will implement data sources in two separate, complete phases:

### **Phase I: Hugging Face Support (Do First)**
Complete end-to-end implementation of `--hf <dataset_name>` flag, including:
- Adapter infrastructure
- CLI integration
- Testing & validation
- Documentation
- **Exit Criteria**: Users can load from any Hugging Face dataset using `--hf` flag

### **Phase II: CSV Support (Do After HF Works)**
Complete end-to-end implementation of `--csv <path>` flag, building on the adapter infrastructure from Phase I.

**⚠️ CRITICAL**: Do NOT start CSV implementation until Hugging Face support is fully working and tested!

---

## Overview

This plan implements multi-source ingestion for the `lmsys load` command with two new mutually exclusive flags:
- `--hf <dataset_name>` - Load from Hugging Face dataset (train split) **[PHASE I - DO FIRST]**
- `--csv <path/to/file.csv>` - Load from local CSV file **[PHASE II - DO AFTER HF]**

All existing functionality (batching, deduplication, ChromaDB integration) remains unchanged.

### Key Implementation Principles

1. **Phased Delivery**: Complete HF support end-to-end before starting CSV
2. **Test-Driven Development**: Write tests immediately after each component implementation
3. **Incremental Validation**: Run tests and verify they pass before moving to the next component
4. **Backwards Compatibility**: 100% compatibility - existing commands work unchanged
5. **No Big Bang Testing**: Final validation phases are for verification only, not discovering bugs

### Implementation Rhythm (Repeat for Every Component)

```
┌─────────────────────────────────────────────────────────┐
│  1. Implement Component (e.g., CSVAdapter)              │
│  2. Write Tests Immediately                             │
│  3. Run Tests: uv run pytest                            │
│  4. ✅ Tests Pass? → Move to next component             │
│  5. ❌ Tests Fail? → Fix immediately, then move on      │
└─────────────────────────────────────────────────────────┘
```

**NOT this:**
```
❌ 1. Implement everything
❌ 2. Write all tests
❌ 3. Discover everything is broken
❌ 4. Spend hours debugging
```

---

# PART 1: HUGGING FACE IMPLEMENTATION 🎯

Complete end-to-end support for `--hf <dataset_name>` flag before moving to CSV.

**Progress**: Phase 1 ✅ Complete | Phase 2 ✅ Complete | Phase 3 🎯 Ready to Start | Phases 4-5 ⏳ Pending

---

## Phase 1: Core Infrastructure & HuggingFaceAdapter ✅ COMPLETE

**Goal**: Create the adapter pattern and implement Hugging Face support ONLY.

**Scope**: Hugging Face adapter only - NO CSV implementation in this phase!

**Completed**: All adapter infrastructure and HuggingFaceAdapter implemented, loader refactored to use adapters, all tests updated and passing (19/19 loader tests).

### Files to Modify/Create

- `src/lmsys_query_analysis/db/loader.py` - Core loader refactoring
- `src/lmsys_query_analysis/db/adapters.py` - **NEW** - Source adapters (HF only for now)

### Tasks

- [x] **1.1**: Create `adapters.py` with base adapter interface
  - [x] Define `BaseAdapter` protocol/ABC with `iter_records()` method
  - [x] Define `RecordDict` TypedDict for normalized record format
  - [x] Add docstrings explaining adapter contract

- [x] **1.2**: Implement `HuggingFaceAdapter` ONLY
  - [x] Extract existing HF loading logic from `load_lmsys_dataset()`
  - [x] Accept `dataset_name` parameter (defaults to "lmsys/lmsys-chat-1m")
  - [x] Always use "train" split
  - [x] Support streaming and non-streaming modes
  - [x] Handle limit parameter correctly
  - [x] Yield normalized `RecordDict` instances

- [x] **1.3**: Replace `load_lmsys_dataset()` with generic `load_dataset()`
  - [x] Rename function to `load_dataset()` (no alias - complete replacement)
  - [x] Accept `adapter: BaseAdapter` parameter
  - [x] Replace HF-specific logic with `adapter.iter_records()`
  - [x] Keep all batching, deduplication, ChromaDB logic unchanged
  - [x] Update docstring to reflect generic nature
  - [x] Ensure stats reporting works the same

- [x] **1.4**: ~~Add helper function for creating HF adapters~~ (SKIPPED - not needed)
  - ~~`create_adapter(source_type, **kwargs)` factory function (HF only for now)~~
  - ~~Clear error messages for invalid source types~~

### Implementation & Testing Approach (Phase 1)

**Test as we build each component! HF ONLY - NO CSV!**

- [x] **1.1a**: Create `adapters.py` with base interface (as specified above)
- [x] **1.1b**: Write tests for adapter interface before implementations
  - [x] Create `tests/unit/db/test_adapters.py`
  - [x] Define test fixtures for mock records

- [x] **1.2a**: Implement `HuggingFaceAdapter` (as specified above)
- [x] **1.2b**: **TEST IT IMMEDIATELY** - Unit tests for HuggingFaceAdapter
  - [x] Test with mocked HF dataset
  - [x] Verify record normalization
  - [x] Test limit parameter
  - [x] Test streaming vs non-streaming
  - [x] **Run tests and verify they pass before moving on**

- [x] **1.3a**: Replace `load_lmsys_dataset()` with `load_dataset()` (as specified above)
- [x] **1.3b**: **TEST IT IMMEDIATELY** - Integration test for refactored loader
  - [x] Update `tests/unit/db/test_loader.py` to use `load_dataset()`
  - [x] Test with HuggingFaceAdapter produces same results as before  
  - [x] Update runner.py and CLI data.py to use new function
  - [x] All 19 loader tests passing - infrastructure verified working ✅

- [x] **1.4a**: ~~Add helper function for creating adapters~~ (SKIPPED - not needed)
- [x] **1.4b**: ~~Unit tests for factory function~~ (SKIPPED - not needed)

**Files**: 
- `src/lmsys_query_analysis/db/adapters.py` (NEW - HF only)
- `src/lmsys_query_analysis/db/loader.py` (UPDATE)
- `tests/unit/db/test_adapters.py` (NEW - HF tests only)
- `tests/unit/db/test_loader.py` (UPDATE)

**Phase 1 Exit Criteria:**
- ✅ All Phase 1 unit tests passing (19/19 loader tests + adapter tests)
- ✅ All existing tests still passing (no regressions)
- ✅ Can load data from HF programmatically with adapter pattern
- ✅ ChromaDB integration works with adapter

**Phase 1 Status: COMPLETE ✅** (All tests passing, ready for Phase 2)

---

## Phase 2: CLI Integration (HF Only) ✅ COMPLETE

**Goal**: Add `--hf` CLI flag and wire up HuggingFaceAdapter to the load command.

**Scope**: HF flag only - NO CSV flag in this phase!

### Files Modified ✅

- ✅ `src/lmsys_query_analysis/cli/commands/data.py` - CLI command layer
- ✅ `tests/unit/cli/commands/test_data.py` - Unit tests (4 tests updated/added)
- ✅ `tests/integration/test_cli_hf.py` - NEW - Integration tests (4 tests added)
- ✅ `tests/integration/test_cli.py` - Updated help text assertion

### Tasks

- [x] **2.1**: Add `--hf` CLI option to `load()` command
  - [x] Add `--hf` option with type `Optional[str]`, help text
  - [x] **BACKWARDS COMPATIBILITY**: If `--hf` not provided, default to `"lmsys/lmsys-chat-1m"`
  - [x] Keep existing flags (`--limit`, `--use-chroma`, etc.) working
  - [x] Optionally show info message when using default HF dataset

- [x] **2.2**: Wire up HF adapter creation
  - [x] Based on `--hf` flag value (or default), create HuggingFaceAdapter
  - [x] Pass adapter to refactored loader function
  - [x] Ensure all existing options (limit, chroma, etc.) work with adapter

- [x] **2.3**: Update help text and examples
  - [x] Update command docstring with new `--hf` usage examples
  - [x] Show HF examples in `--help`

### Implementation & Testing Approach (Phase 2)

**Test each CLI change immediately! HF ONLY!**

- [x] **2.1a**: Add `--hf` CLI option (as specified above)
- [x] **2.1b**: **TEST IT IMMEDIATELY** - CLI flag validation
  - [x] Update `tests/unit/cli/commands/test_data.py`
  - [x] Test with `--hf <dataset>` (should work)
  - [x] **CRITICAL: Test with no `--hf` flag (should work - backwards compatible default)**
  - [x] Verify default uses "lmsys/lmsys-chat-1m" dataset
  - [x] **Run tests and verify they pass before moving on**

- [x] **2.2a**: Wire up HF adapter creation (as specified above)
- [x] **2.2b**: **TEST IT IMMEDIATELY** - Adapter wiring
  - [x] Mock adapter creation and verify HuggingFaceAdapter is used
  - [x] Test with `--hf` flag creates HuggingFaceAdapter with correct dataset
  - [x] Test without flag creates HuggingFaceAdapter with default dataset
  - [x] Verify all other options passed through correctly (limit, chroma, etc.)
  - [x] **Run tests and verify they pass**

- [x] **2.3a**: Update help text and examples (as specified above)
- [x] **2.3b**: **VERIFY IT IMMEDIATELY** - Manual testing
  - [x] Run `lmsys load --help` and verify output
  - [x] Check that examples are clear and accurate

- [x] **2.4**: **END-TO-END CLI TEST**
  - [x] Created `tests/integration/test_cli_hf.py` with 4 integration tests
  - [x] Test default HF dataset (backwards compatibility)
  - [x] Test explicit `--hf` flag with custom dataset
  - [x] Test `--hf` with `--streaming` flag
  - [x] Test `--hf` with `--use-chroma` flag
  - [x] **All 406 tests passing - ready for Phase 3**

**Files**:
- `src/lmsys_query_analysis/cli/commands/data.py`
- `tests/unit/cli/commands/test_data.py`
- `tests/integration/test_cli_hf.py` (NEW - HF integration tests)

**Phase 2 Exit Criteria:** ✅ ALL COMPLETE
- ✅ All Phase 2 CLI tests passing (4 unit + 4 integration)
- ✅ All existing CLI tests still passing
- ✅ Backwards compatibility verified (commands without flags work)
- ✅ Can load from HF via CLI with `--hf` flag
- ✅ Can load from default HF dataset without flags
- ✅ Help text is clear and accurate

**Phase 2 Status: COMPLETE ✅** (406 tests passing, ready for Phase 3)

### Phase 2 Summary - What We Accomplished

**Implementation Changes:**
1. Added `--hf` flag to `load()` command with default value `"lmsys/lmsys-chat-1m"`
2. Wired up adapter creation: `HuggingFaceAdapter(dataset_name=hf, use_streaming=streaming)`
3. Updated help text with clear examples showing both default and explicit usage
4. Added dataset name to startup output message

**Test Coverage Added:**
- **Unit Tests** (`tests/unit/cli/commands/test_data.py`):
  - `test_load_command_basic` - Updated to mock adapter
  - `test_load_command_with_custom_hf_dataset` - Custom HF dataset
  - `test_load_command_without_hf_flag_uses_default` - Backwards compatibility
  - `test_load_command_with_streaming` - Streaming integration

- **Integration Tests** (`tests/integration/test_cli_hf.py` - NEW FILE):
  - `test_load_with_default_hf_dataset` - CLI without --hf flag
  - `test_load_with_explicit_hf_dataset` - CLI with custom dataset
  - `test_load_with_hf_and_streaming` - CLI with streaming
  - `test_load_with_hf_and_chroma` - CLI with ChromaDB

**Test Results:**
- ✅ 406 tests passing (402 existing + 4 new integration tests)
- ✅ 1 skipped
- ✅ Zero regressions - all existing functionality preserved

**Backwards Compatibility:**
- ✅ `lmsys load --limit 10000` works without `--hf` flag (uses default)
- ✅ All existing commands continue to work exactly as before

---

## Phase 3: Documentation & Examples (HF Only)

**Goal**: Update documentation with `--hf` flag usage examples.

**Scope**: HF documentation only - NO CSV documentation in this phase!

### Files to Create/Modify

- `README.md` - Update usage examples
- `docs/cli/load.md` - CLI documentation (if exists)
- `CLAUDE.md` - Update for agents

### Tasks

- [ ] **3.1**: Update README.md
  - [ ] Add `--hf` flag documentation to "Data Loading" section
  - [ ] Show examples of loading default dataset (no flags)
  - [ ] Show examples of loading custom HF datasets with `--hf`
  - [ ] Document backwards compatibility

- [ ] **3.2**: Update CLI documentation (if exists)
  - [ ] Update `docs/cli/load.md` with `--hf` flag
  - [ ] Show full examples with different HF datasets
  - [ ] Show examples with different embedding models

- [ ] **3.3**: Update agent documentation
  - [ ] Update `CLAUDE.md` with new `--hf` capability
  - [ ] Show example workflow for custom HF datasets
  - [ ] Document that adapter pattern is ready for future sources

**Files**:
- `README.md`
- `docs/cli/load.md` (UPDATE if exists)
- `CLAUDE.md`

---

## Phase 4: Error Handling & Edge Cases (HF Only)

**Goal**: Robust error handling and helpful error messages for HF datasets.

**Scope**: HF error handling only - NO CSV error handling in this phase!

### Tasks

- [ ] **4.1**: HF dataset validation
  - [ ] Handle dataset not found errors (show clear message)
  - [ ] Handle authentication errors (remind about `huggingface-cli login`)
  - [ ] Handle network errors gracefully
  - [ ] Validate dataset has expected fields

- [ ] **4.2**: Improved error messages for HF
  - [ ] "Dataset not found" → Show HF Hub link, suggest checking name
  - [ ] "Authentication required" → Show login command
  - [ ] "Missing field" → List expected fields and what's present
  - [ ] "Malformed conversation" → Show record ID and issue

- [ ] **4.3**: Add data validation warnings
  - [ ] Warn if >50% of records have missing required fields
  - [ ] Warn if conversation_id not unique
  - [ ] Show stats about data quality (% valid records)

### Implementation & Testing Approach (Phase 4)

**Test error cases as we implement them! HF ONLY!**

- [ ] **4.1a**: Implement HF dataset validation (as specified above)
- [ ] **4.1b**: **TEST IT IMMEDIATELY** - Validation tests
  - [ ] Update `tests/unit/db/test_adapters.py`
  - [ ] Test dataset not found error
  - [ ] Test authentication error handling
  - [ ] Test network error handling
  - [ ] Test missing field validation
  - [ ] **Run tests and verify they pass before moving on**

- [ ] **4.2a**: Implement improved error messages for HF (as specified above)
- [ ] **4.2b**: **TEST IT IMMEDIATELY** - Error message tests
  - [ ] Test error message for dataset not found (verify helpful suggestions)
  - [ ] Test error message for auth errors (verify login command shown)
  - [ ] Test error message for missing fields (verify expected vs actual)
  - [ ] **Manually verify error messages are clear and helpful**
  - [ ] **Run tests and verify they pass**

- [ ] **4.3a**: Implement data validation warnings (as specified above)
- [ ] **4.3b**: **TEST IT IMMEDIATELY** - Warning tests
  - [ ] Test warning when >50% records have missing fields
  - [ ] Test warning for non-unique conversation_ids
  - [ ] Test data quality stats display
  - [ ] Verify warnings don't block execution
  - [ ] **Run tests and verify they pass**

**Files**:
- `src/lmsys_query_analysis/db/adapters.py`
- `tests/unit/db/test_adapters.py`

**Phase 4 Exit Criteria:**
- ✅ All error handling tests passing
- ✅ Error messages are clear and actionable
- ✅ HF-specific errors handled gracefully
- ✅ Validation warnings work correctly

---

## Phase 5: Final Integration & Smoke Tests (HF Only)

**Goal**: Final validation of HF support now that components are tested individually.

**Scope**: HF validation only - CSV will be validated in PART 2!

Since we've been testing throughout Phases 1-4, this phase is much lighter - just final integration checks and smoke tests.

### Tasks & Testing (Phase 5)

- [ ] **5.1**: HF dataset loading workflow test
  - [ ] Load default HF data (small limit, no flags)
  - [ ] Verify data in database
  - [ ] Run `lmsys clear`
  - [ ] Load custom HF dataset with `--hf` flag
  - [ ] Verify data loaded correctly
  - [ ] **Verify workflow is clean and no issues**

- [ ] **5.2**: Downstream command compatibility test
  - [ ] Load HF data (100 rows) with `--hf` flag
  - [ ] Run `lmsys cluster kmeans --n-clusters 10`
  - [ ] Run `lmsys summarize <run_id>`
  - [ ] Run `lmsys search "test query"`
  - [ ] Run `lmsys export <run_id>`
  - [ ] **Verify all downstream commands work identically with custom HF datasets**

- [ ] **5.3**: End-to-end smoke test
  - [ ] Run full pipeline with default HF dataset: load → cluster → summarize → search
  - [ ] Run full pipeline with custom HF dataset using `--hf` flag
  - [ ] **Verify smoke test passes for both default and explicit HF modes**

- [ ] **5.4**: Backwards compatibility regression validation
  - [ ] Run `uv run pytest -v` - **ALL tests must pass**
  - [ ] Run existing README commands without modification
  - [ ] Verify `lmsys load --limit 100` uses default HF dataset (backwards compatible!)
  - [ ] Test that help text is clear and doesn't confuse existing users
  - [ ] **Zero regressions allowed**

- [ ] **5.5**: Performance spot-check (optional)
  - [ ] Load 1000 rows from default HF dataset
  - [ ] Load 1000 rows from custom HF dataset with `--hf`
  - [ ] Compare load times (should be identical)
  - [ ] Verify no performance degradation

**Files**:
- `tests/integration/test_hf_integration.py` (NEW)
- `smoketest.sh` (UPDATE if needed)

**Phase 5 Exit Criteria:**
- ✅ Full test suite passes (uv run pytest)
- ✅ Smoke tests pass for HF datasets
- ✅ HF dataset switching workflow works
- ✅ All downstream commands work with custom HF datasets
- ✅ Backwards compatibility verified - zero regressions
- ✅ README commands work without modification
- ✅ **READY FOR PART 2: CSV IMPLEMENTATION**

---

# END OF PART 1: HUGGING FACE IMPLEMENTATION ✅

**At this point, users should be able to:**
- Load from default LMSYS dataset (backwards compatible, no flags)
- Load from any Hugging Face dataset using `--hf <dataset_name>`
- Use all existing features (clustering, search, etc.) with custom HF datasets
- See clear error messages for HF-specific issues

**Do NOT proceed to PART 2 until all Phase 1-5 exit criteria are met!**

---

---

# PART 2: CSV IMPLEMENTATION 📋

**⚠️ CRITICAL**: Only start this after PART 1 (Hugging Face) is fully complete and tested!

Complete end-to-end support for `--csv <path>` flag, building on the adapter infrastructure from PART 1.

---

## Phase 6: CSVAdapter Implementation

**Goal**: Implement CSV data source adapter and integrate with existing loader.

**Prerequisites**: PART 1 must be complete - adapter infrastructure exists!

### Files to Modify/Create

- `src/lmsys_query_analysis/db/adapters.py` - Add CSVAdapter
- `tests/unit/db/test_adapters.py` - Add CSV tests
- `tests/fixtures/sample_data.csv` - Test data

### Tasks

- [ ] **6.1**: Implement `CSVAdapter`
  - [ ] Use Python's `csv.DictReader` for parsing
  - [ ] Expect columns: `id,text,timestamp,model,language,metadata`
  - [ ] Map CSV columns to normalized `RecordDict`:
    - `id` → `conversation_id`
    - `text` → use for query_text directly
    - `timestamp` → parse ISO format or pass through
    - `model` → default to "unknown" if missing/empty
    - `language` → default to None if missing/empty
    - `metadata` → parse as JSON string or use empty dict
  - [ ] Handle missing/malformed rows gracefully (log and skip)
  - [ ] Support limit parameter (stop after N valid rows)
  - [ ] Add file existence check with clear error message
  - [ ] Stream processing (don't load entire file into memory)

- [ ] **6.2**: Update factory function
  - [ ] Update `create_adapter()` to support "csv" source type
  - [ ] Pass file path to CSVAdapter

- [ ] **6.3**: CSV validation and error handling
  - [ ] Check file exists before processing
  - [ ] Validate CSV headers match expected schema
  - [ ] Show clear error if columns missing
  - [ ] Handle BOM and encoding issues (UTF-8)
  - [ ] Detect and warn about empty files

### Implementation & Testing Approach (Phase 6)

**Test CSV adapter thoroughly!**

- [ ] **6.1a**: Implement `CSVAdapter` (as specified above)
- [ ] **6.1b**: **TEST IT IMMEDIATELY** - Unit tests for CSVAdapter
  - [ ] Create test CSV files in `tests/fixtures/`
  - [ ] Test with valid CSV file
  - [ ] Test missing columns (should handle gracefully or error clearly)
  - [ ] Test malformed rows (skip and log)
  - [ ] Test empty CSV
  - [ ] Test limit parameter
  - [ ] Test JSON metadata parsing
  - [ ] Test file not found error
  - [ ] **Run tests and verify they pass before moving on**

- [ ] **6.2a**: Update factory function (as specified above)
- [ ] **6.2b**: **TEST IT IMMEDIATELY** - Factory tests
  - [ ] Test creating CSVAdapter
  - [ ] Test error for invalid source type
  - [ ] **Run tests and verify they pass**

- [ ] **6.3a**: Implement CSV validation (as specified above)
- [ ] **6.3b**: **TEST IT IMMEDIATELY** - Validation tests
  - [ ] Test file not found error
  - [ ] Test missing column error
  - [ ] Test header validation
  - [ ] Test BOM and encoding edge cases
  - [ ] Test empty file detection
  - [ ] **Run tests and verify they pass**

**Files**:
- `src/lmsys_query_analysis/db/adapters.py` (UPDATE - add CSVAdapter)
- `tests/unit/db/test_adapters.py` (UPDATE - add CSV tests)
- `tests/fixtures/sample_data.csv` (NEW)
- `tests/fixtures/invalid_data.csv` (NEW)

**Phase 6 Exit Criteria:**
- ✅ All CSV adapter tests passing
- ✅ All existing tests still passing
- ✅ Can load data from CSV programmatically
- ✅ CSV validation works correctly

---

## Phase 7: CSV CLI Integration

**Goal**: Add `--csv` flag to CLI and wire up CSVAdapter.

### Files to Modify

- `src/lmsys_query_analysis/cli/commands/data.py` - Add CSV flag

### Tasks

- [ ] **7.1**: Add `--csv` CLI option
  - [ ] Add `--csv` option with type `Optional[str]`, help text
  - [ ] Add validation: `--hf` and `--csv` are mutually exclusive
  - [ ] Show clear error if BOTH provided
  - [ ] Handle file path resolution (absolute vs relative)

- [ ] **7.2**: Wire up CSV adapter creation
  - [ ] Based on flag, create CSVAdapter
  - [ ] Pass file path to adapter
  - [ ] Add file existence check with friendly error

- [ ] **7.3**: Update help text
  - [ ] Show CSV examples in `--help`
  - [ ] Document CSV format requirements

### Implementation & Testing Approach (Phase 7)

**Test CSV CLI thoroughly!**

- [ ] **7.1a**: Add `--csv` CLI option (as specified above)
- [ ] **7.1b**: **TEST IT IMMEDIATELY** - CLI validation
  - [ ] Update `tests/unit/cli/commands/test_data.py`
  - [ ] Test with `--csv` only (should work)
  - [ ] Test with both `--hf` and `--csv` (should error)
  - [ ] Test with neither flag (should use HF default)
  - [ ] **Run tests and verify they pass**

- [ ] **7.2a**: Wire up CSV adapter (as specified above)
- [ ] **7.2b**: **TEST IT IMMEDIATELY** - Adapter wiring
  - [ ] Test CSV flag creates CSVAdapter
  - [ ] Test file path resolution
  - [ ] **Run tests and verify they pass**

- [ ] **7.3**: **END-TO-END CLI TEST**
  - [ ] Create test CSV file
  - [ ] Run `lmsys load --csv test.csv --limit 10`
  - [ ] Verify data loaded correctly
  - [ ] Test with `--use-chroma` flag
  - [ ] **All tests must pass**

**Files**:
- `src/lmsys_query_analysis/cli/commands/data.py` (UPDATE)
- `tests/unit/cli/commands/test_data.py` (UPDATE)
- `tests/integration/test_cli_csv.py` (NEW)

**Phase 7 Exit Criteria:**
- ✅ All CSV CLI tests passing
- ✅ Can load from CSV via CLI
- ✅ Mutual exclusivity validation works
- ✅ Help text includes CSV examples

---

## Phase 8: CSV Documentation

**Goal**: Document CSV support with examples.

### Files to Modify/Create

- `README.md` - Add CSV examples
- `examples/sample_data.csv` - Example file
- `CLAUDE.md` - Update agent docs

### Tasks

- [ ] **8.1**: Create example CSV file
  - [ ] Create `examples/sample_data.csv` with 10-20 sample rows
  - [ ] Use realistic query examples
  - [ ] Include all required columns
  - [ ] Add README explaining format

- [ ] **8.2**: Update README.md
  - [ ] Add CSV format documentation
  - [ ] Show `--csv` examples
  - [ ] Document expected CSV schema
  - [ ] Add troubleshooting section

- [ ] **8.3**: Update agent documentation
  - [ ] Update `CLAUDE.md` with CSV capabilities
  - [ ] Show workflow for CSV ingestion
  - [ ] Document when to use HF vs CSV

**Files**:
- `README.md`
- `examples/sample_data.csv` (NEW)
- `CLAUDE.md`

---

## Phase 9: CSV Error Handling

**Goal**: Robust error handling for CSV files.

### Tasks

- [ ] **9.1**: Improved error messages
  - [ ] "File not found" → Show absolute path checked
  - [ ] "Missing column" → List expected vs actual
  - [ ] "Malformed row" → Show row number and issue
  - [ ] "Parse error" → Show line number

- [ ] **9.2**: Data validation warnings
  - [ ] Warn if >50% rows have missing fields
  - [ ] Warn if conversation_id not unique
  - [ ] Show stats about data quality

### Implementation & Testing

- [ ] **9.1a**: Implement error messages
- [ ] **9.1b**: **TEST IT IMMEDIATELY**
  - [ ] Test each error message
  - [ ] Verify messages are helpful
  - [ ] **Run tests and verify they pass**

- [ ] **9.2a**: Implement warnings
- [ ] **9.2b**: **TEST IT IMMEDIATELY**
  - [ ] Test validation warnings
  - [ ] Verify warnings don't block
  - [ ] **Run tests and verify they pass**

**Files**:
- `src/lmsys_query_analysis/db/adapters.py`
- `tests/unit/db/test_adapters.py`

---

## Phase 10: Final CSV Validation

**Goal**: End-to-end validation of CSV support.

### Tasks

- [ ] **10.1**: Source switching test
  - [ ] Load HF data, clear, load CSV
  - [ ] Load CSV data, clear, load HF
  - [ ] Verify clean switching

- [ ] **10.2**: Downstream compatibility
  - [ ] Load CSV data
  - [ ] Run clustering, search, export
  - [ ] Verify all commands work

- [ ] **10.3**: Performance check
  - [ ] Load 1000 rows from CSV
  - [ ] Compare to HF performance
  - [ ] Verify reasonable speed

- [ ] **10.4**: Full regression suite
  - [ ] Run `uv run pytest -v`
  - [ ] **ALL tests must pass**
  - [ ] Zero regressions

**Files**:
- `tests/integration/test_multi_source.py` (NEW)

**Phase 10 Exit Criteria:**
- ✅ Full test suite passes
- ✅ Source switching works
- ✅ All downstream commands work with CSV
- ✅ Performance is acceptable
- ✅ Zero regressions
- ✅ **IMPLEMENTATION COMPLETE!**

---

## Testing Strategy Summary

**Philosophy: Test as we go, not all at the end!**

Each phase follows an incremental test-as-you-build approach:
1. **Implement a component**
2. **Write tests immediately**
3. **Run tests and verify they pass**
4. **Move to next component only after tests pass**

This approach:
- Catches bugs early when they're easier to fix
- Prevents accumulation of technical debt
- Provides confidence at each step
- Makes debugging much easier (you know what you just changed)
- Reduces the "big bang" integration testing at the end

### PART 1: Hugging Face Testing

#### Unit Tests (Written During Implementation)
- Adapter interface and HuggingFaceAdapter → Phase 1
- Record normalization → Phase 1
- HF error handling → Phase 4
- CLI flag validation → Phase 2
- Adapter wiring → Phase 2

#### Integration Tests (Written During Implementation)
- Refactored loader with adapters → Phase 1
- CLI command execution → Phase 2
- ChromaDB integration → Phases 1 & 2

#### Final Validation (Phase 5)
- HF dataset loading workflow
- Downstream command compatibility
- End-to-end smoke tests
- Backwards compatibility regression tests
- Performance spot-checks

#### Test Data
- Mock HF datasets (Phase 1)
- Test fixtures in `tests/fixtures/` (Throughout)

### PART 2: CSV Testing

#### Unit Tests (Written During Implementation)
- CSVAdapter implementation → Phase 6
- CSV parsing and validation → Phase 6 & 9
- CSV error handling → Phase 9
- CLI flag validation for CSV → Phase 7
- Mutual exclusivity validation → Phase 7

#### Integration Tests (Written During Implementation)
- CSV CLI command execution → Phase 7
- ChromaDB integration with CSV → Phase 7
- Large file handling → Phase 6

#### Final Validation (Phase 10)
- Source switching workflow (HF ↔ CSV)
- Downstream command compatibility
- Performance comparison
- Full regression suite

#### Test Data
- Sample CSV files - valid, invalid, edge cases (Phases 6-9)
- Large CSV files for streaming tests (Phase 6)
- Test fixtures in `tests/fixtures/` (Throughout)

---

## Success Criteria

### PART 1: Hugging Face Implementation (Phases 1-5)

- [ ] CLI accepts `--hf <dataset_name>` flag
- [ ] **CRITICAL: 100% backwards compatible - existing commands work unchanged**
- [ ] Can load from any Hugging Face dataset
- [ ] Adapter pattern implemented and working
- [ ] ChromaDB integration works with adapter
- [ ] All existing tests pass (no regressions)
- [ ] New HF tests achieve >90% coverage for new code
- [ ] Documentation updated with `--hf` examples
- [ ] Error messages are clear and actionable for HF datasets
- [ ] Performance identical to existing implementation

### PART 2: CSV Implementation (Phases 6-10)

- [ ] CLI accepts `--csv <path>` flag
- [ ] `--hf` and `--csv` are mutually exclusive
- [ ] Both sources produce identical database schema
- [ ] ChromaDB integration works with CSV
- [ ] All tests pass (HF + CSV)
- [ ] CSV tests achieve >90% coverage
- [ ] Documentation updated with CSV examples and schema
- [ ] Error messages are clear for CSV issues
- [ ] Performance is acceptable for CSV loading

---

## Backwards Compatibility Guarantee

**CRITICAL**: All existing commands must continue to work without modification.

### Current Behavior (Before Changes)
```bash
# Users run this today - loads from lmsys/lmsys-chat-1m
uv run lmsys load --limit 10000 --use-chroma
```

### New Behavior (After PART 1: Hugging Face)

```bash
# ✅ MUST STILL WORK - Defaults to lmsys/lmsys-chat-1m
uv run lmsys load --limit 10000 --use-chroma

# ✅ NEW - Explicit HF dataset (same as default)
uv run lmsys load --hf lmsys/lmsys-chat-1m --limit 10000 --use-chroma

# ✅ NEW - Different HF dataset
uv run lmsys load --hf username/other-dataset --limit 10000 --use-chroma
```

### New Behavior (After PART 2: CSV)

```bash
# ✅ STILL WORKS - Defaults to lmsys/lmsys-chat-1m
uv run lmsys load --limit 10000 --use-chroma

# ✅ STILL WORKS - Explicit HF dataset
uv run lmsys load --hf username/other-dataset --limit 10000 --use-chroma

# ✅ NEW - CSV file
uv run lmsys load --csv ./data.csv --limit 10000 --use-chroma

# ❌ ERROR - Cannot specify both sources
uv run lmsys load --hf dataset --csv file.csv --limit 10000
```

### Implementation Notes

**PART 1 (HF Only):**
- No `--hf` flag provided → Default to `"lmsys/lmsys-chat-1m"`
- `--hf` provided → Use specified HF dataset

**PART 2 (HF + CSV):**
- Neither `--hf` nor `--csv` provided → Default to `"lmsys/lmsys-chat-1m"`
- Only `--hf` provided → Use specified HF dataset
- Only `--csv` provided → Use CSV file
- Both provided → Error with clear message about mutual exclusivity

### Migration Path (None Required!)

Users don't need to change anything. The default behavior is identical to current behavior throughout both parts.

**Optional**: Users can make their scripts more explicit by adding `--hf lmsys/lmsys-chat-1m`.

---

## Open Questions & Decisions

### 1. CSV Column Validation
**Question**: Should we fail fast if CSV is missing required columns, or skip malformed rows?

**Options**:
- A) Fail fast - Better for catching errors early
- B) Skip rows - More forgiving, useful for partial data

**Decision**: TBD during implementation (suggest: Fail fast for missing columns, skip for malformed data)

### 2. Source Mismatch Detection
**Question**: Should we automatically detect source mismatch and warn/block?

**Options**:
- A) Store source metadata in database, auto-detect and warn
- B) Just document that users should clear DB when switching
- C) Allow mixed sources (append mode)

**Decision**: TBD (suggest: Option B for v1 - document the workflow)

### 3. Large CSV Handling
**Question**: Should we impose limits on CSV file size?

**Options**:
- A) No limit - process any size with streaming
- B) Soft limit - warn above certain size
- C) Hard limit - reject files above threshold

**Decision**: TBD (suggest: Option A with streaming, Option B with warning at 100MB+)

### 4. Timestamp Parsing
**Question**: How strict should timestamp parsing be?

**Options**:
- A) Strict ISO 8601 only
- B) Try multiple formats
- C) Allow empty/null timestamps

**Decision**: TBD (suggest: Option C - allow empty, try ISO 8601 first)

---

## Dependencies

- Python's built-in `csv` module (no new dependencies!)
- All existing dependencies remain

---

## Timeline Estimate

With test-as-we-go approach (includes testing time in each phase):

### PART 1: Hugging Face Implementation (Do First!)

- **Phase 1** (Core Infrastructure & HF Adapter + Tests): 4-6 hours
  - Implementation: 2-3 hours
  - Testing: 2-3 hours (incremental)
  
- **Phase 2** (CLI Integration + Tests): 2-3 hours
  - Implementation: 1-1.5 hours
  - Testing: 1-1.5 hours (incremental)
  
- **Phase 3** (Documentation): 1 hour
  
- **Phase 4** (Error Handling + Tests): 2-3 hours
  - Implementation: 1-1.5 hours
  - Testing: 1-1.5 hours (incremental)
  
- **Phase 5** (Final Integration & Smoke Tests): 1-2 hours
  - Much lighter since we've been testing throughout!

**PART 1 Total**: 10-15 hours

**⚠️ CHECKPOINT**: Validate PART 1 is complete before starting PART 2!

### PART 2: CSV Implementation (Do After HF Works!)

- **Phase 6** (CSV Adapter + Tests): 3-4 hours
  - Implementation: 1.5-2 hours
  - Testing: 1.5-2 hours (incremental)
  
- **Phase 7** (CSV CLI Integration + Tests): 2-3 hours
  - Implementation: 1-1.5 hours
  - Testing: 1-1.5 hours (incremental)
  
- **Phase 8** (CSV Documentation): 1-2 hours
  
- **Phase 9** (CSV Error Handling + Tests): 2-3 hours
  - Implementation: 1-1.5 hours
  - Testing: 1-1.5 hours (incremental)
  
- **Phase 10** (Final CSV Validation): 1-2 hours

**PART 2 Total**: 9-14 hours

**Grand Total**: 19-29 hours

**Note**: This phased approach:
- Delivers value earlier (HF support first!)
- Allows validation at natural checkpoint (after PART 1)
- Catches bugs early when they're easier to fix
- No "surprise bugs" at the end
- More confidence throughout
- Actually saves time overall by avoiding difficult debugging sessions

---

## Implementation Order

Recommended order for implementation (by you or another agent):

### PART 1: Hugging Face Implementation (MUST DO FIRST!)

1. **Phase 1** - Core Infrastructure & HuggingFaceAdapter
   - Implement base adapter interface
   - Implement HuggingFaceAdapter ONLY (no CSV!)
   - Refactor loader to use adapters
   - **Write and run tests after each component**
   - Exit only when all Phase 1 tests pass

2. **Phase 2** - CLI Integration (HF Only)
   - Add `--hf` CLI flag
   - Wire up HuggingFaceAdapter
   - **Write and run tests after each change**
   - Exit only when all CLI tests pass and backwards compatibility verified

3. **Phase 3** - Documentation (HF Only)
   - Update README, CLAUDE.md with `--hf` examples
   - No complex testing needed

4. **Phase 4** - Error Handling (HF Only)
   - Add HF-specific error handling
   - **Write and run tests after each improvement**
   - Exit only when all error cases are tested

5. **Phase 5** - Final Integration & Smoke Tests (HF Only)
   - Light phase since we've been testing throughout
   - Run full regression suite
   - Verify backwards compatibility
   - Final smoke tests
   - **⚠️ VALIDATE PART 1 IS COMPLETE!**

### ⛔ CHECKPOINT: DO NOT START PART 2 UNTIL PART 1 IS COMPLETE!

### PART 2: CSV Implementation (DO AFTER HF WORKS!)

6. **Phase 6** - CSVAdapter Implementation
   - Implement CSVAdapter (adapter infrastructure already exists!)
   - **Write and run tests immediately**
   - Exit only when all CSV adapter tests pass

7. **Phase 7** - CSV CLI Integration
   - Add `--csv` CLI flag
   - Add mutual exclusivity validation
   - Wire up CSVAdapter
   - **Write and run tests after each change**
   - Exit only when all CSV CLI tests pass

8. **Phase 8** - CSV Documentation
   - Add CSV examples and schema docs
   - Create example CSV file

9. **Phase 9** - CSV Error Handling
   - Add CSV-specific error handling
   - **Write and run tests after each improvement**
   - Exit only when all error cases are tested

10. **Phase 10** - Final CSV Validation
    - Light phase since we've been testing throughout
    - Test source switching (HF ↔ CSV)
    - Run full regression suite
    - Final smoke tests
    - **✅ IMPLEMENTATION COMPLETE!**

**Critical**: 
- Each phase has exit criteria. Do NOT move to the next phase until all tests pass!
- Do NOT start PART 2 until PART 1 is fully complete and validated!

---

## Relevant Files Reference

### Core Files (Must Read)
- `src/lmsys_query_analysis/db/loader.py` - Main loading logic
- `src/lmsys_query_analysis/db/models.py` - Data models
- `src/lmsys_query_analysis/cli/commands/data.py` - CLI commands

### Supporting Files
- `src/lmsys_query_analysis/db/chroma.py` - ChromaDB integration
- `src/lmsys_query_analysis/db/connection.py` - Database connection
- `src/lmsys_query_analysis/cli/common.py` - CLI helpers

### Test Files
- `tests/unit/db/test_loader.py` - Loader tests (comprehensive examples!)
- `tests/unit/cli/commands/test_data.py` - CLI tests
- `tests/conftest.py` - Test fixtures

### Documentation Files
- `README.md` - Main documentation
- `CLAUDE.md` - Agent guidelines
- `pyproject.toml` - Project config

---

## Notes for Future Threads

### Codebase Quality
- The existing loader is well-structured and tested
- The adapter pattern is a clean way to add new sources
- All batching, deduplication, and ChromaDB logic can be reused as-is
- The test suite has good coverage and provides excellent examples
- The CLI uses Typer with nice error handling patterns
- The codebase follows good Python practices (type hints, docstrings)

### Testing Approach ⚠️ CRITICAL
- **DO NOT save testing for the end!**
- Write tests immediately after implementing each component
- Run `uv run pytest` frequently to catch issues early
- Each phase has "Exit Criteria" - verify these before moving on
- Final validation phases should be light because you've been testing all along
- If you find yourself writing lots of code without running tests, STOP and test what you have

### Phased Implementation ⚠️ CRITICAL
- **DO HUGGING FACE FIRST (Phases 1-5), END-TO-END!**
- **DO NOT start CSV (Phases 6-10) until HF is complete and tested!**
- This is not optional - the phased approach is the implementation strategy
- PART 1 must be fully validated before starting PART 2
- Deliver value incrementally, validate at natural checkpoints

### Backwards Compatibility Requirements
- **CRITICAL**: Must maintain 100% backwards compatibility - existing commands work unchanged
- Default behavior when no flags provided: load from `"lmsys/lmsys-chat-1m"` (same as today)
- Test this explicitly in Phase 2: `lmsys load --limit 100` must work without any flags
- Backwards compatibility must be maintained throughout BOTH parts

### Key Insights
- This is a well-architected codebase that makes this feature straightforward to add with minimal risk
- The refactoring is clean and the new code should be isolated in adapters
- The backwards compatibility requirement is straightforward - just default to the current hard-coded dataset when no `--hf` flag is specified
- The phased approach delivers HF support (the primary use case) much faster
- The test-as-we-go approach will save hours of debugging later

### Implementation Flow
1. **PART 1 First**: HF support end-to-end (Phases 1-5)
2. **Validate**: All tests pass, backwards compatible, documented
3. **PART 2 Second**: CSV support end-to-end (Phases 6-10)
4. **Per Component**: Write → Test → Verify passes → Next
5. NOT: Write everything → Test everything → Fix everything
6. Confidence comes from incremental validation, not hope

