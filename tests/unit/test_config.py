"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from lmsys_query_analysis.config import (
    RunnerConfig,
    load_config_from_yaml,
    save_config_to_yaml,
)


def test_runner_config_defaults():
    """Test that RunnerConfig has sensible defaults."""
    config = RunnerConfig()

    assert config.query_limit == 1000
    assert config.embedding_provider == "openai"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.n_clusters == 50
    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-4o-mini"
    assert config.log_level == "INFO"
    assert config.enable_hierarchy is True
    assert config.hierarchy_levels == 3


def test_runner_config_validation_n_clusters_exceeds_limit():
    """Test that n_clusters validation works."""
    # Should raise error when n_clusters > query_limit
    with pytest.raises(ValidationError, match="cannot exceed query_limit"):
        RunnerConfig(query_limit=100, n_clusters=200)


def test_runner_config_validation_n_clusters_valid():
    """Test that valid n_clusters passes validation."""
    config = RunnerConfig(query_limit=1000, n_clusters=50)
    assert config.n_clusters == 50
    assert config.query_limit == 1000


def test_runner_config_path_resolution_relative():
    """Test that relative paths are resolved to absolute paths."""
    config = RunnerConfig(db_path="./test.db", chroma_path="./chroma")

    # Should be converted to absolute paths
    assert Path(config.db_path).is_absolute()
    assert Path(config.chroma_path).is_absolute()
    assert config.db_path.endswith("test.db")
    assert config.chroma_path.endswith("chroma")


def test_runner_config_path_resolution_absolute():
    """Test that absolute paths are preserved."""
    abs_db = "/tmp/test.db"
    abs_chroma = "/tmp/chroma"

    config = RunnerConfig(db_path=abs_db, chroma_path=abs_chroma)

    assert config.db_path == abs_db
    assert config.chroma_path == abs_chroma


def test_runner_config_path_resolution_none():
    """Test that None paths remain None."""
    config = RunnerConfig(db_path=None, chroma_path=None)

    assert config.db_path is None
    assert config.chroma_path is None


def test_runner_config_embedding_providers():
    """Test different embedding provider configurations."""
    # OpenAI
    config_openai = RunnerConfig(
        embedding_provider="openai", embedding_model="text-embedding-3-large"
    )
    assert config_openai.embedding_provider == "openai"
    assert config_openai.embedding_model == "text-embedding-3-large"

    # Cohere
    config_cohere = RunnerConfig(embedding_provider="cohere", embedding_model="embed-v4.0")
    assert config_cohere.embedding_provider == "cohere"

    # Sentence transformers
    config_st = RunnerConfig(
        embedding_provider="sentence-transformers", embedding_model="all-MiniLM-L6-v2"
    )
    assert config_st.embedding_provider == "sentence-transformers"


def test_runner_config_invalid_embedding_provider():
    """Test that invalid embedding provider raises error."""
    with pytest.raises(ValidationError):
        RunnerConfig(embedding_provider="invalid-provider")


def test_runner_config_llm_providers():
    """Test different LLM provider configurations."""
    # OpenAI
    config_openai = RunnerConfig(llm_provider="openai", llm_model="gpt-4o")
    assert config_openai.llm_provider == "openai"
    assert config_openai.llm_model == "gpt-4o"

    # Anthropic
    config_anthropic = RunnerConfig(
        llm_provider="anthropic", llm_model="claude-3-5-sonnet-20241022"
    )
    assert config_anthropic.llm_provider == "anthropic"

    # Groq
    config_groq = RunnerConfig(llm_provider="groq", llm_model="llama-3.1-70b-versatile")
    assert config_groq.llm_provider == "groq"


def test_runner_config_invalid_llm_provider():
    """Test that invalid LLM provider raises error."""
    with pytest.raises(ValidationError):
        RunnerConfig(llm_provider="invalid-llm")


def test_runner_config_log_levels():
    """Test different log level configurations."""
    for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        config = RunnerConfig(log_level=level)
        assert config.log_level == level


def test_runner_config_invalid_log_level():
    """Test that invalid log level raises error."""
    with pytest.raises(ValidationError):
        RunnerConfig(log_level="INVALID")


def test_runner_config_hierarchy_settings():
    """Test hierarchy configuration options."""
    config = RunnerConfig(
        enable_hierarchy=True,
        hierarchy_levels=5,
        merge_ratio=0.2,
        neighborhood_size=50,
        concurrency=16,
        rpm=500,
    )

    assert config.enable_hierarchy is True
    assert config.hierarchy_levels == 5
    assert config.merge_ratio == 0.2
    assert config.neighborhood_size == 50
    assert config.concurrency == 16
    assert config.rpm == 500


def test_runner_config_merge_ratio_validation():
    """Test that merge_ratio must be between 0 and 1."""
    # Valid
    config = RunnerConfig(merge_ratio=0.5)
    assert config.merge_ratio == 0.5

    # Invalid - too large
    with pytest.raises(ValidationError):
        RunnerConfig(merge_ratio=1.5)

    # Invalid - negative
    with pytest.raises(ValidationError):
        RunnerConfig(merge_ratio=-0.1)


def test_runner_config_positive_integer_fields():
    """Test that fields requiring positive integers are validated."""
    # Valid
    config = RunnerConfig(
        query_limit=100,
        n_clusters=10,
        chunk_size=1000,
        mb_batch_size=512,
    )
    assert config.query_limit == 100

    # Invalid - zero
    with pytest.raises(ValidationError):
        RunnerConfig(query_limit=0)

    # Invalid - negative
    with pytest.raises(ValidationError):
        RunnerConfig(n_clusters=-5)


def test_load_config_from_yaml():
    """Test loading configuration from YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
query_limit: 5000
n_clusters: 100
embedding_provider: openai
embedding_model: text-embedding-3-small
llm_provider: anthropic
llm_model: claude-3-5-sonnet-20241022
enable_hierarchy: true
hierarchy_levels: 4
log_level: DEBUG
""")
        yaml_path = f.name

    try:
        config = load_config_from_yaml(yaml_path)

        assert config.query_limit == 5000
        assert config.n_clusters == 100
        assert config.embedding_provider == "openai"
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-5-sonnet-20241022"
        assert config.hierarchy_levels == 4
        assert config.log_level == "DEBUG"
    finally:
        Path(yaml_path).unlink()


def test_save_and_load_config_roundtrip():
    """Test saving and loading config maintains data integrity."""
    original_config = RunnerConfig(
        query_limit=2000,
        n_clusters=150,
        embedding_provider="cohere",
        embedding_model="embed-v4.0",
        llm_provider="groq",
        llm_model="llama-3.1-70b-versatile",
        hierarchy_levels=5,
        log_level="WARNING",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name

    try:
        # Save
        save_config_to_yaml(original_config, yaml_path)

        # Load
        loaded_config = load_config_from_yaml(yaml_path)

        # Verify all key fields match
        assert loaded_config.query_limit == original_config.query_limit
        assert loaded_config.n_clusters == original_config.n_clusters
        assert loaded_config.embedding_provider == original_config.embedding_provider
        assert loaded_config.embedding_model == original_config.embedding_model
        assert loaded_config.llm_provider == original_config.llm_provider
        assert loaded_config.llm_model == original_config.llm_model
        assert loaded_config.hierarchy_levels == original_config.hierarchy_levels
        assert loaded_config.log_level == original_config.log_level
    finally:
        Path(yaml_path).unlink()


def test_runner_config_extra_fields_forbidden():
    """Test that extra fields are rejected."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunnerConfig(unknown_field="value")


def test_runner_config_batch_size_settings():
    """Test various batch size configurations."""
    config = RunnerConfig(
        embedding_batch_size=200,
        embed_batch_size=50,
        mb_batch_size=2048,
        chunk_size=10000,
    )

    assert config.embedding_batch_size == 200
    assert config.embed_batch_size == 50
    assert config.mb_batch_size == 2048
    assert config.chunk_size == 10000


def test_runner_config_cleanup_setting():
    """Test cleanup_temp configuration."""
    config_cleanup = RunnerConfig(cleanup_temp=True)
    assert config_cleanup.cleanup_temp is True

    config_no_cleanup = RunnerConfig(cleanup_temp=False)
    assert config_no_cleanup.cleanup_temp is False


def test_runner_config_skip_existing_streaming():
    """Test data loading configuration options."""
    config = RunnerConfig(
        skip_existing=False,
        use_streaming=True,
    )

    assert config.skip_existing is False
    assert config.use_streaming is True
