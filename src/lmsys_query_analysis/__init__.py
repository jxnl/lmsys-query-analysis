"""LMSYS Query Analysis Package.

A comprehensive toolkit for analyzing the LMSYS-1M dataset with clustering,
hierarchical organization, and LLM-powered summarization.
"""

from .config import RunnerConfig, load_config_from_yaml, save_config_to_yaml
from .runner import AnalysisRunner, BaseCluster, run_analysis

__all__ = [
    # Runner API
    "AnalysisRunner",
    "run_analysis",
    "BaseCluster",
    # Configuration
    "RunnerConfig",
    "load_config_from_yaml",
    "save_config_to_yaml",
]

__version__ = "0.1.0"


def hello() -> str:
    """Legacy hello function for backwards compatibility."""
    return "Hello from lmsys-query-analysis!"
