"""
Knowledge management components for AutoLabeler.

This module provides persistent storage and retrieval of labeled examples
and prompt tracking for the labeling system.
"""

from .knowledge_store import KnowledgeStore
from .prompt_manager import PromptManager, PromptRecord

__all__ = ["KnowledgeStore", "PromptManager", "PromptRecord"]
