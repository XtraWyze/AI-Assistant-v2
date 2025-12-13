"""
LocalLibrary module for indexing and resolving user references to files, folders, and apps.
"""
from wyzer.local_library.indexer import refresh_index
from wyzer.local_library.resolver import resolve_target

__all__ = ["refresh_index", "resolve_target"]
