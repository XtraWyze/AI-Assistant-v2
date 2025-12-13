"""
Resolver for LocalLibrary - matches user queries to indexed targets.
"""
from typing import Dict, Any, List
from pathlib import Path
from wyzer.local_library.indexer import get_cached_index


def resolve_target(query: str) -> Dict[str, Any]:
    """
    Resolve a user query to a target (folder, file, app, url).
    
    Args:
        query: User query like "downloads", "my minecraft folder", "chrome"
        
    Returns:
        {
            "type": "folder|file|app|url|unknown",
            "path": "...",  # or "url" for url type
            "confidence": float (0-1),
            "candidates": [...]  # alternative matches
        }
    """
    # Normalize query
    query_lower = query.strip().lower()
    
    # Get cached index
    index = get_cached_index()
    
    # Try exact matches first
    result = _try_exact_match(query_lower, index)
    if result:
        return result
    
    # Try fuzzy matching
    result = _try_fuzzy_match(query_lower, index)
    if result:
        return result
    
    # Check if it looks like a URL
    if _looks_like_url(query):
        return {
            "type": "url",
            "url": query,
            "confidence": 0.9,
            "candidates": []
        }
    
    # No match found
    return {
        "type": "unknown",
        "path": "",
        "confidence": 0.0,
        "candidates": []
    }


def _try_exact_match(query: str, index: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to find an exact match in the index.
    
    Returns:
        Result dict or None if no match
    """
    # Check aliases first (highest priority)
    aliases = index.get("aliases", {})
    if query in aliases:
        alias_data = aliases[query]
        return {
            "type": alias_data.get("type", "unknown"),
            "path": alias_data.get("target", ""),
            "confidence": 1.0,
            "candidates": []
        }
    
    # Check folders
    folders = index.get("folders", {})
    if query in folders:
        return {
            "type": "folder",
            "path": folders[query],
            "confidence": 1.0,
            "candidates": []
        }
    
    # Check apps
    apps = index.get("apps", {})
    if query in apps:
        app_data = apps[query]
        return {
            "type": "app",
            "path": app_data.get("path", ""),
            "confidence": 1.0,
            "candidates": []
        }
    
    return None


def _try_fuzzy_match(query: str, index: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to find a fuzzy match in the index.
    
    Uses simple substring and word matching.
    
    Returns:
        Result dict or None if no match
    """
    candidates = []
    
    # Extract keywords from query
    keywords = _extract_keywords(query)
    
    # Search folders
    folders = index.get("folders", {})
    for folder_name, folder_path in folders.items():
        score = _match_score(keywords, folder_name)
        if score > 0.3:
            candidates.append({
                "type": "folder",
                "path": folder_path,
                "confidence": score,
                "name": folder_name
            })
    
    # Search apps
    apps = index.get("apps", {})
    for app_name, app_data in apps.items():
        score = _match_score(keywords, app_name)
        if score > 0.3:
            candidates.append({
                "type": "app",
                "path": app_data.get("path", ""),
                "confidence": score,
                "name": app_name
            })
    
    # Sort by confidence
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    if candidates:
        best = candidates[0]
        return {
            "type": best["type"],
            "path": best["path"],
            "confidence": best["confidence"],
            "candidates": candidates[1:5]  # Include up to 4 alternatives
        }
    
    return None


def _extract_keywords(query: str) -> List[str]:
    """
    Extract meaningful keywords from query.
    
    Args:
        query: User query string
        
    Returns:
        List of keywords
    """
    # Remove common filler words
    stop_words = {"my", "the", "a", "an", "open", "launch", "start", "run", "show", "go", "to", "folder", "app", "application"}
    
    words = query.split()
    keywords = [w for w in words if w not in stop_words and len(w) > 1]
    
    return keywords


def _match_score(keywords: List[str], target: str) -> float:
    """
    Calculate match score between keywords and target.
    
    Args:
        keywords: List of query keywords
        target: Target name to match against
        
    Returns:
        Score between 0 and 1
    """
    if not keywords:
        return 0.0
    
    matches = 0
    for keyword in keywords:
        if keyword in target:
            matches += 1
        elif any(keyword in part for part in target.split()):
            matches += 0.5
    
    return matches / len(keywords)


def _looks_like_url(query: str) -> bool:
    """
    Check if query looks like a URL.
    
    Args:
        query: User query
        
    Returns:
        True if looks like URL
    """
    query_lower = query.lower()
    
    # Check for URL schemes
    if query_lower.startswith(("http://", "https://", "www.")):
        return True
    
    # Check for domain-like patterns
    if "." in query and " " not in query:
        # Simple heuristic: contains dot, no spaces
        parts = query.split(".")
        if len(parts) >= 2 and all(part.isalnum() for part in parts):
            return True
    
    return False
