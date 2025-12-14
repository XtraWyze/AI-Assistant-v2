"""
Multi-intent command parser for direct orchestration without LLM.

Handles commands like:
  - "open steam and chrome"
  - "pause, then mute"
  - "close spotify; open youtube"
  - "open spotify, chrome, and notepad" (up to 7 items)
  - "turn up volume and play music"
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from wyzer.core.hybrid_router import _decide_single_clause


# Separators for multi-intent commands, in order of preference.
# Most specific/explicit first.
MULTI_INTENT_SEPARATORS = [
    (r"\s+and\s+then\s+", "sequential"),  # "open X and then do Y"
    (r"\s+then\s+", "sequential"),        # "open X then Y"
    (r"\s+and\s+", "parallel"),           # "open X and Y"
    (r"\s*;\s*", "sequential"),           # "open X; do Y"
    (r"\s*,\s*", "parallel"),             # "open X, Y"
]


def _split_by_separator(text: str, separator_pattern: str) -> List[str]:
    """Split text by separator pattern, returning non-empty clauses."""
    parts = re.split(separator_pattern, text, flags=re.IGNORECASE)
    # Strip whitespace AND trailing punctuation that might be separator artifacts
    return [
        p.strip().rstrip(",.;!?") 
        for p in parts 
        if p and p.strip()
    ]


def _split_comma_separated_list(text: str) -> List[str]:
    """
    Split comma-separated lists intelligently.
    
    Handles patterns like:
      "open spotify, chrome, and notepad"
    becoming:
      ["open spotify", "open chrome", "open notepad"]
    """
    text = (text or "").strip()
    if not text:
        return []
    
    # Split by commas and "and" preserving structure
    # Pattern: split by ", and " or ", " or " and "
    parts = re.split(r'\s*,\s*(?:and\s+)?|\s+and\s+', text)
    return [p.strip() for p in parts if p and p.strip()]


def _infer_missing_verb(clause: str, previous_clauses: List[str]) -> str:
    """
    Try to infer a missing verb for a clause based on previous clauses.
    
    For example, if previous clause is "open steam", and current is "chrome",
    infer "open chrome".
    """
    if not previous_clauses:
        return clause
    
    clause = clause.strip()
    
    # If clause already starts with a verb, don't modify it
    verb_patterns = [
        r"^(open|launch|start|close|quit|exit|play|pause|resume|mute|unmute|turn)",
        r"^(volume|set|get)",
    ]
    for pattern in verb_patterns:
        if re.match(pattern, clause, re.IGNORECASE):
            return clause
    
    # Look at the last clause that had a verb
    for prev in reversed(previous_clauses):
        prev = prev.strip().lower()
        
        # Extract verb from previous clause
        m = re.match(r"^(open|launch|start|close|quit|exit)\s+", prev)
        if m:
            verb = m.group(1)
            return f"{verb} {clause}"
        
        # Volume/media verbs
        if any(k in prev for k in ["volume", "mute", "unmute", "turn up", "turn down", "louder", "quieter"]):
            if "mute" in prev and "unmute" not in prev:
                return clause  # "mute X" doesn't apply to next item
            if "unmute" in prev:
                return clause  # "unmute X" doesn't apply to next item
            # For volume changes, don't infer verb
            return clause
        
        if any(k in prev for k in ["play", "pause", "resume"]):
            return clause  # Media controls don't really compose
    
    return clause


def _looks_like_multi_intent(text: str) -> bool:
    """Check if text looks like it contains multiple intents."""
    tl = (text or "").strip().lower()
    if not tl:
        return False
    
    # Normalize whitespace
    tl = re.sub(r"\s+", " ", tl)
    
    # Check for markers
    for sep_pattern, _ in MULTI_INTENT_SEPARATORS:
        if re.search(sep_pattern, tl, re.IGNORECASE):
            return True
    
    return False


def try_parse_multi_intent(text: str) -> Optional[Tuple[List[Dict[str, Any]], float]]:
    """
    Try to parse a multi-intent command deterministically.
    
    Returns:
        (intents, confidence) if successfully parsed, None otherwise.
        
    Intents are in format: {"tool": str, "args": dict, "continue_on_error": bool}
    
    Supports up to 7 commands with various separators:
      "open spotify, chrome, and notepad"
      "open steam and close discord"
      "pause then mute"
    """
    raw_text = (text or "").strip()
    if not raw_text:
        return None
    
    if not _looks_like_multi_intent(raw_text):
        return None
    
    # Try each separator in order of preference
    for sep_pattern, execution_mode in MULTI_INTENT_SEPARATORS:
        clauses = _split_by_separator(raw_text, sep_pattern)
        
        # Need at least 2 clauses to be a valid multi-intent
        if len(clauses) < 2:
            continue
        
        # If we split by " and " or " , ", also check if any clause contains commas
        # This handles cases like "open spotify, chrome, and notepad"
        # which splits by " and " to ["open spotify, chrome", "notepad"]
        # then we need to further split "open spotify, chrome" by commas
        expanded_clauses = []
        for clause in clauses:
            # Check if this clause contains a comma-separated list
            if ',' in clause and sep_pattern != r"\s*,\s*":
                # Split by comma and infer verbs
                comma_parts = _split_comma_separated_list(clause)
                expanded_clauses.extend(comma_parts)
            else:
                expanded_clauses.append(clause)
        
        # Use expanded clauses if we got more of them
        if len(expanded_clauses) > len(clauses):
            clauses = expanded_clauses
        
        # Need at least 2 clauses to be a valid multi-intent, max 7
        if len(clauses) < 2 or len(clauses) > 7:
            continue
        
        # Try to parse each clause deterministically
        parsed_intents = []
        min_confidence = 1.0
        all_succeeded = True
        processed_clauses = []
        
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            
            # Try to infer missing verb from previous clauses
            enhanced_clause = _infer_missing_verb(clause, processed_clauses)
            processed_clauses.append(enhanced_clause)
            
            # Use hybrid_router's single-clause decision
            decision = _decide_single_clause(enhanced_clause)
            
            # If any clause can't be handled deterministically, bail out
            if decision.mode == "llm" or decision.confidence < 0.7:
                all_succeeded = False
                break
            
            # Collect the intents
            if decision.intents:
                parsed_intents.extend(decision.intents)
            
            # Track minimum confidence across all clauses
            min_confidence = min(min_confidence, decision.confidence)
        
        # If all clauses parsed successfully, return the result
        if all_succeeded and parsed_intents:
            return (parsed_intents, min_confidence * 0.95)  # Slight confidence penalty for multi-intent
    
    # No successful parse
    return None


def parse_multi_intent_with_fallback(text: str) -> Optional[Tuple[List[Dict[str, Any]], float]]:
    """
    Parse multi-intent command with fallback to LLM if not fully certain.
    
    Returns:
        (intents, confidence) if successfully parsed, None if should use LLM.
    """
    result = try_parse_multi_intent(text)
    
    if result is None:
        return None
    
    intents, confidence = result
    
    # Only return if we're confident enough (>= 75%)
    # This is lower than single-intent (90%+) because composing multiple
    # deterministic parsers is still fairly reliable
    if confidence >= 0.75:
        return (intents, confidence)
    
    # Otherwise, let LLM handle it
    return None

