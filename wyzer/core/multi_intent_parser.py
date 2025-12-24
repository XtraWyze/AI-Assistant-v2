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
    (r"\s*\.\s+", "sequential"),          # "open X. Do Y" (sentence boundary)
    (r"\s*,\s*", "parallel"),             # "open X, Y"
]

# Verbs that can start a new intent when appearing without separator
ACTION_VERBS = r"(?:open|launch|start|close|quit|exit|minimize|shrink|maximize|fullscreen|expand|move|send|play|pause|resume|mute|unmute|scan|switch|focus|go)"


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
      
    Also handles sentence boundaries like:
      "what's a VPN? Pause music"
    becoming:
      ["what's a VPN", "Pause music"]
    """
    text = (text or "").strip()
    if not text:
        return []
    
    # First, split on sentence boundaries (? or . followed by space and capital letter)
    # This handles cases like "what's a VPN? Pause music"
    sentence_pattern = r'(?<=[.?!])\s+(?=[A-Z])'
    sentence_parts = re.split(sentence_pattern, text)
    
    # Then split each sentence part by commas and "and"
    all_parts = []
    for sentence in sentence_parts:
        # Split by commas and "and" preserving structure
        # Pattern: split by ", and " or ", " or " and "
        parts = re.split(r'\s*,\s*(?:and\s+)?|\s+and\s+', sentence)
        all_parts.extend([p.strip() for p in parts if p and p.strip()])
    
    return all_parts


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
    # This list should match ACTION_VERBS defined above
    # Note: "full screen" (two words) is a common speech variant of "fullscreen"
    verb_patterns = [
        r"^(open|launch|start|close|quit|exit|play|pause|resume|mute|unmute|turn|scan)",
        r"^(move|send|switch|focus|go|minimize|shrink|maximize|fullscreen|expand)",
        r"^(full\s+screen)",  # "full screen it" should not get "open" inferred
        r"^(volume|set|get)",
    ]
    for pattern in verb_patterns:
        if re.match(pattern, clause, re.IGNORECASE):
            return clause
    
    # DON'T infer verbs for questions or informational queries
    # These should be handled by LLM, not treated as tool targets
    clause_lower = clause.lower()
    question_patterns = [
        r"^what(?:'?s|\s+is|\s+are|\s+does|\s+do)\s+",  # what's, what is, what are
        r"^who\s+",                                      # who is
        r"^where\s+",                                    # where is
        r"^when\s+",                                     # when is
        r"^why\s+",                                      # why is
        r"^how\s+",                                      # how do, how is
        r"^is\s+",                                       # is it
        r"^are\s+",                                      # are there
        r"^can\s+",                                      # can you
        r"^does\s+",                                     # does it
        r"^do\s+",                                       # do you
        r"^will\s+",                                     # will it
        r"^should\s+",                                   # should I
        r"^tell\s+me\s+",                                # tell me about
        r"^explain\s+",                                  # explain this
    ]
    for pattern in question_patterns:
        if re.match(pattern, clause_lower):
            return clause  # Don't infer verb for questions
    
    # Also check if it ends with a question mark
    if clause.rstrip().endswith("?"):
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
    
    # Check for explicit separators first
    for sep_pattern, _ in MULTI_INTENT_SEPARATORS:
        if re.search(sep_pattern, tl, re.IGNORECASE):
            return True
    
    # Check for implicit verb boundaries: "verb1 target1 verb2 target2"
    # E.g., "close chrome open spotify" should be detected as 2 intents
    # Pattern: look for action verb, then some words (target), then another action verb
    verb_boundary_pattern = rf"\s{ACTION_VERBS}\s"
    if re.search(verb_boundary_pattern, tl, re.IGNORECASE):
        # Make sure it's not just one verb followed by words
        # Count verb occurrences - need at least 2 for multi-intent
        verb_matches = list(re.finditer(ACTION_VERBS, tl, re.IGNORECASE))
        if len(verb_matches) >= 2:
            return True
    
    return False


def _split_by_verb_boundaries(text: str) -> List[str]:
    """
    Split text by verb boundaries when no explicit separator is found.
    
    Handles patterns like:
      "close chrome open spotify" -> ["close chrome", "open spotify"]
      "minimize chrome maximize spotify" -> ["minimize chrome", "maximize spotify"]
    """
    text = (text or "").strip()
    if not text:
        return []
    
    # Find all verb positions
    verbs = list(re.finditer(ACTION_VERBS, text, re.IGNORECASE))
    if len(verbs) < 2:
        return []
    
    clauses = []
    for i, verb_match in enumerate(verbs):
        verb_start = verb_match.start()
        
        # Find the end of this clause (start of next verb or end of text)
        if i < len(verbs) - 1:
            next_verb_start = verbs[i + 1].start()
            # Find the last word before the next verb
            clause_text = text[verb_start:next_verb_start].rstrip()
        else:
            clause_text = text[verb_start:].strip()
        
        if clause_text:
            clauses.append(clause_text)
    
    return clauses


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
        
        # Expand clauses: handle comma-separated lists AND sentence boundaries
        # This handles cases like "open spotify, chrome, and notepad"
        # AND cases like "what's a VPN? Pause music" within a single clause
        expanded_clauses = []
        for clause in clauses:
            # Always apply sentence boundary splitting (handles "question? Command" patterns)
            # For non-comma separators, also do comma expansion
            if sep_pattern != r"\s*,\s*":
                # Full expansion including commas and sentence boundaries
                comma_parts = _split_comma_separated_list(clause)
                expanded_clauses.extend(comma_parts)
            else:
                # For comma separator, just do sentence boundary splitting
                sentence_pattern = r'(?<=[.?!])\s+(?=[A-Z])'
                import re
                sentence_parts = re.split(sentence_pattern, clause)
                expanded_clauses.extend([p.strip() for p in sentence_parts if p and p.strip()])
        
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
    
    # Try verb boundary splitting as last resort (for "close chrome open spotify" style)
    # ONLY use this if there are NO explicit separators (commas, "and", "then", etc.)
    # If we have commas or "and", the separator-based splitting is more accurate
    has_explicit_separators = any(sep in raw_text.lower() for sep in [',', ' and ', ' then ', ';'])
    if not has_explicit_separators:
        verb_boundary_clauses = _split_by_verb_boundaries(raw_text)
        if len(verb_boundary_clauses) >= 2 and len(verb_boundary_clauses) <= 7:
            parsed_intents = []
            min_confidence = 1.0
            all_succeeded = True
            processed_clauses = []
            
            for clause in verb_boundary_clauses:
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


def parse_multi_intent_partial(text: str) -> Optional[Tuple[List[Dict[str, Any]], str, float]]:
    """
    Parse multi-intent command with PARTIAL success support.
    
    This handles cases like "Open Chrome, Open Chrome, and what's a VPN?" where
    some clauses are tool intents and some need LLM handling.
    
    Returns:
        (tool_intents, leftover_text, confidence) if at least one tool intent parsed
        None if no tool intents could be parsed
        
    The leftover_text contains clauses that need LLM handling (questions, etc.)
    """
    raw_text = (text or "").strip()
    if not raw_text:
        return None
    
    if not _looks_like_multi_intent(raw_text):
        return None
    
    # Try each separator in order of preference
    for sep_pattern, execution_mode in MULTI_INTENT_SEPARATORS:
        clauses = _split_by_separator(raw_text, sep_pattern)
        
        if len(clauses) < 2:
            continue
        
        # Expand clauses: handle comma-separated lists AND sentence boundaries
        expanded_clauses = []
        for clause in clauses:
            if sep_pattern != r"\s*,\s*":
                # Full expansion including commas and sentence boundaries
                comma_parts = _split_comma_separated_list(clause)
                expanded_clauses.extend(comma_parts)
            else:
                # For comma separator, just do sentence boundary splitting
                sentence_pattern = r'(?<=[.?!])\s+(?=[A-Z])'
                sentence_parts = re.split(sentence_pattern, clause)
                expanded_clauses.extend([p.strip() for p in sentence_parts if p and p.strip()])
        
        if len(expanded_clauses) > len(clauses):
            clauses = expanded_clauses
        
        if len(clauses) < 2 or len(clauses) > 7:
            continue
        
        # Parse clauses, collecting tool intents and leftover text
        parsed_intents = []
        leftover_clauses = []
        min_confidence = 1.0
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
            
            # If this clause can be handled as a tool, collect it
            if decision.mode == "tool_plan" and decision.confidence >= 0.7 and decision.intents:
                parsed_intents.extend(decision.intents)
                min_confidence = min(min_confidence, decision.confidence)
            else:
                # This clause needs LLM - add to leftover (use original clause, not enhanced)
                leftover_clauses.append(clause)
        
        # If we got at least one tool intent and have leftover, return partial result
        if parsed_intents and leftover_clauses:
            leftover_text = ", ".join(leftover_clauses)
            return (parsed_intents, leftover_text, min_confidence * 0.90)
        
        # If all clauses parsed as tools, use the regular function
        if parsed_intents and not leftover_clauses:
            return (parsed_intents, "", min_confidence * 0.95)
    
    return None

