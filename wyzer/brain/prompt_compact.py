"""
Prompt compaction utility for reducing token bloat in LLM requests.
Deterministic and safe - preserves semantic meaning while reducing size.
"""
from wyzer.core.logger import get_logger


def compact_prompt(prompt: str, max_chars: int = 8000) -> tuple[str, bool]:
    """
    Compact a prompt by keeping critical parts (system + user input) while
    omitting middle content if necessary.
    
    Strategy:
    - If prompt length <= max_chars: return as-is (no compaction)
    - Otherwise:
        - Keep first 1200 chars (system prompt + key rules)
        - Keep last 1800 chars (user input + immediate context)
        - Replace middle with "...[omitted for length]...\n"
        - Ensure final length <= max_chars
    
    Args:
        prompt: Full prompt text
        max_chars: Maximum character limit (default 8000)
    
    Returns:
        Tuple of (compacted_prompt, was_compacted)
        - compacted_prompt: The possibly-compacted prompt
        - was_compacted: Boolean indicating if compaction occurred
    """
    logger = get_logger()
    
    # No compaction needed
    if len(prompt) <= max_chars:
        return prompt, False
    
    # Calculate sections
    prefix_size = 1200
    suffix_size = 1800
    omission_marker = "\n...[omitted for length]...\n"
    
    # Extract sections
    prefix = prompt[:prefix_size]
    suffix = prompt[-suffix_size:]
    
    # Build compacted prompt
    compacted = prefix + omission_marker + suffix
    
    # If still too long, reduce from middle
    if len(compacted) > max_chars:
        # Reduce prefix/suffix to fit
        available = max_chars - len(omission_marker)
        prefix_chars = available // 2
        suffix_chars = available - prefix_chars
        
        prefix = prompt[:prefix_chars]
        suffix = prompt[-suffix_chars:] if suffix_chars > 0 else ""
        compacted = prefix + omission_marker + suffix
    
    original_len = len(prompt)
    compacted_len = len(compacted)
    logger.debug(f"Prompt compacted: {original_len} -> {compacted_len} chars")
    
    return compacted, True
