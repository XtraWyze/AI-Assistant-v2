"""
Sentence-gated TTS streaming buffer.

This module provides a smarter chunking strategy for streaming LLM output to TTS:
- Buffers text until safe sentence boundaries
- Time-based flushing for long pauses
- Filters out code blocks, JSON payloads, and tool markers
- Handles abbreviations to avoid mid-sentence splits

The goal is "best UX" speech: start speaking quickly (~0.6-1.2s)
but never speak half-sentences when avoidable.
"""

import re
import time
from typing import List, Optional, Tuple

from wyzer.core.logger import get_logger

# Common abbreviations that end with period but are NOT sentence boundaries
ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "e.g.", "i.e.", "etc.", "vs.", "viz.", "approx.",
    "inc.", "ltd.", "co.", "corp.", "llc.",
    "u.s.", "u.k.", "u.n.", "n.a.", "d.c.",
    "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
    "st.", "rd.", "ave.", "blvd.", "apt.", "no.",
    "ph.d.", "m.d.", "b.a.", "m.a.", "b.s.", "m.s.",
    "a.m.", "p.m.",
    "ft.", "in.", "lb.", "oz.", "pt.", "qt.", "gal.",
    "fig.", "eq.", "ref.", "vol.", "pg.", "pp.",
})

# Markers that indicate tool/payload content - should not be spoken
# Markers that indicate tool/payload content - should not be spoken
# Note: Code fences (```) are handled separately via stripping, not here
PAYLOAD_MARKERS = (
    "[TOOLS]", "[TOOL", "args={", "Pool result", "[INTENT",
    "[ACTION]", "[DEBUG]", "[ERROR]", "tool_call",
)

# URL pattern to avoid splitting mid-URL
URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+|'
    r'www\.[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)


def now_ms() -> int:
    """Get current time in milliseconds (monotonic)."""
    return int(time.monotonic() * 1000)


class TTSStreamBuffer:
    """
    Sentence-gated buffer for streaming LLM output to TTS.
    
    Accumulates streamed tokens and flushes chunks to TTS only when safe:
    1. Boundary flush: text ends at sentence punctuation and meets min size
    2. Timeout flush: max_wait_ms elapsed since last flush
    3. End flush: stream ended, flush remaining text
    
    Safety filters:
    - Skips content inside fenced code blocks ```...```
    - Skips JSON/tool payloads
    - Handles abbreviations (doesn't split on "e.g.", "Mr.", etc.)
    """
    
    # Default sentence boundary punctuation
    DEFAULT_BOUNDARIES = ".!?:"
    
    def __init__(
        self,
        min_chars: int = 60,
        min_words: int = 10,
        max_wait_ms: int = 900,
        boundaries: str = DEFAULT_BOUNDARIES,
    ):
        """
        Initialize the TTS stream buffer.
        
        Args:
            min_chars: Minimum characters before boundary flush (default 60)
            min_words: Minimum words before boundary flush (default 10)
            max_wait_ms: Maximum milliseconds to wait before timeout flush (default 900)
            boundaries: Punctuation characters that mark sentence boundaries
        """
        self.min_chars = min_chars
        self.min_words = min_words
        self.max_wait_ms = max_wait_ms
        self.boundaries = set(boundaries)
        
        self._buffer = ""
        self._last_flush_ms = now_ms()
        self._in_code_fence = False
        self._code_fence_count = 0  # Track nested fences
        
        self._logger = get_logger()
    
    def add_text(self, text: str, now_ms_val: Optional[int] = None) -> List[str]:
        """
        Add text to the buffer and return any chunks ready for TTS.
        
        Args:
            text: Text fragment (token) from LLM stream
            now_ms_val: Current time in milliseconds (for testing), or None to use real time
            
        Returns:
            List of text chunks ready for TTS synthesis (may be empty)
        """
        if not text:
            return []
        
        if now_ms_val is None:
            now_ms_val = now_ms()
        
        self._buffer += text
        
        # Track code fence state
        self._update_code_fence_state(text)
        
        # If a code fence just closed, strip the code block from buffer
        # This removes ```...``` content so it won't be spoken
        if not self._in_code_fence and self._code_fence_count > 0 and self._code_fence_count % 2 == 0:
            self._strip_code_blocks()
        
        # If inside code fence, don't flush
        if self._in_code_fence:
            self._logger.debug(f"[TTS_BUFFER] Inside code fence, buffering ({len(self._buffer)} chars)")
            return []
        
        chunks = []
        
        # Try boundary flush
        boundary_chunk = self._try_boundary_flush()
        if boundary_chunk:
            chunks.append(boundary_chunk)
            self._last_flush_ms = now_ms_val
        
        # Check timeout flush if no boundary flush happened
        if not chunks:
            elapsed = now_ms_val - self._last_flush_ms
            if elapsed >= self.max_wait_ms:
                timeout_chunk = self._try_timeout_flush()
                if timeout_chunk:
                    chunks.append(timeout_chunk)
                    self._last_flush_ms = now_ms_val
        
        return chunks
    
    def _strip_code_blocks(self) -> None:
        """Remove complete code blocks (```...```) from the buffer."""
        import re
        # Match complete fenced code blocks and remove them
        # Keep content before and after, just remove the code blocks themselves
        pattern = r'```[\s\S]*?```'
        original = self._buffer
        self._buffer = re.sub(pattern, ' ', self._buffer)
        # Clean up multiple spaces
        self._buffer = re.sub(r' +', ' ', self._buffer)
        if self._buffer != original:
            self._logger.debug(f"[TTS_BUFFER] Stripped code block, buffer now {len(self._buffer)} chars")
    
    def flush_final(self) -> List[str]:
        """
        Flush any remaining text at end of stream.
        
        Returns:
            List containing the final chunk (or empty if nothing remains)
        """
        # Strip any remaining code blocks first
        self._strip_code_blocks()
        
        text = self._buffer.strip()
        self._buffer = ""
        
        if not text:
            return []
        
        # Check if it's a payload that shouldn't be spoken
        if self._looks_like_payload(text):
            self._logger.debug(f"[TTS_BUFFER] Final flush skipped (payload): {len(text)} chars")
            return []
        
        self._logger.debug(f"[TTS_BUFFER] Final flush: {len(text)} chars, {len(text.split())} words")
        return [text]
    
    def reset(self) -> None:
        """Reset buffer state for a new stream."""
        self._buffer = ""
        self._last_flush_ms = now_ms()
        self._in_code_fence = False
        self._code_fence_count = 0
    
    def _update_code_fence_state(self, text: str) -> None:
        """
        Update code fence tracking state based on full buffer content.
        
        Scans the complete buffer for ``` markers since tokens can split markers.
        """
        # Count total fence markers in the FULL buffer (not just new text)
        # This handles cases where ``` is split across tokens
        fence_count = self._buffer.count("```")
        self._code_fence_count = fence_count
        # Odd count = inside fence, even count = outside
        self._in_code_fence = (fence_count % 2) == 1
    
    def _in_code_fence_check(self) -> bool:
        """Check if currently inside a code fence."""
        return self._in_code_fence
    
    def _try_boundary_flush(self) -> Optional[str]:
        """
        Try to flush at a safe sentence boundary.
        
        Returns the chunk to flush, or None if no safe boundary found.
        """
        if not self._buffer:
            return None
        
        # Find the rightmost safe boundary
        boundary_pos = self._find_safe_boundary()
        if boundary_pos < 0:
            return None
        
        # Extract candidate chunk (including boundary punctuation)
        candidate = self._buffer[:boundary_pos + 1].strip()
        
        # Check minimum size requirements
        word_count = len(candidate.split())
        char_count = len(candidate)
        
        if char_count < self.min_chars and word_count < self.min_words:
            return None
        
        # Check if it's a payload that shouldn't be spoken
        if self._looks_like_payload(candidate):
            self._logger.debug(f"[TTS_BUFFER] Boundary flush skipped (payload): {char_count} chars")
            # Remove the payload from buffer but don't speak it
            self._buffer = self._buffer[boundary_pos + 1:].lstrip()
            return None
        
        # Commit the flush
        self._buffer = self._buffer[boundary_pos + 1:].lstrip()
        
        self._logger.debug(
            f"[TTS_BUFFER] Boundary flush: {char_count} chars, {word_count} words, "
            f"ends with '{candidate[-1] if candidate else ''}'"
        )
        
        return candidate
    
    def _try_timeout_flush(self) -> Optional[str]:
        """
        Try to flush due to timeout.
        
        Prefers flushing at a boundary if one exists; otherwise flushes at
        word boundary if buffer is long enough.
        
        Returns the chunk to flush, or None if not ready.
        """
        if not self._buffer.strip():
            return None
        
        # First try: find any boundary in current buffer
        boundary_pos = self._find_safe_boundary()
        if boundary_pos >= 0:
            candidate = self._buffer[:boundary_pos + 1].strip()
            if candidate:
                if self._looks_like_payload(candidate):
                    self._logger.debug(f"[TTS_BUFFER] Timeout boundary flush skipped (payload)")
                    self._buffer = self._buffer[boundary_pos + 1:].lstrip()
                    return None
                
                self._buffer = self._buffer[boundary_pos + 1:].lstrip()
                self._logger.debug(
                    f"[TTS_BUFFER] Timeout flush (boundary): {len(candidate)} chars, "
                    f"{len(candidate.split())} words"
                )
                return candidate
        
        # Second try: flush at word boundary if long enough
        if len(self._buffer) >= self.min_chars:
            # Find last whitespace position
            last_space = self._buffer.rfind(' ')
            if last_space > self.min_chars // 2:
                candidate = self._buffer[:last_space].strip()
                if candidate:
                    if self._looks_like_payload(candidate):
                        self._logger.debug(f"[TTS_BUFFER] Timeout word-boundary flush skipped (payload)")
                        self._buffer = self._buffer[last_space + 1:]
                        return None
                    
                    self._buffer = self._buffer[last_space + 1:]
                    self._logger.debug(
                        f"[TTS_BUFFER] Timeout flush (word boundary): {len(candidate)} chars, "
                        f"{len(candidate.split())} words"
                    )
                    return candidate
        
        # Not ready to flush yet
        return None
    
    def _find_safe_boundary(self) -> int:
        """
        Find the rightmost safe sentence boundary in the buffer.
        
        A "safe" boundary is punctuation that:
        1. Is in self.boundaries
        2. Is followed by whitespace or end of buffer
        3. Is NOT part of an abbreviation
        4. Is NOT inside a URL
        
        Returns:
            Position of boundary character, or -1 if none found
        """
        if not self._buffer:
            return -1
        
        # Get URL ranges to exclude
        url_ranges = self._get_url_ranges()
        
        # Search from right to left for boundary
        best_pos = -1
        
        for i in range(len(self._buffer) - 1, -1, -1):
            char = self._buffer[i]
            
            if char not in self.boundaries:
                continue
            
            # Check if followed by whitespace or end
            if i < len(self._buffer) - 1:
                next_char = self._buffer[i + 1]
                if not next_char.isspace():
                    continue
            
            # Check if inside URL
            if self._pos_in_ranges(i, url_ranges):
                continue
            
            # Check if part of abbreviation (only for period)
            if char == '.' and self._is_abbrev_boundary(i):
                continue
            
            best_pos = i
            break
        
        return best_pos
    
    def _is_abbrev_boundary(self, pos: int) -> bool:
        """
        Check if the period at `pos` is part of an abbreviation.
        
        Uses lookback to extract potential abbreviation and checks against list.
        
        Args:
            pos: Position of the period in self._buffer
            
        Returns:
            True if this period is part of an abbreviation
        """
        if pos < 1:
            return False
        
        # Lookback up to 8 characters to find the token
        lookback_start = max(0, pos - 8)
        lookback = self._buffer[lookback_start:pos + 1]
        
        # Find the start of the word (last whitespace before period)
        word_start = lookback.rfind(' ')
        if word_start >= 0:
            word = lookback[word_start + 1:]
        else:
            word = lookback
        
        # Normalize and check
        word_lower = word.lower().strip()
        
        return word_lower in ABBREVIATIONS
    
    def _get_url_ranges(self) -> List[Tuple[int, int]]:
        """Get list of (start, end) positions for URLs in buffer."""
        ranges = []
        for match in URL_PATTERN.finditer(self._buffer):
            ranges.append((match.start(), match.end()))
        return ranges
    
    def _pos_in_ranges(self, pos: int, ranges: List[Tuple[int, int]]) -> bool:
        """Check if position is inside any of the given ranges."""
        for start, end in ranges:
            if start <= pos < end:
                return True
        return False
    
    def _looks_like_payload(self, text: str) -> bool:
        """
        Check if text looks like a tool payload or code that shouldn't be spoken.
        
        Detects:
        - Known payload markers
        - JSON-like content (high brace/bracket density)
        - Code fence content
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a payload
        """
        if not text:
            return False
        
        text_stripped = text.strip()
        
        # Check for known payload markers
        for marker in PAYLOAD_MARKERS:
            if marker in text:
                return True
        
        # Check for JSON-like structure
        if self._is_json_like(text_stripped):
            return True
        
        return False
    
    def _is_json_like(self, text: str) -> bool:
        """
        Check if text looks like JSON based on brace/bracket density.
        
        Returns True if:
        - Starts with '{' or '[' and ends with '}' or ']'
        - OR has high density of braces/brackets (> 10% of chars)
        """
        if not text:
            return False
        
        text = text.strip()
        
        # Check bracketed structure
        if (text.startswith('{') and text.endswith('}')) or \
           (text.startswith('[') and text.endswith(']')):
            return True
        
        # Check brace density
        brace_chars = sum(1 for c in text if c in '{}[]')
        if len(text) > 10 and brace_chars / len(text) > 0.10:
            return True
        
        return False


def create_buffer_from_config() -> TTSStreamBuffer:
    """
    Create a TTSStreamBuffer with settings from Config.
    
    Reads:
    - TTS_STREAM_MIN_CHARS (default 60)
    - TTS_STREAM_MIN_WORDS (default 10)
    - TTS_STREAM_MAX_WAIT_MS (default 900)
    - TTS_STREAM_BOUNDARIES (default ".!?:")
    """
    from wyzer.core.config import Config
    
    min_chars = getattr(Config, 'TTS_STREAM_MIN_CHARS', 60)
    min_words = getattr(Config, 'TTS_STREAM_MIN_WORDS', 10)
    max_wait_ms = getattr(Config, 'TTS_STREAM_MAX_WAIT_MS', 900)
    boundaries = getattr(Config, 'TTS_STREAM_BOUNDARIES', '.!?:')
    
    return TTSStreamBuffer(
        min_chars=min_chars,
        min_words=min_words,
        max_wait_ms=max_wait_ms,
        boundaries=boundaries,
    )
