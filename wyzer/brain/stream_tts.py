"""
Stream-to-TTS chunk buffering policy.

Buffers incoming LLM text tokens and emits TTS segments when:
1. A sentence boundary is detected (. ! ?)
2. Buffer exceeds a character threshold
3. A paragraph boundary (newline) occurs
4. Final flush at end of stream

This ensures stable TTS output without tiny annoying fragments.
"""
import re
from typing import Iterator, Callable, Optional, Tuple
from wyzer.core.logger import get_logger

# Sentence-ending punctuation pattern
_SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')

# Paragraph boundary pattern (double newline or single newline followed by content)
_PARAGRAPH_RE = re.compile(r'\n\s*\n|\n(?=[A-Z])')


class ChunkBuffer:
    """
    Buffer for accumulating LLM tokens and emitting TTS-ready segments.
    
    Usage:
        buffer = ChunkBuffer(min_chars=150)
        for token in stream:
            segment = buffer.add(token)
            if segment:
                speak(segment)
        final = buffer.flush()
        if final:
            speak(final)
    """
    
    def __init__(self, min_chars: int = 150):
        """
        Initialize the chunk buffer.
        
        Args:
            min_chars: Minimum buffer size before emitting on non-punctuation triggers
        """
        self._buffer = ""
        self._min_chars = min_chars
        self._logger = get_logger()
    
    def add(self, token: str) -> Optional[str]:
        """
        Add a token to the buffer and return a segment if ready.
        
        Args:
            token: Text token from LLM stream
            
        Returns:
            A TTS segment if ready to speak, None otherwise
        """
        if not token:
            return None
        
        self._buffer += token
        
        # Check for sentence boundary
        match = _SENTENCE_END_RE.search(self._buffer)
        if match:
            # Find the end of the sentence (include the punctuation)
            end_pos = match.end()
            segment = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[end_pos:].lstrip()
            
            if segment:
                self._logger.debug(f"[STREAM_TTS] Sentence segment (len={len(segment)})")
                return segment
        
        # Check for paragraph boundary
        para_match = _PARAGRAPH_RE.search(self._buffer)
        if para_match:
            end_pos = para_match.start()
            segment = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[para_match.end():].lstrip()
            
            if segment:
                self._logger.debug(f"[STREAM_TTS] Paragraph segment (len={len(segment)})")
                return segment
        
        # Check buffer size threshold (emit if too long without punctuation)
        if len(self._buffer) >= self._min_chars:
            # Try to break at a word boundary
            last_space = self._buffer.rfind(' ', 0, self._min_chars)
            if last_space > self._min_chars // 2:
                # Found a reasonable word boundary
                segment = self._buffer[:last_space].strip()
                self._buffer = self._buffer[last_space:].lstrip()
            else:
                # No good word boundary, emit what we have
                segment = self._buffer.strip()
                self._buffer = ""
            
            if segment:
                self._logger.debug(f"[STREAM_TTS] Buffer overflow segment (len={len(segment)})")
                return segment
        
        return None
    
    def flush(self) -> Optional[str]:
        """
        Flush any remaining buffered text.
        
        Returns:
            Remaining text if any, None otherwise
        """
        segment = self._buffer.strip()
        self._buffer = ""
        
        if segment:
            self._logger.debug(f"[STREAM_TTS] Final flush segment (len={len(segment)})")
            return segment
        
        return None
    
    def clear(self) -> None:
        """Clear the buffer without emitting."""
        self._buffer = ""
    
    @property
    def pending(self) -> str:
        """Return current buffered text (for debugging)."""
        return self._buffer


def stream_to_tts_segments(
    token_stream: Iterator[str],
    min_chars: int = 150,
    cancel_check: Optional[Callable[[], bool]] = None
) -> Iterator[Tuple[str, bool]]:
    """
    Convert a token stream into TTS-ready segments.
    
    Args:
        token_stream: Iterator yielding text tokens from LLM
        min_chars: Minimum buffer size before emitting on non-punctuation triggers
        cancel_check: Optional callable that returns True if streaming should stop
        
    Yields:
        Tuples of (segment_text, is_final) where is_final is True for the last segment
    """
    logger = get_logger()
    buffer = ChunkBuffer(min_chars=min_chars)
    
    logger.debug("[STREAM_TTS] Starting stream-to-TTS conversion")
    
    try:
        for token in token_stream:
            # Check cancellation
            if cancel_check and cancel_check():
                logger.debug("[STREAM_TTS] Cancelled during stream")
                buffer.clear()
                return
            
            segment = buffer.add(token)
            if segment:
                yield (segment, False)
        
        # Final flush
        final_segment = buffer.flush()
        if final_segment:
            yield (final_segment, True)
        else:
            # No final segment, but mark the last yielded segment as final
            # (This case is handled by the consumer)
            pass
            
    except Exception as e:
        logger.error(f"[STREAM_TTS] Error during streaming: {e}")
        # Let caller handle the exception
        raise
    finally:
        logger.debug("[STREAM_TTS] Stream-to-TTS conversion ended")


def accumulate_full_reply(
    token_stream: Iterator[str],
    on_segment: Callable[[str], None],
    min_chars: int = 150,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Consume a token stream, emit TTS segments via callback, and return full reply.
    
    This is the main integration point: it processes tokens, emits segments for
    TTS as they become ready, and accumulates the full reply for memory storage.
    
    Args:
        token_stream: Iterator yielding text tokens from LLM
        on_segment: Callback to receive each TTS segment
        min_chars: Minimum buffer size before emitting on non-punctuation triggers
        cancel_check: Optional callable that returns True if streaming should stop
        
    Returns:
        The complete accumulated reply text
    """
    logger = get_logger()
    full_reply = ""
    buffer = ChunkBuffer(min_chars=min_chars)
    
    logger.debug("[STREAM_TTS] LLM stream started")
    
    try:
        for token in token_stream:
            # Check cancellation
            if cancel_check and cancel_check():
                logger.debug("[STREAM_TTS] Cancelled during stream")
                # Return what we have so far
                return full_reply
            
            # Accumulate for full reply
            full_reply += token
            
            # Check for TTS segment
            segment = buffer.add(token)
            if segment:
                logger.debug(f"[STREAM_TTS] TTS segment enqueued (len={len(segment)})")
                on_segment(segment)
        
        # Final flush
        final_segment = buffer.flush()
        if final_segment:
            logger.debug(f"[STREAM_TTS] TTS segment enqueued (len={len(final_segment)})")
            on_segment(final_segment)
        
        logger.debug("[STREAM_TTS] LLM stream ended")
        return full_reply
        
    except Exception as e:
        logger.error(f"[STREAM_TTS] Streaming failed, falling back: {e}")
        raise
