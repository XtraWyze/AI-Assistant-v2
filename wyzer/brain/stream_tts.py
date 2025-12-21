"""
Streaming TTS chunk policy and accumulation.

This module provides utilities for streaming LLM tokens directly to TTS:
1. ChunkBuffer - Buffers tokens and emits on sentence/paragraph boundaries
2. accumulate_full_reply - Processes token stream and feeds TTS queue

The goal is to reduce perceived latency by starting TTS synthesis before
the full LLM response is complete.
"""
import re
from typing import Iterator, Callable, Optional

from wyzer.core.logger import get_logger


class ChunkBuffer:
    """
    Buffer that accumulates streaming tokens and emits complete segments.
    
    Emission triggers:
    1. Sentence boundary (. ! ? followed by space or end)
    2. Paragraph boundary (double newline)
    3. Buffer overflow (exceeds min_chars threshold)
    
    First-emit optimization:
    - For the FIRST segment, emit sooner (first_emit_chars or first punctuation)
    - This reduces perceived latency for voice assistants
    
    This enables TTS to start speaking while the LLM is still generating.
    """
    
    # Sentence-ending punctuation followed by space or end
    SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')
    
    # Paragraph boundary (double newline)
    PARAGRAPH_RE = re.compile(r'\n\n+')
    
    # First-emit punctuation (any punctuation that's a natural pause point)
    FIRST_EMIT_PUNCT_RE = re.compile(r'[.!?;:]')
    
    def __init__(self, min_chars: int = 150, first_emit_chars: int = 24):
        """
        Initialize chunk buffer.
        
        Args:
            min_chars: Minimum characters before overflow emission.
                       Lower = faster TTS start, but more TTS calls.
                       Higher = fewer TTS calls, but more latency.
            first_emit_chars: Characters threshold for first emit optimization.
                              Emit first segment sooner for lower perceived latency.
        """
        self.min_chars = min_chars
        self.first_emit_chars = first_emit_chars
        self._buffer = ""
        self._first_emitted = False
        self._logger = get_logger()
    
    def add(self, token: str) -> Optional[str]:
        """
        Add a token to the buffer.
        
        Returns a complete segment if an emission trigger is detected,
        otherwise returns None.
        
        Args:
            token: Token/chunk from streaming LLM
            
        Returns:
            Complete segment to send to TTS, or None if still buffering
        """
        if not token:
            return None
        
        self._buffer += token
        
        # First-emit optimization: emit sooner for the first segment
        if not self._first_emitted:
            segment = self._try_first_emit()
            if segment:
                return segment
        
        # Check for paragraph boundary (highest priority)
        para_match = self.PARAGRAPH_RE.search(self._buffer)
        if para_match:
            # Emit everything before the paragraph break
            segment = self._buffer[:para_match.start()].strip()
            self._buffer = self._buffer[para_match.end():]
            if segment:
                self._first_emitted = True
                self._logger.debug(f"[STREAM_TTS] Paragraph emit: {len(segment)} chars")
                return segment
        
        # Check for sentence boundary
        # Look for sentence end followed by space (not at the very end, in case more comes)
        sent_match = self.SENTENCE_END_RE.search(self._buffer)
        if sent_match:
            # Check if there's content after the sentence end
            end_pos = sent_match.end()
            if end_pos < len(self._buffer) or len(self._buffer) > self.min_chars // 2:
                # Emit the complete sentence
                segment = self._buffer[:end_pos].strip()
                self._buffer = self._buffer[end_pos:].lstrip()
                if segment:
                    self._first_emitted = True
                    self._logger.debug(f"[STREAM_TTS] Sentence emit: {len(segment)} chars")
                    return segment
        
        # Check for buffer overflow (emit at word boundary if possible)
        if len(self._buffer) >= self.min_chars:
            # Find last word boundary
            last_space = self._buffer.rfind(' ', 0, len(self._buffer) - 1)
            if last_space > self.min_chars // 2:
                segment = self._buffer[:last_space].strip()
                self._buffer = self._buffer[last_space + 1:]
                if segment:
                    self._first_emitted = True
                    self._logger.debug(f"[STREAM_TTS] Overflow emit: {len(segment)} chars")
                    return segment
            else:
                # No good word boundary, emit all
                segment = self._buffer.strip()
                self._buffer = ""
                if segment:
                    self._first_emitted = True
                    self._logger.debug(f"[STREAM_TTS] Force emit: {len(segment)} chars")
                    return segment
        
        return None
    
    def _try_first_emit(self) -> Optional[str]:
        """
        Try to emit the first segment sooner for lower perceived latency.
        
        Triggers:
        1. Buffer >= first_emit_chars AND contains any punctuation (.!?,;:)
        2. Buffer contains sentence-ending punctuation (.!?)
        
        Returns:
            First segment if ready, None otherwise
        """
        buf_len = len(self._buffer)
        
        # Check for sentence-ending punctuation (always emit on this)
        sent_match = self.SENTENCE_END_RE.search(self._buffer)
        if sent_match:
            end_pos = sent_match.end()
            segment = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[end_pos:].lstrip()
            if segment:
                self._first_emitted = True
                self._logger.debug(f"[STREAM_TTS] First emit (punctuation): {len(segment)} chars")
                return segment
        
        # Check for first_emit_chars threshold with any pause punctuation
        if buf_len >= self.first_emit_chars:
            punct_match = self.FIRST_EMIT_PUNCT_RE.search(self._buffer)
            if punct_match:
                # Emit up to and including the punctuation
                end_pos = punct_match.end()
                segment = self._buffer[:end_pos].strip()
                self._buffer = self._buffer[end_pos:].lstrip()
                if segment:
                    self._first_emitted = True
                    self._logger.debug(f"[STREAM_TTS] First emit (chars+punct): {len(segment)} chars")
                    return segment
            else:
                # No punctuation but over threshold - find a word boundary
                last_space = self._buffer.rfind(' ', 0, buf_len)
                if last_space > self.first_emit_chars // 2:
                    segment = self._buffer[:last_space].strip()
                    self._buffer = self._buffer[last_space + 1:]
                    if segment:
                        self._first_emitted = True
                        self._logger.debug(f"[STREAM_TTS] First emit (chars): {len(segment)} chars")
                        return segment
        
        return None
    
    def flush(self) -> Optional[str]:
        """
        Flush any remaining buffered text.
        
        Call this when the stream ends to get the final segment.
        
        Returns:
            Remaining text, or None if buffer is empty
        """
        if self._buffer.strip():
            segment = self._buffer.strip()
            self._buffer = ""
            self._logger.debug(f"[STREAM_TTS] Flush emit: {len(segment)} chars")
            return segment
        self._buffer = ""
        return None


def accumulate_full_reply(
    token_stream: Iterator[str],
    on_segment: Callable[[str], None],
    min_chars: int = 150,
    first_emit_chars: int = 24,
    cancel_check: Optional[Callable[[], bool]] = None
) -> str:
    """
    Process a streaming token generator and emit TTS segments.
    
    As tokens arrive, they are buffered and emitted to TTS via on_segment
    callback when sentence/paragraph boundaries are detected. The full
    reply is accumulated and returned.
    
    Args:
        token_stream: Iterator yielding text tokens from LLM
        on_segment: Callback to receive TTS segments as they're ready.
                    Each segment is a complete sentence or paragraph.
        min_chars: Minimum characters before overflow emission
        first_emit_chars: Emit first segment sooner (for lower perceived latency)
        cancel_check: Optional callable returning True to cancel streaming
        
    Returns:
        Complete accumulated reply text
        
    Raises:
        Any exception from the token_stream is propagated to caller
    """
    logger = get_logger()
    buffer = ChunkBuffer(min_chars=min_chars, first_emit_chars=first_emit_chars)
    full_reply = []
    segment_count = 0
    
    try:
        for token in token_stream:
            # Check for cancellation
            if cancel_check and cancel_check():
                logger.debug("[STREAM_TTS] Cancelled by cancel_check")
                break
            
            # Accumulate full reply
            full_reply.append(token)
            
            # Check for segment emission
            segment = buffer.add(token)
            if segment:
                segment_count += 1
                try:
                    on_segment(segment)
                except Exception as e:
                    logger.error(f"[STREAM_TTS] on_segment callback error: {e}")
        
        # Flush remaining buffer
        final_segment = buffer.flush()
        if final_segment:
            segment_count += 1
            try:
                on_segment(final_segment)
            except Exception as e:
                logger.error(f"[STREAM_TTS] on_segment callback error (flush): {e}")
        
        logger.debug(f"[STREAM_TTS] Emitted {segment_count} segments total")
        
    except Exception as e:
        # Log but re-raise for caller to handle
        logger.error(f"[STREAM_TTS] Stream error: {e}")
        raise
    
    return "".join(full_reply)
