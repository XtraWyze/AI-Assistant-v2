"""
Internal message representation for LLM prompt construction.

This module provides a structured messages[] representation for building prompts
internally while maintaining compatibility with Ollama's single-prompt-string API.

Architecture note:
- messages[] is INTERNAL ONLY; the transport to Ollama remains a single prompt string.
- This enables future streaming-to-TTS without changing routing, by allowing us to
  selectively stream only user/assistant blocks while keeping system context fixed.
- No tool/function syntax is introduced - this is purely for prompt organization.

Usage:
    from wyzer.brain.messages import msg_system, msg_user, msg_assistant, flatten_messages
    
    messages = [
        msg_system("You are Wyzer..."),
        msg_user("What time is it?"),
    ]
    prompt_string = flatten_messages(messages)
"""
from typing import List, Literal, TypedDict


# -----------------------------------------------------------------------------
# Message Type Definition
# -----------------------------------------------------------------------------

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """
    A single message in the internal conversation representation.
    
    Attributes:
        role: One of "system", "user", or "assistant"
        content: The text content of the message
    """
    role: Role
    content: str


# -----------------------------------------------------------------------------
# Message Constructors
# -----------------------------------------------------------------------------

def msg_system(content: str) -> Message:
    """
    Create a system message.
    
    System messages contain instructions, persona, tool definitions, etc.
    
    Args:
        content: The system instruction text
        
    Returns:
        A Message dict with role="system"
    """
    return {"role": "system", "content": content}


def msg_user(content: str) -> Message:
    """
    Create a user message.
    
    User messages contain the transcribed user speech or text input.
    
    Args:
        content: The user's input text
        
    Returns:
        A Message dict with role="user"
    """
    return {"role": "user", "content": content}


def msg_assistant(content: str) -> Message:
    """
    Create an assistant message.
    
    Assistant messages contain previous Wyzer responses (for context).
    
    Args:
        content: The assistant's previous response
        
    Returns:
        A Message dict with role="assistant"
    """
    return {"role": "assistant", "content": content}


# -----------------------------------------------------------------------------
# Flatten Function - Converts messages[] to single prompt string
# -----------------------------------------------------------------------------

def flatten_messages(
    messages: List[Message],
    include_role_headers: bool = False,
    block_separator: str = "\n\n"
) -> str:
    """
    Flatten a list of messages into a single prompt string for Ollama.
    
    This function preserves the current prompt format - it does NOT introduce
    new "role:" markers unless include_role_headers is explicitly set to True.
    The default behavior produces output identical to the existing prompt format.
    
    Args:
        messages: List of Message dicts to flatten
        include_role_headers: If True, prefix each block with "System:", "User:", etc.
                             Default False to match existing prompt format.
        block_separator: String to place between message blocks. Default "\\n\\n".
    
    Returns:
        A single string suitable for Ollama's prompt field.
        
    Example (default, no headers):
        >>> msgs = [msg_system("You are Wyzer."), msg_user("Hello")]
        >>> flatten_messages(msgs)
        "You are Wyzer.\\n\\nHello"
        
    Example (with headers):
        >>> flatten_messages(msgs, include_role_headers=True)
        "System:\\nYou are Wyzer.\\n\\nUser:\\nHello"
    """
    if not messages:
        return ""
    
    parts = []
    for msg in messages:
        content = msg.get("content", "").strip()
        if not content:
            continue
            
        if include_role_headers:
            role = msg.get("role", "user")
            # Capitalize role for display
            role_header = role.capitalize()
            parts.append(f"{role_header}:\n{content}")
        else:
            parts.append(content)
    
    return block_separator.join(parts)


# -----------------------------------------------------------------------------
# Message Builder - Convenience class for building message lists
# -----------------------------------------------------------------------------

class MessageBuilder:
    """
    Convenience class for building message lists incrementally.
    
    Example:
        builder = MessageBuilder()
        builder.system("You are Wyzer...")
        builder.user("What time is it?")
        messages = builder.build()
        prompt = builder.flatten()
    """
    
    def __init__(self):
        """Initialize an empty message list."""
        self._messages: List[Message] = []
    
    def system(self, content: str) -> "MessageBuilder":
        """Add a system message. Returns self for chaining."""
        if content and content.strip():
            self._messages.append(msg_system(content))
        return self
    
    def user(self, content: str) -> "MessageBuilder":
        """Add a user message. Returns self for chaining."""
        if content and content.strip():
            self._messages.append(msg_user(content))
        return self
    
    def assistant(self, content: str) -> "MessageBuilder":
        """Add an assistant message. Returns self for chaining."""
        if content and content.strip():
            self._messages.append(msg_assistant(content))
        return self
    
    def add(self, message: Message) -> "MessageBuilder":
        """Add a pre-constructed message. Returns self for chaining."""
        if message and message.get("content", "").strip():
            self._messages.append(message)
        return self
    
    def build(self) -> List[Message]:
        """Return the constructed message list."""
        return self._messages.copy()
    
    def flatten(self, include_role_headers: bool = False) -> str:
        """Flatten messages to a prompt string."""
        return flatten_messages(self._messages, include_role_headers=include_role_headers)
    
    def clear(self) -> "MessageBuilder":
        """Clear all messages. Returns self for chaining."""
        self._messages = []
        return self
    
    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self._messages)
    
    def __bool__(self) -> bool:
        """Return True if there are any messages."""
        return bool(self._messages)
