"""
Memory Command Detector for Wyzer AI Assistant - Phase 7

Detects explicit memory commands in user text:
- "remember that X" / "remember X" / "save X" / "note that X"
- "forget that X" / "forget X" / "delete that X"
- "what do you remember" / "list memories" / "show memories"

IMPORTANT:
- Only matches when the ENTIRE utterance is clearly a memory command
- Does NOT match if multi-intent separators are present (leaves those for multi_intent_parser)
- Returns deterministic actions without LLM involvement
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

# Multi-intent separators - if present, do NOT treat as memory command
# This preserves multi-intent behavior: "open chrome and then remember X" is multi-intent, not memory
MULTI_INTENT_SEPARATORS_RE = re.compile(
    r'\b(?:and\s+then|then)\b|[;]',
    re.IGNORECASE
)


def _transform_first_to_second_person(text: str) -> Optional[str]:
    """
    Transform first-person memory text to second-person for pronoun-safe responses.
    
    This prevents LLM confusion when session context contains "my name is X" -
    by echoing "your name is X" in the acknowledgment, the session transcript
    correctly frames whose name it is.
    
    Transformations:
    - "my name is X" / "my name's X" → "your name is X"
    - "I'm X" / "I am X" → "your name is X" (only for name-like patterns)
    - "my X is Y" → "your X is Y" (general possessive)
    
    Args:
        text: The memory text to transform
        
    Returns:
        Transformed text in second person, or None if no confident transformation
    """
    if not text:
        return None
    
    text_lower = text.lower().strip()
    
    # Pattern: "my name is X" or "my name's X"
    match = re.match(r"^my\s+name(?:'s|\s+is)\s+(.+)$", text_lower, re.IGNORECASE)
    if match:
        name = match.group(1).strip().rstrip('.!?')
        return f"your name is {name}"
    
    # Pattern: "I'm X" or "I am X" (likely a name introduction)
    match = re.match(r"^i(?:'m|\s+am)\s+(.+)$", text_lower, re.IGNORECASE)
    if match:
        name = match.group(1).strip().rstrip('.!?')
        # Only transform if it looks like a name (short, no "a/the" articles)
        if len(name.split()) <= 3 and not name.startswith(('a ', 'an ', 'the ')):
            return f"your name is {name}"
    
    # Pattern: "my X is Y" (general possessive - favorite color, birthday, etc.)
    match = re.match(r"^my\s+(.+?)\s+is\s+(.+)$", text_lower, re.IGNORECASE)
    if match:
        what = match.group(1).strip()
        value = match.group(2).strip().rstrip('.!?')
        return f"your {what} is {value}"
    
    # Pattern: "my X" without "is" (e.g., "my wifi password BlueHouse123")
    match = re.match(r"^my\s+(.+)$", text_lower, re.IGNORECASE)
    if match:
        rest = match.group(1).strip().rstrip('.!?')
        return f"your {rest}"
    
    # No confident transformation
    return None


class MemoryCommandType(Enum):
    """Types of memory commands."""
    REMEMBER = "remember"
    FORGET = "forget"
    FORGET_LAST = "forget_last"  # "forget that" / "forget it" / "delete that"
    LIST = "list"
    RECALL = "recall"  # Phase 8: "do you remember X" / "what do you remember about X"
    PROMOTE = "promote"  # Phase 9: "use that" / "use this" / "use it"
    CLEAR_PROMOTED = "clear_promoted"  # Phase 9: "stop using that" / "don't use that"
    USE_ALL_MEMORIES = "use_all_memories"  # "use memories" / "enable memories"
    DISABLE_MEMORIES = "disable_memories"  # "stop using memories" / "disable memories"
    # Phase 11: Structured memory commands
    LIST_ALL = "list_all"  # "what do you remember about me" - list all memories
    SEARCH = "search"  # "what do you remember about X" - search/filter
    DELETE = "delete"  # "forget everything about X" - delete matching
    EXPORT = "export"  # "export my memory" / "export memory"
    IMPORT = "import"  # "import memory from X" (optional, explicit only)
    # Deterministic injection (pinned + mention-triggered)
    PIN = "pin"  # "pin that memory" / "always remember X"
    UNPIN = "unpin"  # "unpin X" / "stop always remembering X"
    ADD_ALIAS = "add_alias"  # "add alias X to Y" / "also call Y X"
    NONE = "none"


@dataclass
class MemoryCommand:
    """Parsed memory command."""
    command_type: MemoryCommandType
    text: str  # The content to remember/forget, or empty for list
    original_input: str  # Original user input


def detect_memory_command(user_text: str) -> Optional[MemoryCommand]:
    """
    Detect if user text is an explicit memory command.
    
    IMPORTANT: Only matches if the entire utterance is clearly a memory command.
    Does NOT match if multi-intent separators are present.
    
    Args:
        user_text: User's input text
        
    Returns:
        MemoryCommand if detected, None otherwise
    """
    if not user_text:
        return None
    
    text = user_text.strip()
    text_lower = text.lower()
    
    # === Safety check: Don't process multi-intent commands ===
    # If there are separators like "and then", ";", etc., let multi_intent_parser handle it
    if MULTI_INTENT_SEPARATORS_RE.search(text):
        return None
    
    # === USE_ALL_MEMORIES: Enable memory injection for this session ===
    # "use memories", "enable memories", "turn on memories", "use all memories"
    use_memories_patterns = [
        r'^use\s+(?:all\s+)?memories\.?$',
        r'^enable\s+memories\.?$',
        r'^turn\s+on\s+memories\.?$',
        r'^activate\s+memories\.?$',
        r'^use\s+my\s+memories\.?$',
        r'^enable\s+(?:all\s+)?(?:my\s+)?memories\.?$',
    ]
    for pattern in use_memories_patterns:
        if re.match(pattern, text_lower):
            return MemoryCommand(
                command_type=MemoryCommandType.USE_ALL_MEMORIES,
                text="",
                original_input=text
            )
    
    # === DISABLE_MEMORIES: Disable memory injection for this session ===
    # "stop using memories", "disable memories", "turn off memories", "don't use memories"
    disable_memories_patterns = [
        r'^stop\s+using\s+(?:all\s+)?memories\.?$',
        r'^disable\s+memories\.?$',
        r'^turn\s+off\s+memories\.?$',
        r'^deactivate\s+memories\.?$',
        r'^(?:don\'?t|do\s+not)\s+use\s+(?:all\s+)?memories\.?$',
        r'^stop\s+(?:using\s+)?my\s+memories\.?$',
        r'^disable\s+(?:all\s+)?(?:my\s+)?memories\.?$',
    ]
    for pattern in disable_memories_patterns:
        if re.match(pattern, text_lower):
            return MemoryCommand(
                command_type=MemoryCommandType.DISABLE_MEMORIES,
                text="",
                original_input=text
            )
    
    # === Phase 11: EXPORT command ===
    # "export my memory" / "export memory" / "export memories"
    export_patterns = [
        r'^export\s+(?:my\s+)?memory\.?$',
        r'^export\s+(?:my\s+)?memories\.?$',
        r'^export\s+(?:all\s+)?(?:my\s+)?memories?\.?$',
        r'^save\s+(?:my\s+)?memories?\s+(?:to\s+(?:a\s+)?file|as\s+json)\.?$',
        r'^download\s+(?:my\s+)?memories?\.?$',
        r'^backup\s+(?:my\s+)?memories?\.?$',
    ]
    for pattern in export_patterns:
        if re.match(pattern, text_lower):
            return MemoryCommand(
                command_type=MemoryCommandType.EXPORT,
                text="",
                original_input=text
            )
    
    # === Phase 11: IMPORT command (with path) ===
    # "import memory from X" / "import memories from X"
    import_patterns = [
        # "import memory from X" / "import memories from X"
        (r'^import\s+(?:my\s+)?memor(?:y|ies)\s+from\s+(.+)$', 1),
        # "load memory/memories from X"
        (r'^load\s+(?:my\s+)?memor(?:y|ies)\s+from\s+(.+)$', 1),
        # "restore memory/memories from X"
        (r'^restore\s+(?:my\s+)?memor(?:y|ies)\s+from\s+(.+)$', 1),
    ]
    for pattern, group_idx in import_patterns:
        match = re.match(pattern, text_lower)
        if match:
            path_start = match.start(group_idx)
            # Extract path from original text (preserve case) and clean up
            path = text[path_start:].strip()
            # Remove trailing punctuation but preserve path extensions like .json
            path = re.sub(r'[.!?]+$', '', path) if not path.endswith('.json') else path.rstrip('!?')
            if path:
                return MemoryCommand(
                    command_type=MemoryCommandType.IMPORT,
                    text=path,
                    original_input=text
                )
    
    # =========================================================================
    # Deterministic Injection: PIN / UNPIN / ADD_ALIAS
    # These commands modify the pinned or aliases fields on memory records
    # =========================================================================
    
    # === PIN command: No longer needed ===
    # Since remember() now sets pinned=True by default, we don't need separate PIN commands.
    # Users just say "remember X" and it's automatically always injected.
    # The old "always remember X" patterns are removed to avoid confusion.
    
    # === UNPIN command: "you can forget X" / "sometimes forget X" ===
    # Removes pinned flag from a memory (keeps it but doesn't always inject)
    unpin_patterns = [
        # "you can forget that" / "you can forget it"
        (r'^you\s+can\s+forget\s+(?:that|it|this)\.?$', None, True),
        # "sometimes forget that"
        (r'^sometimes\s+forget\s+(?:that|it|this)\.?$', None, True),
        # "you can forget X"
        (r'^you\s+can\s+forget\s+(.+?)\.?$', 1, False),
        # "sometimes forget X"
        (r'^sometimes\s+forget\s+(.+?)\.?$', 1, False),
        # "don't always use X" / "do not always use X"
        (r'^(?:don\'?t|do\s+not)\s+always\s+use\s+(.+?)\.?$', 1, False),
    ]
    for pattern, group_idx, use_last_recall in unpin_patterns:
        match = re.match(pattern, text_lower)
        if match:
            if use_last_recall:
                # "you can forget that" - uses last recall result
                return MemoryCommand(
                    command_type=MemoryCommandType.UNPIN,
                    text="",
                    original_input=text
                )
            else:
                query_start = match.start(group_idx)
                query = text[query_start:].strip().rstrip('.!?')
                if query:
                    return MemoryCommand(
                        command_type=MemoryCommandType.UNPIN,
                        text=query,
                        original_input=text
                    )
    
    # === ADD_ALIAS command: "add alias X to Y" / "also call Y X" ===
    # Adds an alias to a memory for better mention matching
    alias_patterns = [
        # "add alias X to Y" / "add alias X for Y"
        (r'^add\s+alias\s+["\']?(.+?)["\']?\s+(?:to|for)\s+(.+?)\.?$', 1, 2),
        # "alias X as Y" (alias the memory matching X with name Y)
        (r'^alias\s+(.+?)\s+as\s+["\']?(.+?)["\']?\.?$', 1, 2),
        # "also call X Y" (call the memory matching X by alias Y)
        (r'^also\s+call\s+(.+?)\s+["\']?(.+?)["\']?\.?$', 1, 2),
        # "give X the alias Y"
        (r'^give\s+(.+?)\s+the\s+alias\s+["\']?(.+?)["\']?\.?$', 1, 2),
        # "nickname X as Y" / "nickname X Y"
        (r'^nickname\s+(.+?)\s+(?:as\s+)?["\']?(.+?)["\']?\.?$', 1, 2),
    ]
    for pattern, query_group, alias_group in alias_patterns:
        match = re.match(pattern, text_lower)
        if match:
            # Extract query (memory to find) and alias (new name) from original text
            query = match.group(query_group).strip().strip('"\'')
            alias = match.group(alias_group).strip().strip('"\'')
            if query and alias:
                # Pack both into text as "query||alias"
                return MemoryCommand(
                    command_type=MemoryCommandType.ADD_ALIAS,
                    text=f"{query}||{alias}",
                    original_input=text
                )
    
    # === Phase 11: LIST_ALL command (about me) ===
    # "what do you remember about me" - lists ALL memories
    list_all_patterns = [
        r'^what\s+do\s+you\s+remember\s+about\s+me\??$',
        r'^what\s+do\s+you\s+know\s+about\s+me\??$',
        r'^tell\s+me\s+(?:everything\s+)?(?:you\s+)?(?:know|remember)\s+about\s+me\.?$',
        r'^list\s+(?:all\s+)?(?:my\s+)?memories?\s+about\s+me\.?$',
        r'^show\s+me\s+(?:all\s+)?(?:my\s+)?memories?\.?$',
        r'^what\s+have\s+you\s+learned\s+about\s+me\??$',
    ]
    for pattern in list_all_patterns:
        if re.match(pattern, text_lower):
            return MemoryCommand(
                command_type=MemoryCommandType.LIST_ALL,
                text="",
                original_input=text
            )
    
    # === Phase 11: DELETE command (forget everything about X) ===
    # "forget everything about X" / "forget all about X"
    # IMPORTANT: Must be checked BEFORE generic forget patterns
    delete_patterns = [
        # "forget everything about X"
        (r'^forget\s+everything\s+about\s+(.+?)\.?$', 1),
        # "forget all about X"
        (r'^forget\s+all\s+(?:about\s+)?(.+?)\.?$', 1),
        # "delete everything about X"
        (r'^delete\s+everything\s+(?:about\s+)?(.+?)\.?$', 1),
        # "delete all memories about X"
        (r'^delete\s+all\s+(?:memories?\s+)?(?:about\s+)?(.+?)\.?$', 1),
        # "remove all memories about X"
        (r'^remove\s+all\s+(?:memories?\s+)?(?:about\s+)?(.+?)\.?$', 1),
        # "clear everything about X"
        (r'^clear\s+everything\s+(?:about\s+)?(.+?)\.?$', 1),
    ]
    for pattern, group_idx in delete_patterns:
        match = re.match(pattern, text_lower)
        if match:
            query_start = match.start(group_idx)
            query = text[query_start:].strip().rstrip('.!?')
            if query:
                return MemoryCommand(
                    command_type=MemoryCommandType.DELETE,
                    text=query,
                    original_input=text
                )
    
    # === Phase 11: SEARCH command (what do you remember about X, where X != "me") ===
    # This overrides the RECALL behavior for Phase 11 - search returns formatted list
    search_patterns = [
        # "what do you remember about X" (X != "me", handled above)
        (r'^what\s+do\s+you\s+remember\s+about\s+(.+?)\??$', 1),
        # "what do you know about X" (X != "me")
        (r'^what\s+do\s+you\s+know\s+about\s+(.+?)\??$', 1),
        # "search memories for X"
        (r'^search\s+(?:my\s+)?memories?\s+(?:for\s+)?(.+?)\.?$', 1),
        # "find memories about X"
        (r'^find\s+(?:my\s+)?memories?\s+(?:about\s+)?(.+?)\.?$', 1),
    ]
    for pattern, group_idx in search_patterns:
        match = re.match(pattern, text_lower)
        if match:
            query_start = match.start(group_idx)
            query = text[query_start:].strip().rstrip('.!?')
            # Skip if query is "me" (handled by LIST_ALL above)
            if query and query.lower() != "me":
                return MemoryCommand(
                    command_type=MemoryCommandType.SEARCH,
                    text=query,
                    original_input=text
                )
    
    # === List memories ===
    list_patterns = [
        r'^what\s+do\s+you\s+remember\??$',
        r'^list\s+(?:my\s+)?memories?\??$',
        r'^show\s+(?:my\s+)?memories?\??$',
        r'^(?:what\s+)?memories?\s+do\s+(?:you|i)\s+have\??$',
        r'^recall\s+(?:all\s+)?memories?\??$',
        r'^what\s+have\s+(?:you|i)\s+(?:remembered|saved)\??$',
    ]
    for pattern in list_patterns:
        if re.match(pattern, text_lower):
            return MemoryCommand(
                command_type=MemoryCommandType.LIST,
                text="",
                original_input=text
            )
    
    # === Recall commands (Phase 8) ===
    # "do you remember X" / "what do you remember about X" / "do you know X"
    # These search for specific memories without listing all
    recall_patterns = [
        # "do you remember X" / "do you remember my X"
        (r'^do\s+you\s+remember\s+(?:my\s+)?(.+?)\??$', 1),
        # "what do you remember about X"
        (r'^what\s+do\s+you\s+remember\s+about\s+(.+?)\??$', 1),
        # "do you know X" / "do you know my X"
        (r'^do\s+you\s+know\s+(?:my\s+)?(.+?)\??$', 1),
        # "what do you know about X"
        (r'^what\s+do\s+you\s+know\s+about\s+(.+?)\??$', 1),
    ]
    for pattern, group_idx in recall_patterns:
        match = re.match(pattern, text_lower)
        if match:
            # Extract the query (use original case from input for context)
            query_start = match.start(group_idx)
            query_end = match.end(group_idx)
            # Map back to original text - find where pattern match starts in lowered
            query = text_lower[query_start:query_end].strip()
            # Remove trailing punctuation for cleaner query
            query = re.sub(r'[.!?]+$', '', query).strip()
            if query:
                return MemoryCommand(
                    command_type=MemoryCommandType.RECALL,
                    text=query,
                    original_input=text
                )
    
    # === Remember commands ===
    remember_patterns = [
        # "remember that X" - most explicit
        (r'^remember\s+that\s+(.+)$', 1),
        # "remember, X" - with comma (e.g., "Remember, my name is Levi")
        (r'^remember\s*,\s*(.+)$', 1),
        # "remember X" - simple form
        (r'^remember\s+(.+)$', 1),
        # "save that X" / "save X"
        (r'^save\s+(?:that\s+)?(.+)$', 1),
        # "note that X" / "note X" / "make a note that X"
        (r'^(?:make\s+a\s+)?note\s+(?:that\s+)?(.+)$', 1),
        # "keep in mind that X" / "keep in mind X"
        (r'^keep\s+in\s+mind\s+(?:that\s+)?(.+)$', 1),
    ]
    for pattern, group_idx in remember_patterns:
        match = re.match(pattern, text_lower)
        if match:
            # Extract the content to remember (use original case)
            content_start = match.start(group_idx)
            content = text[content_start:].strip()
            if content:
                return MemoryCommand(
                    command_type=MemoryCommandType.REMEMBER,
                    text=content,
                    original_input=text
                )
    
    # === Trailing "remember that" patterns (common speech: "X, remember that" / "X. Remember that.") ===
    # These patterns have the content BEFORE "remember that"
    trailing_remember_patterns = [
        # "X, remember that" - content before comma + remember that
        (r'^(.+?)[,;]\s*remember\s+that\.?$', 1),
        # "X. Remember that." - content before period + remember that
        (r'^(.+?)\.\s*remember\s+that\.?$', 1),
        # "X - remember that" - content before dash
        (r'^(.+?)\s*[-–—]\s*remember\s+that\.?$', 1),
        # "X, remember this" - variation with "this"
        (r'^(.+?)[,;]\s*remember\s+(?:this|it)\.?$', 1),
        # "X. Remember this." - period variant
        (r'^(.+?)\.\s*remember\s+(?:this|it)\.?$', 1),
    ]
    for pattern, group_idx in trailing_remember_patterns:
        match = re.match(pattern, text_lower)
        if match:
            # Extract the content to remember from the FIRST group (use original case)
            content_start = match.start(group_idx)
            content_end = match.end(group_idx)
            content = text[content_start:content_end].strip()
            if content:
                return MemoryCommand(
                    command_type=MemoryCommandType.REMEMBER,
                    text=content,
                    original_input=text
                )
    
    # === Forget LAST (special case: "forget that" / "forget it" / "delete that") ===
    # These exact phrases mean "delete the most recent memory"
    forget_last_patterns = [
        r'^forget\s+that\.?$',
        r'^forget\s+it\.?$',
        r'^delete\s+that\.?$',
        r'^remove\s+that\.?$',
        r'^never\s*mind\.?$',
        r'^nevermind\.?$',
    ]
    for pattern in forget_last_patterns:
        if re.match(pattern, text_lower):
            return MemoryCommand(
                command_type=MemoryCommandType.FORGET_LAST,
                text="",
                original_input=text
            )
    
    # === Phase 9: CLEAR_PROMOTED commands (stop using promoted memory) ===
    # These phrases mean "stop using the promoted memories"
    # IMPORTANT: Check BEFORE generic forget patterns to avoid false matches
    clear_promoted_patterns = [
        r'^stop\s+using\s+that\.?$',
        r'^stop\s+using\s+(?:this|it)\.?$',
        r'^(?:don\'?t|do\s+not)\s+use\s+that\.?$',
        r'^(?:don\'?t|do\s+not)\s+use\s+(?:this|it)\.?$',
        r'^forget\s+(?:that|it)\s+for\s+now\.?$',
        r'^(?:don\'?t|do\s+not)\s+use\s+that\s+anymore\.?$',
        r'^stop\s+using\s+(?:that|this|it)\s+(?:for\s+now|anymore)\.?$',
        r'^never\s*mind\s+(?:that|it)\.?$',
    ]
    for pattern in clear_promoted_patterns:
        if re.match(pattern, text_lower):
            return MemoryCommand(
                command_type=MemoryCommandType.CLEAR_PROMOTED,
                text="",
                original_input=text
            )
    
    # === Forget commands (with search query) ===
    forget_patterns = [
        # "forget that X" - most explicit (note: "forget that" alone is caught above)
        (r'^forget\s+that\s+(.+)$', 1),
        # "forget X" - simple form
        (r'^forget\s+(.+)$', 1),
        # "delete that X" / "delete memory X"
        (r'^delete\s+(?:that\s+|memory\s+|the\s+memory\s+(?:about\s+)?)?(.+)$', 1),
        # "remove that X" / "remove memory X"
        (r'^remove\s+(?:that\s+|memory\s+|the\s+memory\s+(?:about\s+)?)?(.+)$', 1),
        # "clear memory about X"
        (r'^clear\s+(?:the\s+)?memory\s+(?:about\s+)?(.+)$', 1),
    ]
    for pattern, group_idx in forget_patterns:
        match = re.match(pattern, text_lower)
        if match:
            content_start = match.start(group_idx)
            content = text[content_start:].strip()
            if content:
                return MemoryCommand(
                    command_type=MemoryCommandType.FORGET,
                    text=content,
                    original_input=text
                )
    
    # === Phase 9: PROMOTE commands (use recalled memory for this conversation) ===
    # These exact phrases mean "use the most recent recall result"
    # IMPORTANT: Only matches if there's a recent recall result (checked at runtime)
    promote_patterns = [
        r'^use\s+that\.?$',
        r'^use\s+this\.?$',
        r'^use\s+it\.?$',
        r'^use\s+that\s+for\s+(?:this\s+)?conversation\.?$',
        r'^use\s+(?:that|this|it)\s+for\s+now\.?$',
        r'^(?:yes|yeah|yep|ok|okay)[,.]?\s*use\s+(?:that|this|it)\.?$',
    ]
    for pattern in promote_patterns:
        if re.match(pattern, text_lower):
            return MemoryCommand(
                command_type=MemoryCommandType.PROMOTE,
                text="",
                original_input=text
            )
    
    return None


def is_memory_command(user_text: str) -> bool:
    """
    Quick check if user text is a memory command.
    
    Args:
        user_text: User's input text
        
    Returns:
        True if it's a memory command, False otherwise
    """
    return detect_memory_command(user_text) is not None


def handle_memory_command(user_text: str) -> Optional[Tuple[str, dict]]:
    """
    Handle a memory command and return the response.
    
    This function:
    1. Detects if the input is a memory command
    2. Executes the appropriate memory action
    3. Returns a spoken response
    
    Does NOT call LLM or tools.
    
    Args:
        user_text: User's input text
        
    Returns:
        Tuple of (response_text, metadata) if handled, None otherwise
    """
    from wyzer.memory.memory_manager import get_memory_manager
    
    cmd = detect_memory_command(user_text)
    if cmd is None:
        return None
    
    memory_mgr = get_memory_manager()
    
    if cmd.command_type == MemoryCommandType.REMEMBER:
        result = memory_mgr.remember(cmd.text)
        if result.get("ok"):
            # Pronoun-safe acknowledgment: transform first-person to second-person
            # This prevents LLM confusion when session context contains "my name is X"
            transformed = _transform_first_to_second_person(cmd.text)
            if transformed:
                response = f"Got it — I'll remember that {transformed}."
            else:
                response = "Got it — I'll remember that."
            return (
                response,
                {"memory_action": "remember", "ok": True, "entry": result.get("entry")}
            )
        else:
            return (
                "Sorry, I couldn't save that.",
                {"memory_action": "remember", "ok": False, "error": result.get("error")}
            )
    
    elif cmd.command_type == MemoryCommandType.FORGET:
        result = memory_mgr.forget(cmd.text)
        removed = result.get("removed", [])
        if removed:
            if len(removed) == 1:
                return (
                    "Okay — I forgot it.",
                    {"memory_action": "forget", "ok": True, "removed_count": 1}
                )
            else:
                return (
                    f"Okay — I forgot {len(removed)} memories.",
                    {"memory_action": "forget", "ok": True, "removed_count": len(removed)}
                )
        else:
            return (
                "I don't have any memories matching that.",
                {"memory_action": "forget", "ok": True, "removed_count": 0}
            )
    
    elif cmd.command_type == MemoryCommandType.FORGET_LAST:
        result = memory_mgr.forget_last()
        removed = result.get("removed")
        if removed:
            return (
                "Okay — I forgot that.",
                {"memory_action": "forget_last", "ok": True, "removed": removed}
            )
        else:
            return (
                "I don't have any memories to forget.",
                {"memory_action": "forget_last", "ok": True, "removed": None}
            )
    
    elif cmd.command_type == MemoryCommandType.LIST:
        # Long-term memory = only surfaced via explicit list command (Phase 7 contract)
        # This is the ONLY place where memory.json contents are exposed to the user
        memories = memory_mgr.list_memories()
        if not memories:
            return (
                "I don't have any saved memories.",
                {"memory_action": "list", "count": 0}
            )
        elif len(memories) == 1:
            text = memories[0].get("text", "")
            return (
                f"Here's what I remember: {text}",
                {"memory_action": "list", "count": 1, "memories": memories}
            )
        else:
            # Summarize multiple memories
            summary_parts = []
            for i, mem in enumerate(memories[:5], 1):  # Limit to 5 for speech
                text = mem.get("text", "")[:80]  # Truncate long entries
                summary_parts.append(f"{i}. {text}")
            
            summary = "; ".join(summary_parts)
            if len(memories) > 5:
                summary += f". And {len(memories) - 5} more."
            
            return (
                f"Here's what I remember: {summary}",
                {"memory_action": "list", "count": len(memories), "memories": memories}
            )
    
    elif cmd.command_type == MemoryCommandType.RECALL:
        # Phase 8: Explicit recall - search memories by query (READ-ONLY)
        # Deterministic search, no LLM involvement, no disk writes
        matches = memory_mgr.recall(cmd.text, limit=5)
        
        if not matches:
            # Clear any previous recall result since this search found nothing
            memory_mgr.set_last_recall_result(None)
            return (
                "I don't have anything saved about that.",
                {"memory_action": "recall", "query": cmd.text, "count": 0, "matches": []}
            )
        elif len(matches) == 1:
            text = matches[0].get("text", "")
            # Phase 9: Store the recall result for potential "use that" promotion
            memory_mgr.set_last_recall_result(text)
            # Quote-safe response: try pronoun transform, else quote the raw text
            # This prevents LLM from misinterpreting "my name" as the assistant's name
            transformed = _transform_first_to_second_person(text)
            if transformed:
                response = f"Yes — {transformed}."
            else:
                # Fallback: quote the original text to make attribution clear
                response = f"Yes — you told me: '{text}'"
            return (
                response,
                {"memory_action": "recall", "query": cmd.text, "count": 1, "matches": matches}
            )
        else:
            # Multiple matches - store the first/best match for potential promotion
            first_match_text = matches[0].get("text", "")
            memory_mgr.set_last_recall_result(first_match_text)
            
            # Multiple matches - list as bullets with quotes for safety
            bullets = []
            for mem in matches:
                text = mem.get("text", "")[:80]  # Truncate long entries
                transformed = _transform_first_to_second_person(text)
                if transformed:
                    bullets.append(f"• {transformed}")
                else:
                    bullets.append(f"• '{text}'")
            
            response = "Here's what I remember: " + "; ".join(bullets)
            return (
                response,
                {"memory_action": "recall", "query": cmd.text, "count": len(matches), "matches": matches}
            )
    
    elif cmd.command_type == MemoryCommandType.PROMOTE:
        # Phase 9: Promote recalled memory for temporary use in this session
        # IMPORTANT: Only works if there's a recent recall result
        last_recall = memory_mgr.get_last_recall_result()
        
        if not last_recall:
            # No recent recall - cannot promote
            return (
                "I don't have anything to use. Try asking 'do you remember...' first.",
                {"memory_action": "promote", "ok": False, "error": "no_recent_recall"}
            )
        
        # Promote the last recalled memory
        if memory_mgr.promote(last_recall):
            # Phase 9 hardening: Clear the recall result after successful promote
            # This prevents double-promotion or stale references
            memory_mgr.set_last_recall_result(None)
            return (
                "Okay — I'll use that for this conversation.",
                {"memory_action": "promote", "ok": True, "promoted": last_recall}
            )
        else:
            return (
                "Sorry, I couldn't use that.",
                {"memory_action": "promote", "ok": False, "error": "promote_failed"}
            )
    
    elif cmd.command_type == MemoryCommandType.CLEAR_PROMOTED:
        # Phase 9: Clear promoted memories (stop using them)
        count = memory_mgr.clear_promoted()
        
        if count > 0:
            return (
                "Okay — I won't use that anymore.",
                {"memory_action": "clear_promoted", "ok": True, "cleared_count": count}
            )
        else:
            return (
                "I wasn't using any saved memories.",
                {"memory_action": "clear_promoted", "ok": True, "cleared_count": 0}
            )
    
    elif cmd.command_type == MemoryCommandType.USE_ALL_MEMORIES:
        # Enable injection of ALL long-term memories into LLM prompts
        # Session-scoped only, no disk writes
        changed = memory_mgr.set_use_memories(True, source="voice_command")
        if changed:
            return (
                "Okay — I'll use all your saved memories now.",
                {"memory_action": "use_all_memories", "ok": True, "enabled": True}
            )
        else:
            return (
                "I'm already using your memories.",
                {"memory_action": "use_all_memories", "ok": True, "enabled": True, "already_set": True}
            )
    
    elif cmd.command_type == MemoryCommandType.DISABLE_MEMORIES:
        # Disable injection of long-term memories into LLM prompts
        # Session-scoped only, no disk writes
        changed = memory_mgr.set_use_memories(False, source="voice_command")
        if changed:
            return (
                "Okay — I'll stop using your memories.",
                {"memory_action": "disable_memories", "ok": True, "enabled": False}
            )
        else:
            return (
                "I wasn't using your memories.",
                {"memory_action": "disable_memories", "ok": True, "enabled": False, "already_set": True}
            )
    
    # =========================================================================
    # Phase 11: Structured Memory Commands
    # =========================================================================
    
    elif cmd.command_type == MemoryCommandType.LIST_ALL:
        # Phase 11: "What do you remember about me?" - list all memories grouped by type
        grouped = memory_mgr.get_memories_grouped_by_type()
        total = sum(len(mems) for mems in grouped.values())
        
        if total == 0:
            return (
                "I don't have any memories about you yet. You can say 'remember' followed by something you want me to remember.",
                {"memory_action": "list_all", "count": 0}
            )
        
        # Build a human-readable response grouped by type
        lines = []
        type_labels = {
            "fact": "Facts",
            "preference": "Preferences",
            "skill": "Skills",
            "history_marker": "History"
        }
        
        # Collect all for full listing
        all_memories = []
        for mem_type in ["fact", "preference", "skill", "history_marker"]:
            mems = grouped.get(mem_type, [])
            if mems:
                label = type_labels.get(mem_type, mem_type.capitalize())
                lines.append(f"{label}:")
                for mem in mems[:10]:  # Limit per category for speech
                    value = mem.get("value") or mem.get("text") or ""
                    # Transform to second person for clarity
                    transformed = _transform_first_to_second_person(value)
                    display = transformed if transformed else value
                    lines.append(f"  • {display}")
                    all_memories.append(mem)
                if len(mems) > 10:
                    lines.append(f"  ... and {len(mems) - 10} more")
        
        # Build speech response (abbreviated for TTS)
        if total == 1:
            mem = all_memories[0]
            value = mem.get("value") or mem.get("text") or ""
            transformed = _transform_first_to_second_person(value)
            speech = f"Here's what I remember about you: {transformed if transformed else value}"
        elif total <= 5:
            items = []
            for mem in all_memories[:5]:
                value = mem.get("value") or mem.get("text") or ""
                transformed = _transform_first_to_second_person(value)
                items.append(transformed if transformed else value)
            speech = "Here's what I remember about you: " + "; ".join(items)
        else:
            # Summarize for speech
            speech = f"I have {total} memories about you. " + "; ".join(lines[:3])
            if total > 5:
                speech += f". And {total - 5} more."
        
        # Add helpful hint
        speech += " You can say 'forget everything about X' or 'export my memory'."
        
        return (
            speech,
            {
                "memory_action": "list_all",
                "count": total,
                "grouped": grouped,
                "formatted": "\n".join(lines)
            }
        )
    
    elif cmd.command_type == MemoryCommandType.SEARCH:
        # Phase 11: "What do you remember about X?" - search/filter memories
        query = cmd.text
        matches = memory_mgr.search(query)
        
        if not matches:
            return (
                f"I don't have any memories about '{query}'.",
                {"memory_action": "search", "query": query, "count": 0, "matches": []}
            )
        elif len(matches) == 1:
            value = matches[0].get("value") or matches[0].get("text") or ""
            transformed = _transform_first_to_second_person(value)
            response = f"Here's what I know about {query}: {transformed if transformed else value}"
            return (
                response,
                {"memory_action": "search", "query": query, "count": 1, "matches": matches}
            )
        else:
            # Multiple matches
            items = []
            for mem in matches[:5]:
                value = mem.get("value") or mem.get("text") or ""
                transformed = _transform_first_to_second_person(value)
                items.append(transformed if transformed else value)
            
            response = f"Here's what I know about {query}: " + "; ".join(items)
            if len(matches) > 5:
                response += f". And {len(matches) - 5} more."
            
            return (
                response,
                {"memory_action": "search", "query": query, "count": len(matches), "matches": matches}
            )
    
    elif cmd.command_type == MemoryCommandType.DELETE:
        # Phase 11: "Forget everything about X" - delete matching memories
        query = cmd.text
        deleted_count = memory_mgr.delete_by_query(query)
        
        if deleted_count == 0:
            return (
                f"I don't have any memories about '{query}'.",
                {"memory_action": "delete", "query": query, "deleted_count": 0}
            )
        elif deleted_count == 1:
            return (
                f"Okay — I forgot everything about {query}.",
                {"memory_action": "delete", "query": query, "deleted_count": 1}
            )
        else:
            return (
                f"Okay — I forgot {deleted_count} memories about {query}.",
                {"memory_action": "delete", "query": query, "deleted_count": deleted_count}
            )
    
    elif cmd.command_type == MemoryCommandType.EXPORT:
        # Phase 11: "Export my memory" - write to JSON file
        try:
            export_path = memory_mgr.export_to()
            return (
                f"Done! I exported your memories to {export_path}",
                {"memory_action": "export", "ok": True, "path": export_path}
            )
        except Exception as e:
            return (
                f"Sorry, I couldn't export your memories: {str(e)}",
                {"memory_action": "export", "ok": False, "error": str(e)}
            )
    
    elif cmd.command_type == MemoryCommandType.IMPORT:
        # Phase 11: "Import memory from X" - import from JSON file
        path = cmd.text
        try:
            imported_count = memory_mgr.import_from(path)
            if imported_count == 0:
                return (
                    "The file was read, but all memories were already saved.",
                    {"memory_action": "import", "ok": True, "imported_count": 0, "path": path}
                )
            elif imported_count == 1:
                return (
                    "Done! I imported 1 new memory.",
                    {"memory_action": "import", "ok": True, "imported_count": 1, "path": path}
                )
            else:
                return (
                    f"Done! I imported {imported_count} new memories.",
                    {"memory_action": "import", "ok": True, "imported_count": imported_count, "path": path}
                )
        except FileNotFoundError:
            return (
                f"I couldn't find the file: {path}",
                {"memory_action": "import", "ok": False, "error": "file_not_found", "path": path}
            )
        except ValueError as e:
            return (
                f"The file format is invalid: {str(e)}",
                {"memory_action": "import", "ok": False, "error": str(e), "path": path}
            )
        except Exception as e:
            return (
                f"Sorry, I couldn't import your memories: {str(e)}",
                {"memory_action": "import", "ok": False, "error": str(e), "path": path}
            )
    
    # =========================================================================
    # Deterministic Injection: PIN / UNPIN / ADD_ALIAS handlers
    # =========================================================================
    
    elif cmd.command_type == MemoryCommandType.PIN:
        # PIN is no longer needed - all remembered memories are auto-pinned
        # This handler exists for backwards compatibility but shouldn't be triggered
        # since we removed the PIN detection patterns
        query = cmd.text or ""
        if query:
            # If somehow triggered, just do a remember which auto-pins
            result = memory_mgr.remember(query)
            if result.get("ok"):
                return (
                    f"I'll remember that.",
                    {"memory_action": "remember", "ok": True, "pinned": True, "entry": result.get("entry")}
                )
        return (
            "All memories are now automatically remembered. Just say 'remember X'.",
            {"memory_action": "pin", "ok": False, "error": "deprecated"}
        )
    
    elif cmd.command_type == MemoryCommandType.UNPIN:
        # Unpin a memory (remove from always-injected)
        query = cmd.text
        
        if not query:
            # "unpin that" - use last recall result
            last_recall = memory_mgr.get_last_recall_result()
            if not last_recall:
                return (
                    "I don't have a memory to unpin. Try 'do you remember...' first.",
                    {"memory_action": "unpin", "ok": False, "error": "no_recent_recall"}
                )
            query = last_recall
        
        # Try to unpin a matching memory
        result = memory_mgr.set_pinned_by_query(query, pinned=False)
        if result.get("ok"):
            value = result.get("value", query)
            transformed = _transform_first_to_second_person(value)
            display = transformed if transformed else value
            return (
                f"Okay — I won't always include that: {display}",
                {"memory_action": "unpin", "ok": True, "pinned": False, "query": query, "entry": result.get("entry")}
            )
        else:
            return (
                f"I don't have any memories about '{query}' to unpin.",
                {"memory_action": "unpin", "ok": False, "error": "not_found", "query": query}
            )
    
    elif cmd.command_type == MemoryCommandType.ADD_ALIAS:
        # Add an alias to a memory
        # text format: "query||alias"
        parts = cmd.text.split("||", 1)
        if len(parts) != 2:
            return (
                "I couldn't understand which memory to alias. Try 'add alias X to Y'.",
                {"memory_action": "add_alias", "ok": False, "error": "invalid_format"}
            )
        
        query, alias = parts[0].strip(), parts[1].strip()
        
        result = memory_mgr.add_alias_by_query(query, alias)
        if result.get("ok"):
            return (
                f"Okay — I added '{alias}' as an alias.",
                {"memory_action": "add_alias", "ok": True, "query": query, "alias": alias, "entry": result.get("entry")}
            )
        else:
            return (
                f"I don't have any memories about '{query}' to add an alias to.",
                {"memory_action": "add_alias", "ok": False, "error": "not_found", "query": query}
            )
    
    return None


def handle_source_question(user_text: str) -> Optional[Tuple[str, dict]]:
    """
    Handle "how do you know X?" questions deterministically.
    
    This intercepts questions about information source and provides
    truthful answers based on where the info actually came from:
    - Session RAM: "You told me earlier in this session."
    - Long-term memory: Redirect to "list memories" command
    - Unknown: "I don't know that yet."
    
    Phase 7 polish: Keeps the assistant honest about memory sources.
    
    Args:
        user_text: User's input text
        
    Returns:
        Tuple of (response_text, metadata) if handled, None otherwise
    """
    if not user_text:
        return None
    
    text = user_text.strip()
    text_lower = text.lower()
    
    # Patterns for "how do you know" questions
    how_know_patterns = [
        r'^how\s+(?:do|did)\s+you\s+know\s+(?:that|my\s+\w+)\??$',
        r'^how\s+(?:do|did)\s+you\s+know\s+(?:that|what|who)\??$',
        r'^where\s+did\s+(?:that|you\s+get\s+that)\s+(?:come\s+from|info(?:rmation)?)\??$',
        r'^how\s+(?:do|did)\s+you\s+(?:know|remember)\s+that\??$',
        r'^how\s+(?:do|did)\s+you\s+know\??$',
    ]
    
    is_source_question = False
    for pattern in how_know_patterns:
        if re.match(pattern, text_lower):
            is_source_question = True
            break
    
    if not is_source_question:
        return None
    
    # Check session memory for recent context
    from wyzer.memory.memory_manager import get_memory_manager
    mem_mgr = get_memory_manager()
    
    session_context = mem_mgr.get_session_context(max_turns=10)
    has_session_context = bool(session_context and session_context.strip())
    
    if has_session_context:
        # Info came from session RAM
        return (
            "You told me earlier in this session.",
            {"source_question": True, "source": "session"}
        )
    else:
        # No session context - we don't know
        return (
            "I don't know that yet. You haven't told me in this session.",
            {"source_question": True, "source": "none"}
        )
