"""wyzer.core.hybrid_router

Hybrid deterministic router for obvious commands.

It returns either:
- a deterministic tool plan (one or more tool calls), OR
- a decision to use the LLM.

This module is intentionally conservative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from wyzer.core.multi_intent_parser import parse_multi_intent_with_fallback


def _strip_trailing_punct(text: str) -> str:
    return (text or "").strip().rstrip(".?!,;:\"'")


def _extract_volume_percent(text: str) -> Optional[int]:
    m = re.search(r"\b(\d{1,3})\s*%?\b", text or "")
    if not m:
        return None
    try:
        v = int(m.group(1))
    except Exception:
        return None
    if 0 <= v <= 100:
        return v
    return None


def _parse_volume_delta_hint(text_lower: str) -> int:
    tl = text_lower or ""
    if any(k in tl for k in ["a little", "a bit", "slightly", "tiny bit", "small"]):
        return 5
    if any(k in tl for k in ["a lot", "much", "way", "significantly"]):
        return 20
    return 10


def _parse_volume_scope_and_process(clause: str) -> tuple[str, str]:
    """Return (scope, process) where scope is 'master' or 'app'.

    Keep this conservative: only treat a token as an app/process hint when it looks
    like one of the common "<proc> volume" / "volume ... for <proc>" patterns.
    """
    c = (clause or "").strip()
    cl = c.lower()

    # Strip common query prefixes so we don't treat them as app names.
    # Include bare "what" for patterns like "what spotify volume at"
    cl = re.sub(r"^(?:what\s+is|what's|whats|what|get|check|show|tell\s+me|current)\s+", "", cl)
    cl = re.sub(r"^the\s+", "", cl)
    # Strip trailing question words like "at", "at now", etc.
    cl = re.sub(r"\s+(?:at|at now|now|right now)\s*\??\s*$", "", cl)

    # "set <proc> volume ..." should treat <proc> as app.
    m = re.match(r"^set\s+(?P<proc>.+?)\s+(?:volume|sound|audio)\b", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        if proc and proc not in {"the", "my", "this", "that", "volume", "sound", "audio"}:
            return "app", proc

    # "volume 30 for spotify" / "mute discord" style: trailing "for/in/on <proc>".
    m = re.search(r"\b(?:for|in|on)\s+(?P<proc>[a-z0-9][a-z0-9 _\-\.]{1,60})$", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        if proc:
            return "app", proc

    # "spotify volume ..." / "chrome sound ..." (but NOT "set volume ...")
    m = re.match(r"^(?P<first>[a-z0-9][a-z0-9 _\-\.]{1,60})\s+(?:volume|sound|audio)\b", cl)
    if m:
        first = _strip_trailing_punct(m.group("first")).strip()
        if first and first not in {"the", "my", "this", "that", "set", "volume", "sound", "audio"}:
            return "app", first

    # "mute spotify" / "turn down chrome" / "turn down spotify by 10%" etc.
    # Stop at "by", "to", percent sign, or numbers to avoid including delta in process name
    m = re.match(r"^(?:turn\s+(?:up|down)|mute|unmute|quieter|louder)\s+(?P<proc>.+?)(?:\s+(?:by|to)\s+|\s+\d|\s*%|$)", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        proc_l = proc.lower()
        if proc_l in {"it", "it up", "it down", "the volume", "volume", "sound", "audio"}:
            return "master", ""
        if proc and proc not in {"it", "volume", "sound", "audio"}:
            return "app", proc

    # "turn spotify down by 3%" / "turn chrome up" - app name between turn and direction
    m = re.match(r"^turn\s+(?P<proc>[a-z0-9][a-z0-9 _\-\.]{1,60})\s+(?:up|down)\b", cl)
    if m:
        proc = _strip_trailing_punct(m.group("proc")).strip()
        if proc and proc not in {"it", "the", "volume", "sound", "audio"}:
            return "app", proc

    return "master", ""


@dataclass
class HybridDecision:
    mode: Literal["tool_plan", "llm"]
    intents: Optional[List[Dict[str, Any]]] = None
    reply: str = ""
    confidence: float = 0.0


MULTI_INTENT_MARKERS = [
    " and ",
    " then ",
    " after ",
    " before ",
    " while ",
    " also ",
    " plus ",
    " & ",
    ";",
    ",",
]

# Patterns that indicate the query needs LLM reasoning/explanation
# These are questions that can't be answered with simple tool calls
_REASONING_RE = re.compile(
    r"(?:"
    r"^why\s+|"                                    # "why is the sky blue"
    r"^how\s+(?:do|does|can|could|would|should|to)\s+|"  # "how do I...", "how to..."
    r"^what\s+(?:is|are|does|do|should|would|could)\s+(?!the\s+time|the\s+date|my\s+|the\s+weather|the\s+temp|[a-z]+\s+volume)|"  # General "what is X" (but not time/weather/date/volume)
    r"^explain\s+|"                                # "explain how..."
    r"^tell\s+me\s+(?:about|why|how)|"            # "tell me about...", "tell me why..."
    r"^can\s+you\s+(?:explain|help|tell)|"        # "can you explain..."
    r"^should\s+i\s+|"                            # "should I..."
    r"^what\s+(?:if|happens?\s+(?:if|when))|"     # "what if...", "what happens if..."
    r"^is\s+(?:it|this|there)\s+(?:a\s+)?(?:good|bad|better|best|way|possible)|"  # "is it possible...", "is there a better way"
    r"^compare\s+|"                                # "compare X and Y"
    r"^difference\s+between\s+|"                   # "difference between X and Y"
    r"^which\s+(?:is|one|should)|"                 # "which is better", "which one should I"
    r"\bwhy\s+(?:is|are|does|do|did|would|should|can|could)\b|"  # mid-sentence "why is..."
    r"\bhow\s+(?:does|do|is|are|would|should|can|could)\s+(?:it|this|that)\b|"  # "how does this work"
    r"\bexplain(?:ing)?\b|"                        # any "explain" in text
    r"\brecommend(?:ation)?s?\b|"                  # "recommend", "recommendations"
    r"\bsuggestion?s?\b|"                          # "suggest", "suggestions"
    r"\badvice\b|"                                 # "advice"
    r"\bopinion\b|"                                # "opinion"
    r"\bsimilar\s+to\b|"                           # "similar to X"
    r"\b(?:any|some|an?)\s+(?:anime|show|movie|game|book|song|album|series|film)\s+like\b|"  # "any anime like X"
    r"\bwhat(?:'?s|\s+is)\s+(?:a|an|some)\s+(?:anime|show|movie|game|book|song|album|series|film)\s+like\b|"  # "what's an anime like X"
    r"\bhelp\s+me\s+(?:understand|decide|choose|figure)|"  # "help me understand"
    r"\btell\s+me\s+(?:about|why|how)\b"           # mid-sentence "tell me about..."
    r")",
    re.IGNORECASE,
)


# Check if text is a volume query that should bypass reasoning check
def _is_volume_query(text: str) -> bool:
    """Check if text is a volume-related query that should bypass LLM reasoning."""
    tl = (text or "").strip().lower()
    return bool(re.search(r"\b(?:volume|sound|audio|mute|unmute|louder|quieter)\b", tl))


def _is_now_playing_query(text: str) -> bool:
    """Check if text is a now-playing query that should bypass LLM reasoning."""
    tl = (text or "").strip().lower()
    return bool(re.search(
        r"\b(?:what(?:'?s|\s+is)\s+(?:currently\s+)?playing|what\s+(?:song|track|music|media)\s+is\s+(?:this|playing)|now\s+playing|current\s+(?:song|track|media)|playing\s+(?:right\s+)?now)\b",
        tl
    ))


def needs_reasoning(text: str) -> bool:
    """Check if text requires LLM reasoning/explanation rather than tool execution."""
    tl = (text or "").strip()
    if not tl:
        return False
    # Volume queries should never need reasoning - they're simple tool calls
    if _is_volume_query(tl):
        return False
    # Now playing queries should never need reasoning
    if _is_now_playing_query(tl):
        return False
    return bool(_REASONING_RE.search(tl))


def looks_multi_intent(text: str) -> bool:
    tl = (text or "").strip().lower()
    if not tl:
        return False

    # Strip internal punctuation (commas, hyphens) to avoid treating speech recognition
    # artifacts as multi-intent markers (e.g., "Scan, discy." should not split on comma)
    tl = re.sub(r'[,\-]', ' ', tl).replace('  ', ' ')

    # Normalize whitespace to make marker checks more reliable.
    tl = re.sub(r"\s+", " ", tl)
    tl = f" {tl} "

    # Check explicit markers
    if any(marker in tl for marker in MULTI_INTENT_MARKERS):
        return True
    
    # Check for implicit verb boundaries: "verb1 target1 verb2 target2"
    # E.g., "close chrome open spotify" should be detected as 2 intents
    # Verbs that commonly start new intents (from multi_intent_parser.py)
    action_verbs = r"(?:open|launch|start|close|quit|exit|minimize|shrink|maximize|fullscreen|expand|move|send|play|pause|resume|mute|unmute|scan)"
    verb_matches = list(re.finditer(action_verbs, tl, re.IGNORECASE))
    if len(verb_matches) >= 2:
        return True
    
    return False


# Anchored time patterns: only match whole-utterance variants.
_TIME_RE = re.compile(
    r"^(what\s+(?:time|'s\s+the\s+time)|what's\s+the\s+time|time\s+is\s+it|current\s+time|what\s+time\s+is\s+it)\??$",
    re.IGNORECASE,
)

# Weather patterns: match queries about weather, temperature, forecast
_WEATHER_RE = re.compile(
    r"\b(?:"
    r"weather|"
    r"temperature|"
    r"temp|"
    r"forecast|"
    r"how\s+(?:cold|hot|warm)|"
    r"what.{0,10}(?:weather|temperature|temp|forecast)|"
    r"(?:weather|temperature|forecast)\s+(?:in|for|at)|"
    r"is\s+it\s+(?:cold|hot|warm|raining|snowing)|"
    r"will\s+it\s+(?:rain|snow)|"
    r"what.{0,10}like\s+outside"
    r")\b",
    re.IGNORECASE,
)

# System info patterns: queries about system specs, CPU, RAM, hardware
_SYSTEM_INFO_RE = re.compile(
    r"^(?:"
    r"(?:get\s+)?(?:my\s+)?system\s+(?:info|information|specs|specifications)|"
    r"(?:tell\s+)?me\s+about\s+(?:my\s+)?system|"
    r"what\s+(?:are|is)(?:\s+my)?\s+system\s+(?:specs|specifications)|"
    r"how\s+much\s+(?:ram|memory)\s+do\s+i\s+have|"
    r"what.?s\s+my\s+(?:cpu|processor|system)|"
    r"system\s+information|"
    r"computer\s+specs|"
    r"hardware\s+info|"
    r"about\s+this\s+computer"
    r").*$",
    re.IGNORECASE,
)

# Location/IP patterns: queries about user's location, IP address, timezone
_LOCATION_RE = re.compile(
    r"(?:"
    r"(?:what|where).{0,10}(?:my\s+)?(?:ip|location|address|timezone|time\s+zone|country|city|coordinates)|"
    r"where\s+(?:am\s+i|is\s+(?:my\s+)?(?:device|computer))|"
    r"what\s+(?:is\s+)?my\s+(?:location|ip|address|timezone|time\s+zone|country|city)|"
    r"get\s+(?:my\s+)?(?:location|ip|address|timezone|time\s+zone|country|city)|"
    r"tell\s+(?:me\s+)?(?:my|where)\s+(?:location|ip|address|timezone|time\s+zone|country|city)|"
    r"i\s+am\s+in|"
    r"what\s+(?:country|city|timezone|time\s+zone)\s+(?:am\s+i|is\s+(?:my|i))"
    r")",
    re.IGNORECASE,
)

# Anchored open/launch/start.
_OPEN_RE = re.compile(r"^(open|launch|start)\s+(.+)$", re.IGNORECASE)

# Anchored close/quit/exit.
_CLOSE_RE = re.compile(r"^(close|quit|exit)\s+(.+)$", re.IGNORECASE)

# Anchored minimize/shrink.
_MINIMIZE_RE = re.compile(r"^(minimize|shrink)\s+(.+)$", re.IGNORECASE)

# Anchored maximize/fullscreen/expand.
_MAXIMIZE_RE = re.compile(r"^(maximize|fullscreen|expand|full\s+screen)\s+(.+)$", re.IGNORECASE)

# Anchored audio device switching: "switch/set/change/swap audio [output] [device] to <device>"
_AUDIO_DEVICE_SWITCH_RE = re.compile(
    r"^(?:(?:switch|set|change|swap)\s+(?:audio(?:\s+output)?|sound|output)(?:\s+device)?\s+to)\s+(.+)$",
    re.IGNORECASE,
)

# Anchored audio device listing: "list audio devices", "show audio devices", etc.
_AUDIO_DEVICE_LIST_RE = re.compile(
    r"^(?:list|show|display|what)(?:\s+(?:audio|sound))?(?:\s+(?:output\s+)?devices?|devices?|speakers?)?(?:\s+are\s+available)?\??$",
    re.IGNORECASE,
)

# Word-to-digit mapping for monitor numbers
_WORD_TO_DIGIT = {
    "one": "1", "two": "2", "three": "3", "four": "5", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5",
    "secondary": "2", "other": "2",
}

# Anchored move window to monitor.
# Supports: "move X to monitor 2", "move X to primary monitor", "move X to the second monitor"
_MOVE_MONITOR_RE = re.compile(
    r"^(?:move|send)\s+(.+?)\s+to\s+(?:(?:the\s+)?(?:monitor|screen)\s+)?"
    r"(primary|main|secondary|other|\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|next|previous|left|right)"
    r"(?:\s+(?:monitor|screen))?$",
    re.IGNORECASE
)

# Get window monitor: "what monitor is X on" / "which screen is X on"
_GET_WINDOW_MONITOR_RE = re.compile(
    r"^(?:what|which)\s+(?:monitor|screen|display)\s+is\s+(.+?)\s+(?:on|displayed\s+on|showing\s+on)\??$",
    re.IGNORECASE,
)

# Monitor info patterns: queries about connected monitors (count, resolution, scanning)
# NOTE: Be careful not to match "what monitor is X on" which asks about app location
_MONITOR_INFO_RE = re.compile(
    r"^(?:"
    r"(?:scan|check|list|show|display)\s+(?:my\s+)?(?:monitors?|screens?|displays?)|"  # scan monitors, check monitors
    r"(?:how\s+many)\s+(?:monitors?|screens?|displays?)\s+(?:do\s+i\s+have|are\s+(?:there|connected|available))|"  # how many monitors do i have
    r"(?:get|show|check|tell\s+me)\s+(?:my\s+)?(?:monitor|screen|display)\s+(?:info|information|details)|"  # get monitor info
    r"(?:monitor|screen|display)\s+(?:info|information|details|count|resolution|status)|"  # monitor info, monitor count
    r"(?:what|which)\s+(?:monitors?|screens?|displays?)\s+(?:do\s+i\s+have|are\s+connected|available)|"  # what monitors do i have
    r"(?:list|show|display)\s+(?:all\s+)?(?:my\s+)?(?:monitors?|screens?|displays?)|"  # list all monitors
    r"(?:what)\s+(?:are\s+)?(?:my\s+)?(?:monitors?|screens?|displays?)\s*\??"  # what are my monitors (ends query)
    r")$",
    re.IGNORECASE,
)

# Conservative URL/domain detection: if it looks like a URL/domain, we force LLM.
_URL_SCHEME_RE = re.compile(r"\bhttps?://", re.IGNORECASE)
_WWW_RE = re.compile(r"\bwww\.", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"\b[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.(?:[a-z]{2,})(?:/[^\s]*)?\b",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Timer patterns: set, cancel, or check a countdown timer
# ═══════════════════════════════════════════════════════════════════════════

# Word-to-number mapping for timer durations
_WORD_TO_NUMBER = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "a": 1, "an": 1,  # "a minute", "an hour"
}

# Timer start: "set a timer for X minutes/seconds", "start a timer for X minutes"
# Matches both digits (10) and word numbers (ten, three)
# Also supports compound durations like "4 minutes and 20 seconds"
# Note: Speech recognition often transcribes "a timer" as "our timer", "our timings", "the timer", etc.
_TIMER_NUMBER_PATTERN = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|a|an)"
_TIMER_UNIT_PATTERN = r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)"

# Simple single-unit timer: "set a timer for 5 minutes"
_TIMER_START_RE = re.compile(
    rf"^(?:set|start|create)\s+(?:a|an|the|our|up\s+a|up\s+the)?\s*(?:timer|timers|timing|timings)\s+(?:for\s+)?{_TIMER_NUMBER_PATTERN}\s*{_TIMER_UNIT_PATTERN}(?:\s+timer)?[.?!]?$",
    re.IGNORECASE,
)

# Compound duration timer: "set a timer for 4 minutes and 20 seconds"
# Supports: "X hours Y minutes Z seconds", "X minutes and Y seconds", etc.
_TIMER_COMPOUND_RE = re.compile(
    rf"^(?:set|start|create)\s+(?:a|an|the|our|up\s+a|up\s+the)?\s*(?:timer|timers|timing|timings)\s+(?:for\s+)?{_TIMER_NUMBER_PATTERN}\s*{_TIMER_UNIT_PATTERN}(?:\s+(?:and\s+)?{_TIMER_NUMBER_PATTERN}\s*{_TIMER_UNIT_PATTERN})?(?:\s+(?:and\s+)?{_TIMER_NUMBER_PATTERN}\s*{_TIMER_UNIT_PATTERN})?(?:\s+timer)?[.?!]?$",
    re.IGNORECASE,
)

# Timer cancel: "cancel the timer", "stop the timer", "clear timer"
_TIMER_CANCEL_RE = re.compile(
    r"^(?:cancel|stop|clear|end|delete|remove)\s+(?:the\s+|my\s+)?timer[.?!]?$",
    re.IGNORECASE,
)

# Timer status: "how much time is left", "timer status", "check timer"
_TIMER_STATUS_RE = re.compile(
    r"^(?:"
    r"(?:how\s+much\s+)?time\s+(?:is\s+)?(?:left|remaining)(?:\s+on\s+(?:the\s+|my\s+)?timer)?|"
    r"(?:check|get|show)\s+(?:the\s+|my\s+)?timer(?:\s+status)?|"
    r"timer\s+status|"
    r"what(?:'s|\s+is)\s+(?:the\s+|my\s+)?timer(?:\s+at)?|"
    r"how\s+long\s+(?:is\s+)?(?:left|remaining)(?:\s+on\s+(?:the\s+|my\s+)?timer)?"
    r")[.?!]?$",
    re.IGNORECASE,
)


def _parse_timer_value(value_str: str) -> int:
    """Parse a timer value from string (digit or word) to integer."""
    value_str = value_str.strip().lower()
    # Try as digit first
    if value_str.isdigit():
        return int(value_str)
    # Try word lookup
    return _WORD_TO_NUMBER.get(value_str, 0)


def _parse_timer_duration_seconds(value: int, unit: str) -> int:
    """Convert a timer duration value and unit to seconds."""
    unit_lower = unit.lower().rstrip("s")  # normalize: "minutes" -> "minute"
    if unit_lower in {"sec", "second"}:
        return value
    elif unit_lower in {"min", "minute"}:
        return value * 60
    elif unit_lower in {"hour", "hr"}:
        return value * 3600
    return value  # default to seconds


def _parse_compound_timer_duration(text: str) -> int:
    """
    Parse compound timer durations like "4 minutes and 20 seconds" into total seconds.
    
    Supports:
    - "4 minutes and 20 seconds" -> 260
    - "1 hour 30 minutes" -> 5400
    - "2 hours and 15 minutes and 30 seconds" -> 8130
    - "5 minutes" -> 300 (single unit still works)
    """
    text_lower = text.lower()
    total_seconds = 0
    
    # Pattern to find all number+unit pairs
    pattern = re.compile(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|a|an)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)",
        re.IGNORECASE
    )
    
    for match in pattern.finditer(text_lower):
        value = _parse_timer_value(match.group(1))
        unit = match.group(2)
        total_seconds += _parse_timer_duration_seconds(value, unit)
    
    return total_seconds


def _looks_like_url_or_domain(text: str) -> bool:
    tl = (text or "").strip().lower()
    if not tl:
        return False

    if _URL_SCHEME_RE.search(tl) or _WWW_RE.search(tl):
        return True

    # Any obvious domain-like token (foo.com, foo.co.uk, etc.).
    if _DOMAIN_RE.search(tl):
        return True

    return False


def _decide_single_clause(text: str) -> HybridDecision:
    clause = (text or "").strip()
    if not clause:
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.0)

    # Safety: URLs/domains go to LLM.
    if _looks_like_url_or_domain(clause):
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.8)

    # Time queries.
    if _TIME_RE.match(clause.strip()):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_time", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Timer queries: start, cancel, or check status
    # ═══════════════════════════════════════════════════════════════════════
    
    # Timer start: "set a timer for 5 minutes" or "set a timer for 4 minutes and 20 seconds"
    # Try compound pattern first (it handles both compound and simple cases)
    m = _TIMER_COMPOUND_RE.match(clause)
    if m:
        duration_seconds = _parse_compound_timer_duration(clause)
        if duration_seconds < 1:
            duration_seconds = 1  # Minimum 1 second
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "timer",
                "args": {"action": "start", "duration_seconds": duration_seconds},
                "continue_on_error": False
            }],
            reply="",
            confidence=0.95,
        )
    
    # Timer cancel: "cancel the timer"
    if _TIMER_CANCEL_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "timer",
                "args": {"action": "cancel"},
                "continue_on_error": False
            }],
            reply="",
            confidence=0.92,
        )
    
    # Timer status: "how much time is left"
    if _TIMER_STATUS_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "timer",
                "args": {"action": "status"},
                "continue_on_error": False
            }],
            reply="",
            confidence=0.92,
        )

    # Weather queries - extract location if provided
    if _WEATHER_RE.search(clause):
        clause_lower = clause.lower()
        # Try to extract location from the query
        location = None
        
        # Weekday names for temporal parsing
        weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        
        # Pattern: "weather in <location>" / "temperature in <location>" / "forecast for <location>"
        m = re.search(r"\b(?:in|for|at|near|on)\s+(.+?)(?:\?|$)", clause, re.IGNORECASE)
        if m:
            location = (m.group(1) or "").strip().rstrip("?").strip()
            # Filter out common non-location words and temporal words
            location_l = location.lower()
            temporal_words = {"it", "this", "here", "there", "the area", "outside", "tomorrow", "today", "the week", "this week", "next week", "weather", "the weather", "forecast", "the forecast"}
            temporal_words.update(weekday_names)
            if location_l in temporal_words:
                location = None
        
        # Extract temporal reference (tomorrow, this week, weekday names, etc.)
        day_offset = 0  # 0 = today
        days = 3  # default forecast days
        
        if "tomorrow" in clause_lower:
            day_offset = 1
            days = 2  # Include today + tomorrow
        elif re.search(r"\bnext\s+week\b", clause_lower):
            days = 14
            day_offset = 7
        elif re.search(r"\b(?:this\s+)?week(?:ly)?\b", clause_lower):
            days = 7
        else:
            # Check for weekday names (e.g., "on Thursday", "this Friday", "next Monday")
            import datetime
            today = datetime.date.today()
            today_weekday = today.weekday()  # 0=Monday, 6=Sunday
            
            # Check if user said "next <weekday>" (meaning next week's occurrence)
            next_week_match = re.search(r"\bnext\s+(" + "|".join(weekday_names) + r")\b", clause_lower)
            
            for i, day_name in enumerate(weekday_names):
                if day_name in clause_lower:
                    # Calculate days until that weekday
                    target_weekday = i  # 0=Monday, 6=Sunday
                    days_until = (target_weekday - today_weekday) % 7
                    
                    if next_week_match and next_week_match.group(1) == day_name:
                        # "next Thursday" means next week's Thursday
                        days_until = days_until + 7 if days_until > 0 else 7
                    elif days_until == 0:
                        # Same day of week - could be today or next week
                        # If they say "this Monday" it's today, otherwise assume next week
                        if "this" in clause_lower:
                            days_until = 0
                        else:
                            days_until = 7  # Next occurrence
                    
                    day_offset = days_until
                    days = max(day_offset + 1, 7)  # Ensure we fetch enough days
                    break
        
        # Build arguments
        weather_args = {}
        if location:
            weather_args["location"] = location
        if day_offset > 0:
            weather_args["day_offset"] = day_offset
        if days != 3:
            weather_args["days"] = days
        
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "get_weather_forecast",
                    "args": weather_args,
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.92,
        )

    # System info queries: queries about system specs, CPU, RAM, hardware
    if _SYSTEM_INFO_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_system_info", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.9,
        )

    # Location/IP queries: queries about user's location, IP address, timezone
    if _LOCATION_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_location", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.9,
        )

    # Get window monitor queries: "what monitor is X on"
    m = _GET_WINDOW_MONITOR_RE.match(clause)
    if m:
        target = (m.group(1) or "").strip().strip('"').strip("'")
        if target and target.lower() not in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "get_window_monitor", "args": {"process": target}, "continue_on_error": False}],
                reply="",
                confidence=0.92,
            )

    # Monitor info queries: queries about connected monitors
    if _MONITOR_INFO_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "monitor_info", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.92,
        )

    # Local library refresh/scan commands
    tl_norm = _strip_trailing_punct(clause).lower()
    
    # "scan files", "scan my files", "scan apps", "scan my apps" -> tier 3 (full file system scan)
    if re.match(r"^scan\s+(?:my\s+)?(?:files|apps?)$", tl_norm):
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "local_library_refresh",
                    "args": {"mode": "tier3"},
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.92,
        )
    
    # "refresh library", "rebuild library" -> normal mode
    if re.match(r"^(?:refresh|rebuild|rescan)\s+library$", tl_norm):
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "local_library_refresh",
                    "args": {},
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.93,
        )

    # System storage commands (check before generic open pattern).
    tl = _strip_trailing_punct(clause).lower()
    # Remove internal punctuation (commas, hyphens, periods) for more flexible matching
    tl_normalized = re.sub(r'[,\-.\']', ' ', tl).replace('  ', ' ').strip()
    
    # "scan devices" -> deep tier (full file system scan)
    if re.match(r"^scan\s+devices?$", tl_normalized):
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "system_storage_scan",
                    "args": {"tier": "deep"},
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.92,
        )
    
    # "scan drive c", "scan d", "scan disc e" etc -> deep tier for specific drive
    m = re.search(r"\bscan\s*(?:hard\s+)?(?:drive|disc|disk)?\s*([a-z])\b|^scandisk([a-z])$", tl_normalized)
    if m:
        drive_letter = (m.group(1) or m.group(2) or "").upper()
        if drive_letter:
            return HybridDecision(
                mode="tool_plan",
                intents=[
                    {
                        "tool": "system_storage_scan",
                        "args": {"tier": "deep", "drive": drive_letter},
                        "continue_on_error": False,
                    }
                ],
                reply="",
                confidence=0.92,
            )
    
    # "system scan" / "scan my drives" / "refresh drive index" / "scan disc" / "scan discs" / "scan discy"
    if re.match(r"^(?:system\s+scan|scan\s+(?:my\s+)?drives?|scan\s+dis(?:c|k)[ys]?|refresh\s+drive\s+index)$", tl_normalized):
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "system_storage_scan",
                    "args": {"refresh": True},
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.95,
        )

    # "list drives" / "show drives" / "how much space do i have" / "storage summary"
    if re.search(r"\b(?:list\s+drives|show\s+drives|how\s+much\s+space\s+do\s+i\s+have|storage\s+summary)\b", tl_normalized):
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "system_storage_list",
                    "args": {},
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.92,
        )

    # "how much space does d drive have" / "space on d drive" / "how much storage is on d" / "what on e" / "what does e have" / "how much storage do i have on c" / "what does edrive have"
    m = re.search(r"(?:what\s+does\s+|what\s+(?:is\s+)?on|how\s+much\s+(?:space|storage)(?:\s+(?:is|do\s+i\s+have))?\s+on|space\s+on|storage\s+on)\s*(?:drive\s+)?([a-z])|(?:what\s+does\s+)?([a-z])drive(?:\s+have)?|(?:space|storage)\s+on\s+([a-z])", tl_normalized)
    if m:
        drive_letter = m.group(1) or m.group(2) or m.group(3)
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "system_storage_list",
                    "args": {"drive": drive_letter},
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.91,
        )

    # "open d drive" / "open drive d" / "open hard drive d" / "open d:" / "open /mnt/storage" / "open d" (single letter)
    m = re.match(r"^open\s+(?:hard\s+)?(?:drive\s+)?([a-z]|[a-z]:|/[a-z0-9/_\-]+)(?:\s+drive)?$", tl)
    if m:
        drive_token = m.group(1)
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "system_storage_open",
                    "args": {"drive": drive_token},
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.93,
        )

    # Open/launch/start X (non-URL) -> open_target.
    clause_stripped = _strip_trailing_punct(clause)
    m = _OPEN_RE.match(clause_stripped)
    if m:
        target = (m.group(2) or "").strip().strip('"').strip("'")
        # If the target is missing or too ambiguous, defer to LLM.
        if not target or target.lower() in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)

        # Double-check: the extracted target itself may look like a URL.
        if _looks_like_url_or_domain(target):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.8)

        # Extra defense: if target includes other action verbs, defer to LLM.
        target_l = re.sub(r"\s+", " ", target.lower()).strip()
        if any(v in target_l.split() for v in ["play", "pause", "resume", "then", "and", "also", "plus"]):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.3)

        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "open_target",
                    "args": {"query": target},
                    "continue_on_error": False,
                }
            ],
            reply=f"Opening {target}.",
            confidence=0.9,
        )

    # Close/quit/exit X -> close_window.
    m = _CLOSE_RE.match(clause_stripped)
    if m:
        target = (m.group(2) or "").strip().strip('"').strip("'")
        # If the target is missing or too ambiguous, defer to LLM.
        if not target or target.lower() in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)

        # Extra defense: if target includes other action verbs, defer to LLM.
        target_l = re.sub(r"\s+", " ", target.lower()).strip()
        if any(v in target_l.split() for v in ["play", "pause", "resume", "then", "and", "also", "plus"]):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.3)

        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "close_window",
                    "args": {"title": target},
                    "continue_on_error": False,
                }
            ],
            reply=f"Closing {target}.",
            confidence=0.85,

        )

    # Minimize/shrink X -> minimize_window.
    m = _MINIMIZE_RE.match(clause_stripped)
    if m:
        target = (m.group(2) or "").strip().strip('"').strip("'")
        # If the target is missing or too ambiguous, defer to LLM.
        if not target or target.lower() in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)

        # Extra defense: if target includes other action verbs, defer to LLM.
        target_l = re.sub(r"\s+", " ", target.lower()).strip()
        if any(v in target_l.split() for v in ["play", "pause", "resume", "then", "and", "also", "plus"]):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.3)

        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "minimize_window",
                    "args": {"title": target},
                    "continue_on_error": False,
                }
            ],
            reply=f"Minimizing {target}.",
            confidence=0.85,
        )

    # Maximize/fullscreen/expand X -> maximize_window.
    m = _MAXIMIZE_RE.match(clause_stripped)
    if m:
        target = (m.group(2) or "").strip().strip('"').strip("'")
        # If the target is missing or too ambiguous, defer to LLM.
        if not target or target.lower() in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)

        # Extra defense: if target includes other action verbs, defer to LLM.
        target_l = re.sub(r"\s+", " ", target.lower()).strip()
        if any(v in target_l.split() for v in ["play", "pause", "resume", "then", "and", "also", "plus"]):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.3)

        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "maximize_window",
                    "args": {"title": target},
                    "continue_on_error": False,
                }
            ],
            reply=f"Maximizing {target}.",
            confidence=0.85,
        )

    # Move window to monitor: "move X to monitor 2" / "send chrome to monitor next"
    m = _MOVE_MONITOR_RE.match(clause_stripped)
    if m:
        target = (m.group(1) or "").strip().strip('"').strip("'")
        monitor = (m.group(2) or "").strip().lower()
        
        # Convert word numbers to digits
        monitor = _WORD_TO_DIGIT.get(monitor, monitor)
        
        # If the target is missing or too ambiguous, defer to LLM.
        if not target or target.lower() in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)

        # Extra defense: if target includes other action verbs, defer to LLM.
        target_l = re.sub(r"\s+", " ", target.lower()).strip()
        if any(v in target_l.split() for v in ["play", "pause", "resume", "then", "and", "also", "plus"]):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.3)

        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "move_window_to_monitor",
                    "args": {"title": target, "monitor": monitor},
                    "continue_on_error": False,
                }
            ],
            reply=f"Moving {target} to monitor {monitor}.",
            confidence=0.85,
        )

    # Audio device listing: "list audio devices" / "show audio devices"
    tl_audio = _strip_trailing_punct(clause).lower()
    if _AUDIO_DEVICE_LIST_RE.match(tl_audio):
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "set_audio_output_device",
                    "args": {"action": "list"},
                    "continue_on_error": False,
                }
            ],
            reply="",
            confidence=0.92,
        )

    # Audio device switching: "switch audio to vizio" / "set audio to headphones"
    m = _AUDIO_DEVICE_SWITCH_RE.match(clause_stripped)
    if m:
        device = (m.group(1) or "").strip().strip('"').strip("'")
        
        # If the device is missing or too ambiguous, defer to LLM.
        if not device or device.lower() in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)
        
        return HybridDecision(
            mode="tool_plan",
            intents=[
                {
                    "tool": "set_audio_output_device",
                    "args": {"action": "set", "device": device},
                    "continue_on_error": False,
                }
            ],
            reply=f"Switching audio to {device}.",
            confidence=0.9,
        )

    # Minimal media/volume controls (only if tools exist; existence checked upstream).
    tl = clause.lower()

    # --- True volume control (pycaw) ---
    # If the command looks like volume/mute and the tool exists, prefer volume_control.
    # We keep this conservative and only match obvious phrases.
    # Match: "turn down X", "turn X down", "turn it up", etc.
    if re.search(r"\b(?:mute|unmute|volume|sound|audio|louder|quieter|turn\s+(?:it\s+)?(?:up|down)|turn\s+\w+\s+(?:up|down))\b", tl):
        scope, proc = _parse_volume_scope_and_process(clause)

        # Get volume / what is the volume
        # Expanded query detection: explicit query words OR "what ... volume" patterns OR bare "<app> volume" queries
        is_explicit_query = bool(re.search(r"\b(?:get|check|show|tell\s+me|what\s+is|what's|whats|current)\b", tl))
        is_what_volume = bool(re.search(r"\bwhat\b.*\bvolume\b", tl))  # "what spotify volume at"
        is_volume_worded = any(k in tl for k in ["volume", "sound", "audio"])
        has_action_word = bool(re.search(r"\b(?:set|up|down|louder|quieter|increase|decrease|raise|lower|mute|unmute)\b", tl))
        has_percent = _extract_volume_percent(tl) is not None
        
        # Bare volume query: "<app> volume" or "volume" with no action/percent = asking for current level
        is_bare_volume_query = is_volume_worded and not has_action_word and not has_percent
        is_query = is_explicit_query or is_what_volume or is_bare_volume_query
        
        if is_query and is_volume_worded and not has_percent and not re.search(
            r"\b(?:up|down|louder|quieter|increase|decrease|raise|lower)\b", tl
        ):
            args: Dict[str, Any] = {"scope": scope, "action": "get"}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.92,
            )

        # Mute/unmute
        if re.search(r"\bunmute\b", tl):
            args = {"scope": scope, "action": "unmute"}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.93,
            )

        if re.search(r"\bmute\b", tl) and not re.search(r"\bunmute\b", tl):
            args = {"scope": scope, "action": "mute"}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.93,
            )

        # Absolute set: "volume 35" / "set volume to 35" / "spotify volume 35"
        if is_volume_worded:
            percent = _extract_volume_percent(tl)
            # Avoid interpreting "volume down 10" as set-to.
            has_direction = bool(re.search(r"\b(?:up|down|increase|decrease|raise|lower|louder|quieter)\b", tl))
            if percent is not None and not has_direction:
                args = {"scope": scope, "action": "set", "level": int(percent)}
                if scope == "app":
                    args["process"] = proc
                return HybridDecision(
                    mode="tool_plan",
                    intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                    reply="",
                    confidence=0.9,
                )

        # Relative change: up/down/louder/quieter, optional numeric delta.
        # Match: "volume up", "turn up X", "turn X up", "louder", etc.
        if re.search(r"\b(?:volume\s+up|turn\s+up|louder|raise|increase|sound\s+up)\b", tl) or re.search(
            r"\bturn\s+(?:it\s+)?up\b|\bturn\s+\w+\s+up\b", tl
        ):
            pct = _extract_volume_percent(tl)
            delta = int(pct) if pct is not None else _parse_volume_delta_hint(tl)
            args = {"scope": scope, "action": "change", "delta": int(delta)}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.88,
            )

        # Match: "volume down", "turn down X", "turn X down", "quieter", etc.
        if re.search(r"\b(?:volume\s+down|turn\s+down|quieter|lower|decrease|sound\s+down)\b", tl) or re.search(
            r"\bturn\s+(?:it\s+)?down\b|\bturn\s+\w+\s+down\b", tl
        ):
            pct = _extract_volume_percent(tl)
            delta = int(pct) if pct is not None else _parse_volume_delta_hint(tl)
            args = {"scope": scope, "action": "change", "delta": -int(delta)}
            if scope == "app":
                args["process"] = proc
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "volume_control", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.88,
            )

    if re.search(r"\b(?:mute|unmute)\b", tl):
        # Fallback for older setups without volume_control.
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "volume_mute_toggle", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.9,
        )

    if re.search(r"\b(?:volume\s+up|turn\s+up|louder)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "volume_up", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.85,
        )

    if re.search(r"\b(?:volume\s+down|turn\s+down|quieter)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "volume_down", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.85,
        )

    # "What's playing", "what song is playing", "what is currently playing", "now playing"
    if re.search(r"\b(?:what(?:'?s|\s+is)\s+(?:currently\s+)?playing|what\s+(?:song|track|music|media)\s+is\s+(?:this|playing)|now\s+playing|current\s+(?:song|track|media)|playing\s+(?:right\s+)?now)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_now_playing", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.9,
        )

    if re.search(r"\b(?:play\s*pause|play/pause|pause|play|resume)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "media_play_pause", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.8,
        )

    if re.search(r"\b(?:next\s+track|skip|next\s+song|next\s+video|next\s+media)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "media_next", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.85,
        )

    if re.search(r"\b(?:previous\s+track|back|prior\s+track|last\s+song|previous\s+song|previous\s+video|previous\s+media|go\s+back)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "media_previous", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.85,
        )

    return HybridDecision(mode="llm", intents=None, reply="", confidence=0.3)


def decide(text: str) -> HybridDecision:
    """Decide whether to run tools deterministically or use the LLM.

    Args:
        text: Raw user text

    Returns:
        HybridDecision with mode tool_plan or llm.
    """
    raw = (text or "").strip()
    if not raw:
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.0)

    # Check for reasoning/explanation questions first - these always go to LLM
    if needs_reasoning(raw):
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.85)

    # Try multi-intent parsing: handle "open steam and chrome" directly without LLM
    if looks_multi_intent(raw):
        try:
            from wyzer.core.multi_intent_parser import parse_multi_intent_with_fallback
            result = parse_multi_intent_with_fallback(raw)
            if result is not None:
                intents, confidence = result
                return HybridDecision(mode="tool_plan", intents=intents, reply="", confidence=confidence)
        except Exception:
            # If multi-intent parser fails, fall back to LLM
            pass
        
        # If multi-intent parsing didn't work, use LLM for reasoning/splitting
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.2)

    return _decide_single_clause(raw)
