"""wyzer.core.hybrid_router

Hybrid deterministic router for obvious commands.

It returns either:
- a deterministic tool plan (one or more tool calls), OR
- a decision to use the LLM.

This module is intentionally conservative.
"""

from __future__ import annotations

import datetime as _datetime_mod
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple, TYPE_CHECKING

_datetime_date = _datetime_mod.date

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from wyzer.core.multi_intent_parser import parse_multi_intent_with_fallback


def _strip_trailing_punct(text: str) -> str:
    return (text or "").strip().rstrip(".?!,;:\"'")


_ROUTING_NORMALIZE_PUNCT_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)


def _normalize_text_for_routing(text: str) -> str:
    """Normalize text for deterministic routing checks.

    Keep this conservative: we only use it for lightweight keyword detection,
    not for the primary anchored command regexes.
    """
    tl = (text or "").strip().lower()
    if not tl:
        return ""
    # Remove apostrophes first so contractions stay joined: "what's" -> "whats"
    tl = tl.replace("'", "").replace("\u2019", "")
    # Replace remaining punctuation with spaces so multi-sentence utterances can be scanned.
    tl = _ROUTING_NORMALIZE_PUNCT_RE.sub(" ", tl)
    tl = re.sub(r"\s+", " ", tl).strip()
    if not tl:
        return ""
    # Collapse adjacent repeats: "time time time" -> "time time" -> "time"
    tokens = tl.split()
    collapsed: List[str] = []
    for tok in tokens:
        if not collapsed or collapsed[-1] != tok:
            collapsed.append(tok)
    return " ".join(collapsed)


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


def _parse_volume_scope_and_process(clause: str) -> Tuple[str, str]:
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
    # Phase 17: True when the orchestrator MUST run perception before acting.
    _needs_perception: bool = False


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
    r"^what\s+(?:is|are|does|do|should|would|could)\s+(?!the\s+time|the\s+date|today\b|today\s*(?:'s)?\s+date|the\s+day\b|day\s+of\s+the\s+week|my\s+|the\s+weather|the\s+temp|[a-z]+\s+volume)|"  # General "what is X" (but not time/weather/date/day/volume)
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

# Conversational questions about past actions/history - route to LLM, not tools
# These are questions asking about what has been done, not requests to do something
_HISTORY_QUESTION_RE = re.compile(
    r"(?:"
    r"^have\s+(?:i|you|we)\s+|"                    # "have I...", "have you...", "have we..."
    r"^did\s+(?:i|you|we)\s+|"                     # "did I...", "did you...", "did we..."
    r"^has\s+(?:anything|something|it)\s+|"        # "has anything...", "has something..."
    r"^(?:what|which)\s+(?:have|did)\s+(?:i|you|we)\s+|"  # "what have I...", "what did you..."
    r"^(?:when|where)\s+did\s+(?:i|you|we)\s+|"    # "when did I...", "where did you..."
    r"^(?:how\s+many|how\s+much)\s+(?:have|did)\s+|"  # "how many have...", "how much did..."
    r"\byet\s*\??\s*$|"                            # ends with "yet?" - usually asking about past
    r"\balready\b.*\?|"                            # contains "already" with question mark
    r"\bso\s+far\b.*\?"                            # "so far" with question mark
    r")",
    re.IGNORECASE,
)

# Creative content patterns: stories, poems, jokes, narratives
# These should ALWAYS route to LLM reply-only, never attempt tool execution
_CREATIVE_CONTENT_RE = re.compile(
    r"(?:"
    r"(?:tell|write|create|make|give)\s+(?:me\s+)?(?:a\s+)?(?:short\s+)?(?:story|tale|narrative|poem|joke|limerick|haiku)|"  # "tell me a story"
    r"^(?:a\s+)?story\s+(?:about|of|for)|"         # "a story about..."
    r"\bstory\s+(?:about|of|for|featuring|with)\b|"  # "story about X"
    r"^once\s+upon\s+a\s+time|"                    # fairy tale prompt
    r"\b(?:creative|fictional|fantasy)\s+(?:story|tale|narrative)|"  # "creative story"
    r"\bwrite\s+(?:about|something|me)\b|"         # "write about..."
    r"\bcompose\s+(?:a\s+)?(?:story|poem|song|lyrics)|"  # "compose a poem"
    r"\bmake\s+up\s+(?:a\s+)?(?:story|tale)|"      # "make up a story"
    r"\bimagine\s+(?:a\s+)?(?:story|scenario|world)|"  # "imagine a story"
    r"\binvent\s+(?:a\s+)?(?:story|tale)"          # "invent a story"
    r")",
    re.IGNORECASE,
)


def _is_creative_request(text: str) -> bool:
    """Check if text is a creative content request (story, poem, joke, etc.)."""
    tl = (text or "").strip()
    if not tl:
        return False
    return bool(_CREATIVE_CONTENT_RE.search(tl))


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
    # Creative content requests should always need LLM reply-only (no tools)
    if _is_creative_request(tl):
        return True
    # Volume queries should never need reasoning - they're simple tool calls
    if _is_volume_query(tl):
        return False
    # Now playing queries should never need reasoning
    if _is_now_playing_query(tl):
        return False
    # History/past-action questions need LLM conversational response
    if _HISTORY_QUESTION_RE.search(tl):
        return True
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
    # Use word boundaries to avoid matching substrings like "play" in "playback"
    action_verbs = r"\b(?:open|launch|start|close|quit|exit|minimize|shrink|maximize|fullscreen|expand|move|send|play|pause|resume|mute|unmute|scan|switch|focus|go)\b"
    verb_matches = list(re.finditer(action_verbs, tl, re.IGNORECASE))
    if len(verb_matches) >= 2:
        return True
    
    return False


# Anchored time patterns: only match whole-utterance variants.
_TIME_RE = re.compile(
    r"^(?:what\s+time(?:\s+is\s+it)?|(?:what\s+s|whats|what'?s)\s+the\s+time|time(?:\s+is\s+it)?|current\s+time|the\s+time)\??$",
    re.IGNORECASE,
)

# Time keywords inside longer utterances (e.g., "Time. What time is it?").
# Keep this narrowly focused on *asking the current time*, not conceptual "what is time".
_TIME_KEYWORDS_ANYWHERE_RE = re.compile(
    r"\b(?:what\s+time(?:\s+is\s+it)?|(?:what\s+s|whats|what'?s)\s+the\s+time|current\s+time|time\s+is\s+it)\b",
    re.IGNORECASE,
)

# Common time *request* phrases that can appear inside longer utterances.
# Intentionally does NOT match conceptual questions like "what is time".
_TIME_REQUEST_ANYWHERE_RE = re.compile(
    r"\b(?:"
    r"(?:can\s+you\s+)?tell\s+me\s+(?:the\s+)?time|"
    r"check\s+(?:the\s+)?time|"
    r"get\s+(?:the\s+)?time|"
    r"give\s+me\s+(?:the\s+)?time|"
    r"say\s+(?:the\s+)?time|"
    r"what\s+time(?:\s+is\s+it)?|"
    r"(?:what\s+s|whats|what'?s)\s+the\s+time|"
    r"current\s+time|"
    r"time\s+is\s+it"
    r")\b",
    re.IGNORECASE,
)


# Anchored date/day patterns: only match whole-utterance variants.
_DATE_RE = re.compile(
    r"^(?:"
    r"what\s+is\s+today|"
    r"(?:what\s+s|whats|what'?s)\s+today|"
    r"what\s+day(?:\s+of\s+the\s+week)?\s+is\s+it|"
    r"what\s+date\s+is\s+it|"
    r"what\s+is\s+the\s+date|"
    r"what\s+is\s+today\s+s\s+date|"
    r"what\s+is\s+todays\s+date|"
    r"(?:what\s+s|whats|what'?s)\s+the\s+date|"
    r"(?:what\s+s|whats|what'?s)\s+today\s+s\s+date|"
    r"(?:what\s+s|whats|what'?s)\s+todays\s+date|"
    r"today\s+s\s+date|"
    r"todays\s+date|"
    r"current\s+date|"
    r"the\s+date|"
    r"day\s+of\s+the\s+week"
    r")\??$",
    re.IGNORECASE,
)

# Date/day keywords inside longer utterances (e.g., "hey wyzer, what is today").
_DATE_KEYWORDS_ANYWHERE_RE = re.compile(
    r"\b(?:"
    r"what\s+is\s+today|"
    r"(?:what\s+s|whats|what'?s)\s+today|"
    r"what\s+day(?:\s+of\s+the\s+week)?\s+is\s+it|"
    r"what\s+date\s+is\s+it|"
    r"what\s+is\s+the\s+date|"
    r"(?:what\s+s|whats|what'?s)\s+the\s+date|"
    r"today\s+s\s+date|"
    r"todays\s+date|"
    r"current\s+date|"
    r"day\s+of\s+the\s+week|"
    r"time\s+and\s+date|"
    r"date\s+and\s+time"
    r")\b",
    re.IGNORECASE,
)

# Common date/day *request* phrases that can appear inside longer utterances.
_DATE_REQUEST_ANYWHERE_RE = re.compile(
    r"\b(?:"
    r"(?:can\s+you\s+)?tell\s+me\s+(?:the\s+)?(?:current\s+)?date|"
    r"(?:can\s+you\s+)?tell\s+me\s+what\s+day\s+it\s+is|"
    r"tell\s+me\s+today\s+s\s+date|"
    r"check\s+(?:the\s+)?date|"
    r"get\s+(?:the\s+)?date|"
    r"give\s+me\s+(?:the\s+)?date"
    r")\b",
    re.IGNORECASE,
)


def _extract_leftover_around_date(raw_text: str, normalized: str) -> str:
    """Extract non-date leftover from an utterance that also asks today's date/day.

    Used to support mixed queries like "What's the date and tell me a story?"
    by returning get_time as a tool_plan plus an LLM leftover.
    """
    raw = (raw_text or "").strip()
    if not raw:
        return ""

    n = (normalized or "").strip().lower()
    if not n or not (_DATE_KEYWORDS_ANYWHERE_RE.search(n) or _DATE_REQUEST_ANYWHERE_RE.search(n)):
        return ""

    splitter = re.compile(r"\s+(?:and|then|also|plus)\s+", re.IGNORECASE)
    parts = splitter.split(raw, maxsplit=1)
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        a_n = _normalize_text_for_routing(a)
        b_n = _normalize_text_for_routing(b)
        a_has_date = bool(_DATE_KEYWORDS_ANYWHERE_RE.search(a_n) or _DATE_REQUEST_ANYWHERE_RE.search(a_n))
        b_has_date = bool(_DATE_KEYWORDS_ANYWHERE_RE.search(b_n) or _DATE_REQUEST_ANYWHERE_RE.search(b_n))
        if a_has_date and b and not b_has_date:
            return b
        if b_has_date and a and not a_has_date:
            return a

    cleaned = raw
    cleaned = re.sub(_DATE_REQUEST_ANYWHERE_RE, " ", cleaned)
    cleaned = re.sub(_DATE_KEYWORDS_ANYWHERE_RE, " ", cleaned)
    cleaned = re.sub(r"\b(?:and|then|also|plus)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n,;:.?!")
    return cleaned.strip()


def _looks_like_date_fragment(normalized: str) -> bool:
    """Return True for short/fragment utterances that should map to get_time.

    Examples:
    - "today"
    - "date"
    - "today's date" (normalized: "today s date")
    - "day of the week"
    """
    n = (normalized or "").strip().lower()
    if not n:
        return False

    if n in {
        "today",
        "date",
        "the date",
        "current date",
        "todays date",
        "today s date",
        "day of week",
        "day of the week",
        "what is today",
        "what day is it",
        "what date is it",
        "what s the date",
        "whats the date",
        "what's the date",
        "time and date",
        "date and time",
    }:
        return True

    toks = n.split()
    if toks and all(t == "today" for t in toks):
        return True
    if toks and all(t == "date" for t in toks):
        return True

    # Allow small filler around "today"/"date": "uh today please", "hey date".
    if ("today" in toks or "date" in toks) and len(toks) <= 4:
        filler = {"uh", "um", "please", "pls", "hey", "yo", "ok", "okay"}
        non_key = [t for t in toks if t not in {"today", "date"}]
        if non_key and all(t in filler for t in non_key):
            return True

    return False


def _extract_leftover_around_time(raw_text: str, normalized: str) -> str:
    """Extract non-time leftover from an utterance that also asks the current time.

    Used to support mixed queries like "What's the time and give me a short story?"
    by returning get_time as a tool_plan plus an LLM leftover.
    """
    raw = (raw_text or "").strip()
    if not raw:
        return ""

    n = (normalized or "").strip().lower()
    if not n or not (_TIME_KEYWORDS_ANYWHERE_RE.search(n) or _TIME_REQUEST_ANYWHERE_RE.search(n)):
        return ""

    # Prefer splitting on common conjunctions to preserve user phrasing.
    splitter = re.compile(r"\s+(?:and|then|also|plus)\s+", re.IGNORECASE)
    parts = splitter.split(raw, maxsplit=1)
    if len(parts) == 2:
        a, b = parts[0].strip(), parts[1].strip()
        a_n = _normalize_text_for_routing(a)
        b_n = _normalize_text_for_routing(b)
        a_has_time = bool(_TIME_KEYWORDS_ANYWHERE_RE.search(a_n) or _TIME_REQUEST_ANYWHERE_RE.search(a_n))
        b_has_time = bool(_TIME_KEYWORDS_ANYWHERE_RE.search(b_n) or _TIME_REQUEST_ANYWHERE_RE.search(b_n))
        if a_has_time and b and not b_has_time:
            return b
        if b_has_time and a and not a_has_time:
            return a

    # Fallback: remove time-keyword phrases and cleanup separators.
    cleaned = raw
    cleaned = re.sub(_TIME_REQUEST_ANYWHERE_RE, " ", cleaned)
    cleaned = re.sub(_TIME_KEYWORDS_ANYWHERE_RE, " ", cleaned)
    cleaned = re.sub(r"\b(?:and|then|also|plus)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n,;:.?!")
    return cleaned.strip()


def _looks_like_time_fragment(normalized: str) -> bool:
    """Return True for short/fragment utterances that should map to get_time.

    Examples:
    - "time"
    - "time what is it"
    - repeated: "time time time"
    """
    n = (normalized or "").strip().lower()
    if not n:
        return False

    if n in {"time", "the time", "current time", "what time", "whats the time", "what's the time", "time is it"}:
        return True

    toks = n.split()
    if toks and all(t == "time" for t in toks):
        return True

    # "time what is it" (common STT fragment)
    if toks and toks[0] == "time" and set(toks[1:]) <= {"what", "is", "it"}:
        return True

    # Allow small filler around "time": "uh time please", "hey time".
    # Keep conservative by restricting to short utterances and a small filler set.
    if "time" in toks and len(toks) <= 4:
        filler = {"uh", "um", "please", "pls", "hey", "yo", "ok", "okay"}
        non_time = [t for t in toks if t != "time"]
        if non_time and all(t in filler for t in non_time):
            return True

    return False

# Weather patterns: match queries about weather, temperature, forecast
# Also includes implicit weather queries about clothing/outdoor activities
_WEATHER_RE = re.compile(
    r"(?:"
    # Explicit weather keywords
    r"\b(?:weather|temperature|temp|forecast)\b|"
    r"\bhow\s+(?:cold|hot|warm)\b|"
    r"\bwhat.{0,10}(?:weather|temperature|temp|forecast)\b|"
    r"\b(?:weather|temperature|forecast)\s+(?:in|for|at)\b|"
    r"\bis\s+it\s+(?:cold|hot|warm|raining|snowing)\b|"
    r"\bwill\s+it\s+(?:rain|snow)\b|"
    r"\bwhat.{0,10}like\s+outside\b|"
    # Implicit weather - clothing/outdoor prep questions
    r"\b(?:need|bring|wear|grab|take)\s+(?:a\s+|an\s+|my\s+)?(?:jacket|coat|umbrella|sweater|hoodie|raincoat|sunglasses|sunscreen|hat|scarf|gloves|boots)\b|"
    r"\bshould\s+i\s+(?:bring|wear|take|grab)\s+(?:a\s+|an\s+|my\s+)?(?:jacket|coat|umbrella|sweater|hoodie|raincoat|sunglasses|sunscreen|hat|scarf|gloves|boots)\b|"
    r"\b(?:do\s+i\s+need|will\s+i\s+need)\s+(?:a\s+|an\s+|my\s+)?(?:jacket|coat|umbrella|sweater|hoodie|raincoat|sunglasses|sunscreen|hat|scarf|gloves)\b|"
    r"\bdress\s+(?:warm|warmly|cool|light|lightly)\b|"
    r"\b(?:you\s+think|think)\s+i\s+(?:need|should)\b.{0,30}\b(?:jacket|coat|umbrella|sweater|warm|cold)\b"
    r")",
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

# ═══════════════════════════════════════════════════════════════════════════
# Phase 9: Window context patterns - "what am I looking at", "what's the active window"
# Screen awareness (READ-ONLY) - NO OCR, NO screenshots, NO automation
# ═══════════════════════════════════════════════════════════════════════════
_WINDOW_CONTEXT_RE = re.compile(
    r"^(?:"
    # NOTE: "what am I looking at" is intentionally routed to describe_screen
    # via _WHATS_ON_SCREEN_DEEP_RE / SCREEN_STATE_PHRASES instead.
    r"what(?:'?s|\s+is)\s+(?:the\s+)?(?:currently\s+)?(?:active|current|foreground|focused)\s+(?:windows?|app|application|program)|"  # "what's the active window", "what is the currently focused window"
    r"what\s+(?:windows?|app|application|program)\s+(?:is\s+)?(?:currently\s+)?(?:this|active|open|focus(?:ed)?)|"  # "what window is this", "what window is currently focused", "what windows currently focus"
    r"which\s+(?:windows?|app|application|program)\s+(?:is\s+)?(?:currently\s+)?(?:active|focus(?:ed)?|open)|"  # "which app is active", "which window is currently focused"
    r"what\s+(?:app|application|program)\s+(?:am\s+i\s+(?:currently\s+)?(?:in|using|on)|is\s+(?:this|active))|"  # "what app am I in" / "what app am I currently using"
    r"what(?:'?s|\s+is)\s+(?:this\s+)?(?:windows?|app|application)|"  # "what's this window"
    r"tell\s+me\s+(?:about\s+)?(?:the\s+)?(?:active|current|foreground)\s+(?:windows?|app)|"  # "tell me the active window"
    r"(?:current|active|focused)\s+(?:windows?|app|application)(?:\s+info)?|"  # "active window", "current app"
    # Broader natural variants: "what am I in", "where am I"
    r"what\s+am\s+i\s+(?:currently\s+)?(?:in|on|using)|"  # "what am I in", "what am I currently using"
    r"where\s+am\s+i(?:\s+right\s+now)?"  # "where am I", "where am I right now"
    r")\??\.*$",
    re.IGNORECASE,
)

# "what's on my screen/monitor" -> get_window_context (foreground)
_WHATS_ON_MY_SCREEN_RE = re.compile(
    r"^(?:what(?:'?s|\s+is)\s+on\s+(?:my\s+)?(?:screen|monitor)(?:\s+right\s+now)?|"
    r"what\s+am\s+i\s+seeing|"
    r"what\s+is\s+on\s+(?:my\s+)?(?:screen|monitor))\??$",
    re.IGNORECASE,
)

# "what windows are open" / "what's open" -> list_open_windows
_LIST_OPEN_WINDOWS_RE = re.compile(
    r"^(?:what\s+windows\s+are\s+open|"
    r"what(?:'?s|\s+is)\s+open|"
    r"what\s+do\s+i\s+have\s+open|"
    r"what\s+windows\s+do\s+i\s+have\s+open)\??$",
    re.IGNORECASE,
)

# "is notepad open" / "is notepad still open" / "there's notepad open" -> list_open_windows (filtered)
_IS_APP_OPEN_RE = re.compile(
    r"^(?:"
    # "there's notepad open" / "is there a notepad open" / "there is notepad open"
    r"(?:is\s+there|there(?:'?s|\s+is))\s+(?:a\s+|an\s+)?(?P<app2>.+?)\s+(?:(?:still\s+)?(?:open|running|active|up)|window)|"
    # "is notepad open" / "is notepad still open" / "is notepad running"
    r"is\s+(?P<app1>.+?)\s+(?:still\s+)?(?:open|running|active|up)"
    r")\s*[?.!]*$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Verification / status queries about the LAST action.
# "did the last thing I opened open successfully", "did that work",
# "did it open", "did it succeed", "what just happened"
# These MUST route to get_recent_events, NOT be misinterpreted as open_target.
# ═══════════════════════════════════════════════════════════════════════════
_VERIFICATION_QUERY_RE = re.compile(
    r"^(?:"
    # "did the last thing ... open/work/succeed"
    r"did\s+(?:the\s+last\s+thing\s+(?:you|i|we)\s+)?(?:open(?:ed)?|launch(?:ed)?|start(?:ed)?|close(?:d)?|do)\s+"
    r"(?:actually\s+)?(?:open|work|succeed|close|launch|start|run|happen|finish)(?:\s+successfully)?|"
    # "did that/it work" / "did that/it open" / "did that/it succeed" / "did that/it fail"
    r"did\s+(?:that|it|this)\s+(?:actually\s+)?(?:work|open|succeed|fail|close|launch|start|run|happen|finish|go\s+through)(?:\s+successfully)?|"
    # "was that successful" / "was it successful"
    r"was\s+(?:that|it|this)\s+(?:successful|a\s+success)|"
    # "what just happened" / "what happened"
    r"what\s+(?:just\s+)?happened|"
    # "did the last thing you open actually open successfully" (full phrase)
    r"did\s+the\s+last\s+thing\s+.{0,40}(?:open|work|succeed|fail|close|launch|start)(?:\s+successfully)?"
    r")\??$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# QUESTION GUARD: Prevent interrogative sentences from matching open_target.
# If the utterance begins with a question word (did/was/is/has/etc.),
# it is a question, NOT an imperative action command.
# ═══════════════════════════════════════════════════════════════════════════
_QUESTION_PREFIX_RE = re.compile(
    r"^(?:did|was|is|has|have|does|do|were|are|could|would|should|what|how|why|when|where|which)\s",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 14: Desktop Ground Truth patterns
# ═══════════════════════════════════════════════════════════════════════════

# "what's on screen right now" / "tell me what you see" -> describe_screen
_WHATS_ON_SCREEN_DEEP_RE = re.compile(
    r"^(?:what(?:'?s|\s+is)\s+on\s+(?:the\s+)?screen\s+right\s+now|"
    r"describe\s+(?:the\s+|my\s+)?screen|"
    r"describe\s+what(?:'?s|\s+is)\s+(?:on\s+(?:my\s+)?screen|in\s+front\s+of\s+me)|"
    r"read\s+(?:the\s+)?screen|"
    r"what\s+(?:controls?|buttons?|elements?)\s+(?:are|do\s+i\s+see)\s+on\s+(?:the\s+)?screen|"
    r"what\s+(?:do\s+i|can\s+i)\s+see\s+on\s+(?:the\s+)?screen|"
    r"(?:tell\s+me\s+)?what\s+(?:do\s+)?you\s+see|"
    r"what\s+am\s+i\s+looking\s+at)\??$",
    re.IGNORECASE,
)

# "is there a button that says X" / "do you see a button called X"
_BUTTON_CHECK_RE = re.compile(
    r"^(?:is\s+there\s+a\s+button\s+(?:that\s+says?|called|named|labelled?|saying)\s+(.+)|"
    r"do\s+you\s+see\s+(?:a\s+)?(?:button\s+)?(?:that\s+says?|called|named)\s+(.+)|"
    r"can\s+you\s+(?:see|find)\s+(?:a\s+)?(?:button\s+)?(?:that\s+says?|called|named)\s+(.+)|"
    r"(?:is|are)\s+there\s+(?:an?\s+)?(.+?)\s+button)\??$",
    re.IGNORECASE,
)

# "did install succeed" / "did it install" / "is install done"
_INSTALL_CHECK_RE = re.compile(
    r"^(?:"
    # "did the install/download/update succeed/finish/complete/work/fail"
    r"did\s+(?:the\s+)?(?:install(?:ation)?|download|setup|update)\s+(?:succeed|work|finish|complete|fail)|"
    r"did\s+it\s+(?:install|download|update)(?:\s+successfully)?|"
    # "is the install/download/update done/finished/complete/ready"
    r"is\s+(?:the\s+)?(?:install(?:ation)?|download|setup|update)\s+(?:done|complete|finished|ready)|"
    # "is it installed/downloaded/done/finished"
    r"is\s+it\s+(?:installed|downloaded|done|finished|complete)|"
    # "has the download/install completed/finished"
    r"has\s+(?:the\s+)?(?:install(?:ation)?|download|setup|update)\s+(?:completed|finished|succeeded|failed)|"
    r"was\s+(?:the\s+)?(?:install(?:ation)?|download|setup|update)\s+successful"
    r")\??$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Phase 15: Broad UI-state queries that MUST route through perception
# These catch questions the LLM would otherwise hallucinate about.
# ═══════════════════════════════════════════════════════════════════════════
_UI_STATE_QUERY_RE = re.compile(
    r"^(?:"
    # Dialog / popup content
    r"what\s+does\s+(?:the\s+|this\s+|that\s+)?(?:dialog|popup|prompt|notification|alert|window|box|message)\s+say|"
    r"what\s+(?:is|does)\s+(?:the\s+|this\s+|that\s+)?(?:error|warning|message|notification|alert)\s+(?:say|mean|show)|"
    r"read\s+(?:the\s+|this\s+|that\s+)?(?:dialog|popup|prompt|error|warning|message|notification|alert)(?:\s+message)?|"
    # Progress / status  (completion queries handled by _INSTALL_CHECK_RE)
    r"(?:is\s+it|are\s+we)\s+(?:still\s+)?(?:downloading|installing|updating|loading|processing|uploading|extracting|copying)|"
    r"how\s+far\s+(?:along\s+)?(?:is\s+(?:the\s+|it\s+)?)?(?:download|install|update|loading|progress)|"
    r"(?:what(?:'?s|\s+is)\s+the\s+)?(?:download|install|update|loading)\s+(?:progress|status|percentage)|"    
    # What's happening / showing
    r"what(?:'?s|\s+is)\s+(?:happening|showing|displayed?)\s+(?:on\s+(?:the\s+)?screen|now|right\s+now|here)|"
    r"what\s+(?:is\s+)?(?:this|that)\s+(?:on\s+(?:the\s+)?screen|saying|showing)|"
    r"can\s+you\s+(?:read|check|verify|confirm)\s+(?:the\s+|what(?:'?s|\s+is)\s+on\s+(?:the\s+)?)?screen"
    r")\??$",
    re.IGNORECASE,
)

def _is_ui_state_query(text: str) -> bool:
    """Return True if the text is asking about on-screen UI state.
    
    These queries MUST be routed through a perception tool — the LLM
    must never answer them from imagination.
    """
    return bool(_UI_STATE_QUERY_RE.match((text or "").strip()))

# ═══════════════════════════════════════════════════════════════════════════
# Phase 14b: Broad screen-description & element-verify via normalized text
# These use phrase-containment on normalized text (no anchoring) so they
# catch natural speech like "Oh, what's on my screen? Can you describe it?"
# ═══════════════════════════════════════════════════════════════════════════

# Canonical set of screen-state phrases (normalized form, no apostrophes).
# Shared with orchestrator.should_use_streaming_tts() via import.
SCREEN_STATE_PHRASES: tuple[str, ...] = (
    "tell me what you see",
    "what do you see",
    "whats on screen",
    "whats on my screen",
    "what is on my screen",
    "what is on screen",
    "what is on the screen",
    "whats on the screen",
    "describe the screen",
    "describe my screen",
    "describe whats in front of me",
    "what am i looking at",
    "can you describe it",
    "screen right now",
    "describe whats on",
    "read the screen",
    "read my screen",
    "what can you see",
    "tell me whats on my screen",
    "tell me whats on the screen",
    "whats showing on my screen",
    "whats currently on my screen",
    "whats currently on screen",
    # Catch queries about controls/buttons/elements on screen (e.g.
    # "how many interactive controls on screen", "give me the top 10
    # interactive controls on screen", "list the buttons on my screen")
    "controls on screen",
    "controls on my screen",
    "controls on the screen",
    "elements on screen",
    "elements on my screen",
    "elements on the screen",
    "buttons on screen",
    "buttons on my screen",
    "buttons on the screen",
)

_SCREEN_DESCRIBE_PHRASES = list(SCREEN_STATE_PHRASES)

# Verb-phrases that indicate the user is asking about element existence.
_VERIFY_ELEMENT_TRIGGERS_RE = re.compile(
    r"(?:"
    r"\bis\s+there\b|"
    r"\bdo\s+you\s+see\b|"
    r"\bcan\s+you\s+see\b|"
    r"^you\s+see\b|"
    r"\bi\s+see\b"
    r")",
    re.IGNORECASE,
)

# Labels that strongly suggest UIA element queries.
_ELEMENT_LABEL_WORDS = {
    "button", "install", "play", "next", "close", "ok", "okay", "cancel",
    "yes", "no", "submit", "save", "apply", "open", "start", "stop",
    "download", "update", "continue", "accept", "decline", "retry",
    "back", "forward", "settings", "menu", "checkbox", "link",
}


def _match_screen_describe_intent(normalized: str) -> Optional[HybridDecision]:
    """Check if normalized text contains a screen-description phrase.

    Returns a HybridDecision routing to describe_screen, or None.
    """
    if not normalized:
        return None
    for phrase in _SCREEN_DESCRIBE_PHRASES:
        if phrase in normalized:
            logger.info(
                "[ROUTER] matched=screen_describe tool=describe_screen "
                f"reason=phrase '{phrase}' found in normalized text"
            )
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "describe_screen", "args": {}, "continue_on_error": False}],
                reply="",
                confidence=0.94,
            )
    return None


def _extract_element_label(normalized: str) -> Optional[str]:
    """Extract the target label from a verify-element query.

    Rules (applied in order):
    1. Quoted string -> use it.
    2. "button that says X" -> X
    3. "an X button" / "a X button" / "X button" -> X
    4. Last 1-4 tokens after see/says as fallback.
    """
    # 1. Quoted text
    m = re.search(r'["\'](.+?)["\']', normalized)
    if m:
        return m.group(1).strip()

    # 2. "button that says X" / "button called X" / "button named X"
    m = re.search(r"button\s+(?:that\s+says?|called|named|labelled?)\s+(.+)", normalized)
    if m:
        return m.group(1).strip()

    # 3. "an X button" / "a X button" / "the X button"
    m = re.search(r"(?:an?|the)\s+(.+?)\s+button", normalized)
    if m:
        return m.group(1).strip()

    # 3b. "X button" (no article)
    m = re.search(r"(\w+)\s+button", normalized)
    if m:
        label = m.group(1).strip()
        # Avoid capturing trigger words as the label
        if label not in {"a", "an", "the", "any", "some", "this", "that"}:
            return label

    # 4. Fallback: last 1-4 tokens after "see" or "says"
    m = re.search(r"(?:see|says?)\s+(.+)", normalized)
    if m:
        tokens = m.group(1).strip().split()
        # Strip leading articles/filler
        while tokens and tokens[0] in {"a", "an", "the", "that", "some"}:
            tokens = tokens[1:]
        if tokens:
            return " ".join(tokens[:4])

    return None


def _match_verify_element_intent(normalized: str) -> Optional[HybridDecision]:
    """Check if normalized text is asking whether a UI element exists.

    Triggers on phrases like:
      - "is there a button that says Install"
      - "do you see an install button"
      - "you see a play button"
      - "can you see the close button"

    Returns a HybridDecision routing to ui_find_text, or None.
    """
    if not normalized:
        return None
    if not _VERIFY_ELEMENT_TRIGGERS_RE.search(normalized):
        return None

    # Must mention "button" OR contain a well-known element label OR have a
    # quoted target — otherwise it might be a generic "is there/do you see" query.
    has_button_word = "button" in normalized
    tokens_set = set(normalized.split())
    has_label_word = bool(tokens_set & _ELEMENT_LABEL_WORDS)
    has_quoted = bool(re.search(r'["\']', normalized))

    if not (has_button_word or has_label_word or has_quoted):
        return None

    label = _extract_element_label(normalized)
    if not label:
        # If we matched the trigger + button/label but couldn't extract,
        # still route to tool with a broad search.
        label = ""
        # Try one more time: take first label-word found
        for tok in normalized.split():
            if tok in _ELEMENT_LABEL_WORDS and tok != "button":
                label = tok
                break
        if not label:
            return None

    logger.info(
        "[ROUTER] matched=verify_element tool=ui_find_text "
        f"reason=trigger + label '{label}' found in normalized text"
    )
    # When the user mentions "button", narrow to Button type + exact/word mode
    args: Dict[str, Any] = {"text": label, "method": "uia"}
    if has_button_word:
        args["control_type"] = "Button"
        args["match_mode"] = "exact"
    else:
        args["match_mode"] = "word"
    return HybridDecision(
        mode="tool_plan",
        intents=[{"tool": "ui_find_text", "args": args, "continue_on_error": False}],
        reply="",
        confidence=0.93,
    )


# Anchored open/launch/start.
_OPEN_RE = re.compile(r"^(open|launch|start)\s+(.+)$", re.IGNORECASE)

# Anchored close/quit/exit.
_CLOSE_RE = re.compile(r"^(close|quit|exit)\s+(.+)$", re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════
# CAPABILITIES / HELP PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
_CAPABILITIES_RE = re.compile(
    r"(?:"
    r"what\s+can\s+you\s+do|"
    r"what\s+are\s+(?:your|you)\s+(?:capabilities|abilities|features|skills|commands)|"
    r"list\s+(?:your\s+)?(?:tools|commands|capabilities|abilities|features)|"
    r"what\s+(?:tools|commands)\s+(?:do\s+you\s+have|are\s+(?:there|available))|"
    r"help\s*$|"
    r"what\s+do\s+you\s+do|"
    r"show\s+(?:me\s+)?(?:your\s+)?(?:commands|tools|capabilities|abilities)|"
    r"what\s+(?:are\s+you|you)\s+capable\s+of"
    r")",
    re.IGNORECASE,
)


# ── Extract numeric limit from event-log queries ─────────────────────────
_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "fifteen": 15, "twenty": 20,
}
_EVENT_LIMIT_RE = re.compile(
    r"\b(?:last|recent|past)\s+(\d+|"
    + "|".join(_WORD_TO_NUM) +
    r")\s+(?:\w+\s+)?(?:action|event|command|thing|step|item|tool)s?\b",
    re.IGNORECASE,
)

def _extract_event_limit(text: str) -> Optional[int]:
    """Extract a numeric limit from queries like 'last five actions'."""
    m = _EVENT_LIMIT_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1).lower()
    if raw.isdigit():
        return min(max(int(raw), 1), 50)
    return _WORD_TO_NUM.get(raw)


def _is_capabilities_query(text: str) -> bool:
    """Check if text is asking about Wyzer's capabilities/features."""
    tl = (text or "").strip()
    if not tl:
        return False
    n = _normalize_text_for_routing(tl)
    return bool(_CAPABILITIES_RE.search(n))


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-INTENT EXTRACTION (deterministic segment splitting)
# ═══════════════════════════════════════════════════════════════════════════
# Connectors that separate segments inside a compound utterance.
_MI_CONNECTORS_RE = re.compile(
    r"\s*(?:"
    r",?\s*(?:and\s+then|then|and|also|plus|but\s+then|but)\s+"
    r"|\.\s+"
    r"|;\s*"
    r"|,\s*"
    r"|[?!]\s+"       # sentence boundary: "What can you do? Tell me the time"
    r")",
    re.IGNORECASE,
)

# Broader time-request pattern used inside _segment_matches_tool to catch
# phrasings like "what is the time" that the anchored regexes miss.
_SEGMENT_TIME_RE = re.compile(
    r"(?:"
    r"what\s+is\s+the\s+time|"
    r"what\s+time(?:\s+is\s+it)?|"
    r"(?:what\s*s|whats|what'?s)\s+the\s+time|"
    r"(?:can\s+you\s+)?tell\s+me\s+(?:the\s+)?time|"
    r"(?:get|check|give\s+me)\s+(?:the\s+)?time|"
    r"current\s+time|"
    r"time\s+is\s+it"
    r")",
    re.IGNORECASE,
)


# Leading filler words stripped from segments before anchored regex matching.
# Handles Whisper noise ("You open notepad", "Can you open notepad", etc.)
_SEGMENT_FILLER_RE = re.compile(
    r"^(?:"
    r"(?:you\s+)"
    r"|(?:can|could|would)\s+you\s+(?:please\s+)?"
    r"|please\s+"
    r"|hey\s+wyzer\s+"
    r"|wyzer\s+"
    r"|okay\s+"
    r"|ok\s+"
    r"|go\s+ahead\s+and\s+"
    r"|go\s+"
    r")+",
    re.IGNORECASE,
)


def _strip_segment_fillers(text: str) -> str:
    """Strip common leading filler words from a segment."""
    return _SEGMENT_FILLER_RE.sub("", text).strip()


def _segment_matches_tool(segment: str) -> Optional[Dict[str, Any]]:
    """Try to match a single segment to a deterministic tool intent.

    Returns an intent dict ``{"tool": ..., "args": ...}`` or None.
    """
    seg = (segment or "").strip()
    if not seg:
        return None

    seg_norm = _normalize_text_for_routing(seg)

    # Capabilities / help
    if _is_capabilities_query(seg):
        return {"tool": "get_capabilities", "args": {}, "continue_on_error": False}

    # Time
    if (
        _TIME_RE.match(seg_norm) or _looks_like_time_fragment(seg_norm)
        or _TIME_KEYWORDS_ANYWHERE_RE.search(seg_norm)
        or _TIME_REQUEST_ANYWHERE_RE.search(seg_norm)
        or _SEGMENT_TIME_RE.search(seg_norm)
    ):
        return {"tool": "get_time", "args": {}, "continue_on_error": False}

    # Date
    if _DATE_RE.match(seg_norm) or _looks_like_date_fragment(seg_norm):
        return {"tool": "get_time", "args": {}, "continue_on_error": False}

    # Verification / status queries about last action (must be before open/close)
    if _VERIFICATION_QUERY_RE.match(seg.strip()):
        return {"tool": "get_recent_events", "args": {"limit": 5}, "continue_on_error": False}

    # Foreground / window context queries
    if _WINDOW_CONTEXT_RE.match(seg.strip()):
        return {"tool": "get_window_context", "args": {}, "continue_on_error": False}

    # Open / launch — try raw first, then filler-stripped
    # GUARD: skip if the segment is a question (starts with did/was/is/has/etc.)
    seg_clean = _strip_segment_fillers(seg)
    for candidate in (seg.strip(), seg_clean):
        if _QUESTION_PREFIX_RE.match(candidate):
            break  # question, not an imperative command
        m = _OPEN_RE.match(candidate)
        if m:
            target = (m.group(2) or "").strip().strip("\"'")
            if target and target.lower() not in {"it", "this", "that", "something", "anything"}:
                return {"tool": "open_target", "args": {"query": target}, "continue_on_error": False}

    # Close — try raw first, then filler-stripped
    for candidate in (seg.strip(), seg_clean):
        m = _CLOSE_RE.match(candidate)
        if m:
            target = (m.group(2) or "").strip().strip("\"'")
            if target:
                return {"tool": "close_window", "args": {"title": target}, "continue_on_error": False}

    # Single-clause fallback (media, volume, etc.)
    decision = _decide_single_clause(seg)
    if decision.mode == "tool_plan" and decision.intents and decision.confidence >= 0.8:
        return decision.intents[0]

    return None


# ── Tool-trigger patterns for boundary scanning ──────────────────────────
# Used by the fallback path inside extract_multi_intents when connector
# splitting fails (e.g. Whisper mistranscribes "and" as "in").  Each entry
# is (compiled_regex, tool_name, args_factory).
_TOOL_TRIGGER_PATTERNS: List[Tuple[re.Pattern, str, Any]] = [
    (_CAPABILITIES_RE, "get_capabilities", lambda _m: {}),
    (_SEGMENT_TIME_RE, "get_time", lambda _m: {}),
]


def _scan_tool_spans(text: str) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Find all non-overlapping tool-trigger spans in *text*.

    Returns a list of ``(start, end, intent_dict)`` sorted by start.
    """
    norm = _normalize_text_for_routing(text)
    hits: List[Tuple[int, int, Dict[str, Any]]] = []
    for pat, tool, args_fn in _TOOL_TRIGGER_PATTERNS:
        for m in pat.finditer(norm):
            hits.append((m.start(), m.end(), {
                "tool": tool, "args": args_fn(m), "continue_on_error": False,
            }))
    # Sort by position and remove overlapping hits (keep longest).
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    filtered: List[Tuple[int, int, Dict[str, Any]]] = []
    last_end = -1
    for start, end, intent in hits:
        if start >= last_end:
            filtered.append((start, end, intent))
            last_end = end
    return filtered


def extract_multi_intents(text: str) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """Split *text* into deterministic tool intents + freeform leftover.

    Returns ``(tool_intents, leftover_text)`` when at least one tool intent
    is found.  ``leftover_text`` is the remaining freeform text that must be
    handled by the LLM.  Returns ``None`` when no tool segments are detected.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    # Split on connectors, keeping track of positions so we can reconstruct
    # the original segments for leftover text.
    segments = _MI_CONNECTORS_RE.split(raw)
    segments = [s.strip() for s in segments if s and s.strip()]

    if len(segments) >= 2:
        tool_intents: List[Dict[str, Any]] = []
        leftover_parts: List[str] = []

        for seg in segments:
            intent = _segment_matches_tool(seg)
            if intent is not None:
                tool_intents.append(intent)
            else:
                leftover_parts.append(seg)

        if tool_intents:
            leftover = " ".join(leftover_parts).strip()
            return (tool_intents, leftover)

    # ── Fallback: boundary scanning ──────────────────────────────────────
    # When connector splitting fails (e.g. Whisper mistranscribes "and" as
    # "in"), scan for multiple known tool-trigger patterns in the raw text.
    # If 2+ non-overlapping tool triggers are found, extract them directly.
    spans = _scan_tool_spans(raw)
    if len(spans) >= 2:
        tool_intents_fb: List[Dict[str, Any]] = []
        seen_tools: set = set()
        for _start, _end, intent in spans:
            key = intent["tool"]
            if key not in seen_tools:
                tool_intents_fb.append(intent)
                seen_tools.add(key)
        if len(tool_intents_fb) >= 2:
            # Collect text outside any tool span as leftover.
            norm = _normalize_text_for_routing(raw)
            leftover_chars = list(norm)
            for start, end, _ in spans:
                for i in range(start, min(end, len(leftover_chars))):
                    leftover_chars[i] = " "
            leftover_fb = re.sub(r"\s+", " ", "".join(leftover_chars)).strip()
            # Remove stray small connector words left behind.
            leftover_fb = re.sub(
                r"^\s*(?:in|and|but|then|also|plus)\s*$", "", leftover_fb, flags=re.IGNORECASE
            ).strip()
            return (tool_intents_fb, leftover_fb)

    return None

# Anchored minimize/shrink.
_MINIMIZE_RE = re.compile(r"^(minimize|shrink)\s+(.+)$", re.IGNORECASE)

# Anchored maximize/fullscreen/expand.
_MAXIMIZE_RE = re.compile(r"^(maximize|fullscreen|expand|full\s+screen)\s+(.+)$", re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════
# Click / press / UI-action patterns
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# Phase 16: "click <target> and type <text>" — deterministic fast-path
# ═══════════════════════════════════════════════════════════════════════════
# Matches:
#   "click on ask anything and type hello"
#   "click ask anything and type hello world"
#   "press search box and type test query"
_CLICK_AND_TYPE_RE = re.compile(
    r"^(?:click|press|hit|tap)\s+(?:on\s+)?(?:the\s+)?"
    r"(?P<target>.+?)\s+and\s+type[,;:\s]\s*(?P<text>.+)$",
    re.IGNORECASE,
)


def _match_click_and_type(text: str) -> Optional[HybridDecision]:
    """Route 'click <target> and type <text>' to the deterministic
    click-and-type orchestrator.  NO LLM involved.

    Returns a HybridDecision with mode="tool_plan" using the special
    __CLICK_AND_TYPE__ pseudo-tool, or None if the pattern doesn't match.
    """
    m = _CLICK_AND_TYPE_RE.match((text or "").strip())
    if not m:
        return None

    target = m.group("target").strip().rstrip("?.,;: ")
    type_text = m.group("text").strip().rstrip("?.,;: ")
    if not target or not type_text:
        return None

    logger.info(
        f"[ROUTER] matched=click_and_type target={target!r} text={type_text!r}"
    )
    return HybridDecision(
        mode="tool_plan",
        intents=[{
            "tool": "__CLICK_AND_TYPE__",
            "args": {"target": target, "text": type_text},
            "continue_on_error": False,
        }],
        reply="",
        confidence=0.97,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 16b: "type <text>" / "write <text>" — direct typing into focused field
# ═══════════════════════════════════════════════════════════════════════════
# Matches:
#   "type hello world"
#   "write this is a test"
#   "enter some text here"
#   "type out good morning"
#   "write out hello there"
_TYPE_DIRECT_RE = re.compile(
    r"^(?:type|write|enter)\s+(?:out\s+|in\s+)?"
    r"(?P<text>.+)$",
    re.IGNORECASE,
)

# Guard words: if the text part starts with these, it's NOT a typing request
# but a question/command (e.g. "write a poem", "type of file", "enter the app")
_TYPE_DIRECT_GUARD_RE = re.compile(
    r"^(?:a\s|an\s|me\s|the\s|my\s|of\s|code\b|poem\b|essay\b|story\b"
    r"|song\b|email\b|letter\b|script\b|program\b|function\b|paragraph\b"
    r"|article\b|report\b|some\s+code\b)",
    re.IGNORECASE,
)


def _match_type_direct(text: str) -> Optional[HybridDecision]:
    """Route bare 'type <text>' / 'write <text>' to the type_text tool.

    Types the text into whatever field/control is currently focused.
    Returns a HybridDecision or None.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    m = _TYPE_DIRECT_RE.match(raw)
    if not m:
        return None

    type_text = m.group("text").strip().rstrip("?.,;: ")
    if not type_text:
        return None

    # Guard: skip creative/generative requests like "write a poem"
    if _TYPE_DIRECT_GUARD_RE.match(type_text):
        return None

    logger.info(f"[ROUTER] matched=type_direct text={type_text!r}")
    return HybridDecision(
        mode="tool_plan",
        intents=[{
            "tool": "type_text",
            "args": {"text": type_text},
            "continue_on_error": False,
        }],
        reply="",
        confidence=0.95,
    )


# "click the Maximize button" / "click Maximize" / "press the Close button"
_CLICK_WINCTL_RE = re.compile(
    r"^(?:click|press|hit|tap)\s+(?:the\s+|on\s+(?:the\s+)?)?"  # verb + optional article
    r"(maximize|minimise|minimize|close|restore)"
    r"(?:\s+button)?\s*$",
    re.IGNORECASE,
)

# Generic "click X" / "click the X button" / "press the X button"
_CLICK_GENERIC_RE = re.compile(
    r"^(?:click|press|hit|tap)\s+(?:the\s+|on\s+(?:the\s+)?)?(.+)$",
    re.IGNORECASE,
)

# Map well-known window-control names to the native window-management tools
_WINCTL_TOOL_MAP = {
    "maximize": "maximize_window",
    "minimise": "minimize_window",
    "minimize": "minimize_window",
    "close": "close_window",
    "restore": "maximize_window",   # restore uses the same underlying API
}


def _match_click_intent(text: str) -> Optional[HybridDecision]:
    """Route 'click X' / 'press the X button' to deterministic tools.

    - Well-known window controls (Maximize, Minimize, Close) ->
      native window-management tools.
    - Generic click -> desktop_click_uia.

    Returns a HybridDecision or None.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    # 1. Well-known window controls first
    m = _CLICK_WINCTL_RE.match(raw)
    if m:
        ctl = m.group(1).strip().lower()
        tool = _WINCTL_TOOL_MAP.get(ctl)
        if tool:
            logger.info(
                f"[ROUTER] matched=click_winctl tool={tool} target={ctl}"
            )
            # For window controls we use the native tool with an empty title
            # (meaning "focused window").
            return HybridDecision(
                mode="tool_plan",
                intents=[{
                    "tool": tool,
                    "args": {"title": ""},
                    "continue_on_error": False,
                }],
                reply="",
                confidence=0.95,
            )

    # 2. Generic "click X" / "click the X button"
    m = _CLICK_GENERIC_RE.match(raw)
    if m:
        target_raw = m.group(1).strip().rstrip("?. ")
        if not target_raw:
            return None

        # Strip trailing " button" to get the control name
        target_name = re.sub(r"\s+button\s*$", "", target_raw, flags=re.IGNORECASE).strip()
        if not target_name:
            target_name = target_raw

        # Guard: don't swallow multi-action sentences
        target_l = target_name.lower()
        if any(v in target_l.split() for v in ["then", "and", "also", "plus"]):
            return None

        # Check if it's a window control name that slipped past the anchored regex
        ctl_lower = target_name.lower()
        if ctl_lower in _WINCTL_TOOL_MAP:
            tool = _WINCTL_TOOL_MAP[ctl_lower]
            logger.info(
                f"[ROUTER] matched=click_winctl tool={tool} target={ctl_lower}"
            )
            return HybridDecision(
                mode="tool_plan",
                intents=[{
                    "tool": tool,
                    "args": {"title": ""},
                    "continue_on_error": False,
                }],
                reply="",
                confidence=0.95,
            )

        # Route generic "click X" through the full click-and-type
        # orchestrator (with text="") so it benefits from perception,
        # ancestor promotion, OCR fallback, disambiguation overlay, and
        # retry chain.  Do NOT force control_type: "Button" — the target
        # could be an Edit, ListItem, Pane, Hyperlink, etc.
        logger.info(
            f"[ROUTER] matched=click tool=__CLICK_AND_TYPE__ target={target_name}"
        )
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "__CLICK_AND_TYPE__",
                "args": {"target": target_name, "text": ""},
                "continue_on_error": False,
            }],
            reply="",
            confidence=0.90,
        )

    return None

# ═══════════════════════════════════════════════════════════════════════════
# Switch app patterns: deterministic app switching using focus history
# ═══════════════════════════════════════════════════════════════════════════
# "switch to X" / "go to X" / "switch back to X"
_SWITCH_TO_APP_RE = re.compile(
    r"^(?:"
    r"(?:switch(?:\s+back)?|go)\s+to"          # "switch to X", "go to X"
    r"|focus(?:\s+on)?"                         # "focus X", "focus on X"
    r"|bring\s+up"                              # "bring up X"
    r"|pull\s+up"                               # "pull up X"
    r")\s+(.+)$",
    re.IGNORECASE,
)

# "go back" / "switch back" / "previous app" / "last app"
_SWITCH_PREVIOUS_RE = re.compile(
    r"^(?:"
    r"go\s+back|"                          # "go back"
    r"switch\s+back|"                      # "switch back"
    r"previous\s+(?:app|window|application)|"  # "previous app"
    r"last\s+(?:app|window|application)|"      # "last app"
    r"back\s+to\s+(?:the\s+)?(?:last|previous)\s+(?:app|window)"  # "back to the last app"
    r")$",
    re.IGNORECASE,
)

# "next app" / "cycle apps"
_SWITCH_NEXT_RE = re.compile(
    r"^(?:"
    r"next\s+(?:app|window|application)|"  # "next app"
    r"cycle\s+(?:apps?|windows?)|"         # "cycle apps"
    r"switch\s+(?:to\s+)?next(?:\s+(?:app|window))?"  # "switch next", "switch to next app"
    r")$",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Google search patterns: "google this cats", "google cats", "search google for cats"
# ═══════════════════════════════════════════════════════════════════════════
_GOOGLE_SEARCH_RE = re.compile(
    r"^google\s+(?:this:?\s*)?(?P<q>.+)$",
    re.IGNORECASE,
)
_SEARCH_GOOGLE_RE = re.compile(
    r"^search\s+google\s+for\s+(?P<q>.+)$",
    re.IGNORECASE,
)

# Anchored audio device switching: "switch/set/change/swap audio [out/output] [device] to <device>"
_AUDIO_DEVICE_SWITCH_RE = re.compile(
    r"^(?:(?:switch|set|change|swap)\s+(?:audio(?:\s+out(?:put)?)?|sound|out(?:put)?)(?:\s+device)?\s+to)\s+(.+)$",
    re.IGNORECASE,
)

# Anchored audio device listing: "list audio devices", "show audio devices", etc.
_AUDIO_DEVICE_LIST_RE = re.compile(
    r"^(?:list|show|display|what)(?:\s+(?:audio|sound))?(?:\s+(?:output\s+)?devices?|devices?|speakers?)?(?:\s+are\s+available)?\??$",
    re.IGNORECASE,
)

# Word-to-digit mapping for monitor numbers
_WORD_TO_DIGIT = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
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
# Common executable/file extensions that should NOT be treated as domains
_EXECUTABLE_EXTENSIONS = {".exe", ".msi", ".bat", ".cmd", ".ps1", ".sh", ".app", ".dmg", ".deb", ".rpm"}


def _looks_like_url_or_domain(text: str) -> bool:
    tl = (text or "").strip().lower()
    if not tl:
        return False

    if _URL_SCHEME_RE.search(tl) or _WWW_RE.search(tl):
        return True

    # Any obvious domain-like token (foo.com, foo.co.uk, etc.).
    # But exclude common executable extensions like .exe, .msi, etc.
    m = _DOMAIN_RE.search(tl)
    if m:
        matched = m.group(0)
        # Check if it ends with an executable extension
        for ext in _EXECUTABLE_EXTENSIONS:
            if matched.endswith(ext):
                return False
        return True

    return False

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


def _decide_single_clause(text: str) -> HybridDecision:
    clause = (text or "").strip()
    if not clause:
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.0)

    # Capabilities / help queries.
    if _is_capabilities_query(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_capabilities", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # Safety: URLs/domains go to LLM.
    if _looks_like_url_or_domain(clause):
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.8)

    # Time queries.
    clause_norm = _normalize_text_for_routing(_strip_trailing_punct(clause))
    if clause_norm and (_TIME_RE.match(clause_norm) or _looks_like_time_fragment(clause_norm)):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_time", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # Date/day-of-week queries.
    if clause_norm and (_DATE_RE.match(clause_norm) or _looks_like_date_fragment(clause_norm)):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_time", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Google search queries: "google this cats", "google cats", "search google for cats"
    # ═══════════════════════════════════════════════════════════════════════
    m = _GOOGLE_SEARCH_RE.match(clause)
    if m:
        query = (m.group("q") or "").strip()
        if query:
            return HybridDecision(
                mode="tool_plan",
                intents=[{
                    "tool": "google_search_open",
                    "args": {"query": query},
                    "continue_on_error": False
                }],
                reply=f"Opening Google for: {query}.",
                confidence=0.95,
            )
    
    m = _SEARCH_GOOGLE_RE.match(clause)
    if m:
        query = (m.group("q") or "").strip()
        if query:
            return HybridDecision(
                mode="tool_plan",
                intents=[{
                    "tool": "google_search_open",
                    "args": {"query": query},
                    "continue_on_error": False
                }],
                reply=f"Opening Google for: {query}.",
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
        m = re.search(r"\b(?:in|for|at|near|on)\s+(.+?)(?:\?|!|\.)?$", clause, re.IGNORECASE)
        if m:
            location = (m.group(1) or "").strip().rstrip("?!.").strip()
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
            today = _datetime_date.today()
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

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 9: Window context queries - "what am I looking at", "what app is active"
    # Screen awareness (READ-ONLY) - NO OCR, NO screenshots, NO automation
    # ═══════════════════════════════════════════════════════════════════════
    if _WHATS_ON_MY_SCREEN_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "describe_screen", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    if _LIST_OPEN_WINDOWS_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "list_open_windows", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # "is notepad open" / "is notepad still open" / "there's notepad open"
    _app_open_m = _IS_APP_OPEN_RE.match(clause)
    if _app_open_m:
        _app_name = (_app_open_m.group("app2") or _app_open_m.group("app1") or "").strip()
        if _app_name and _app_name.lower() not in {"it", "this", "that", "anything", "something"}:
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "list_open_windows", "args": {"_app_filter": _app_name}, "continue_on_error": False}],
                reply="",
                confidence=0.93,
            )

    if _WINDOW_CONTEXT_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_window_context", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 14: Desktop Ground Truth deterministic routes
    # ═══════════════════════════════════════════════════════════════════════

    # "what's on screen right now" -> describe_screen (formatted UIA summary)
    if _WHATS_ON_SCREEN_DEEP_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "describe_screen", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # "is there a button that says X" -> ui_find_text (exact + Button filter)
    m = _BUTTON_CHECK_RE.match(clause)
    if m:
        # Extract the button name from whichever capture group matched
        btn_name = (m.group(1) or m.group(2) or m.group(3) or m.group(4) or "").strip().rstrip("?. ")
        if btn_name:
            # Determine match_mode: "button that says X" -> exact; "X button" -> exact
            args = {
                "text": btn_name,
                "method": "uia",
                "control_type": "Button",
                "match_mode": "exact",
            }
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "ui_find_text", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.93,
            )

    # "did install succeed" -> install_succeeded_check
    if _INSTALL_CHECK_RE.match(clause):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "install_succeeded_check", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.93,
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

    # ═══════════════════════════════════════════════════════════════════════
    # Switch app patterns: deterministic app switching using focus history
    # ═══════════════════════════════════════════════════════════════════════
    clause_stripped = _strip_trailing_punct(clause)
    
    # "go back" / "switch back" / "previous app" / "last app" -> switch_app mode=previous
    if _SWITCH_PREVIOUS_RE.match(clause_stripped):
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "switch_app",
                "args": {"mode": "previous"},
                "continue_on_error": False
            }],
            reply="",
            confidence=0.95,
        )
    
    # "next app" / "cycle apps" -> switch_app mode=next
    if _SWITCH_NEXT_RE.match(clause_stripped):
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "switch_app",
                "args": {"mode": "next"},
                "continue_on_error": False
            }],
            reply="",
            confidence=0.93,
        )
    
    # "switch to X" / "go to X" -> switch_app mode=named
    m = _SWITCH_TO_APP_RE.match(clause_stripped)
    if m:
        target = (m.group(1) or "").strip().strip('"').strip("'")
        # If the target is missing or too ambiguous, defer to LLM.
        if not target or target.lower() in {"it", "this", "that", "something", "anything"}:
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.4)
        
        # Check for ambiguous targets that could be media commands
        target_l = target.lower().strip()
        if target_l in {"the last app", "last app", "previous app", "the previous app"}:
            # This is actually a "switch back" command
            return HybridDecision(
                mode="tool_plan",
                intents=[{
                    "tool": "switch_app",
                    "args": {"mode": "previous"},
                    "continue_on_error": False
                }],
                reply="",
                confidence=0.95,
            )
        
        if target_l in {"the next app", "next app"}:
            # This is actually a "next app" command
            return HybridDecision(
                mode="tool_plan",
                intents=[{
                    "tool": "switch_app",
                    "args": {"mode": "next"},
                    "continue_on_error": False
                }],
                reply="",
                confidence=0.93,
            )
        
        return HybridDecision(
            mode="tool_plan",
            intents=[{
                "tool": "switch_app",
                "args": {"mode": "named", "app": target},
                "continue_on_error": False
            }],
            reply=f"Switching to {target}.",
            confidence=0.92,
        )

    # Open/launch/start X (non-URL) -> open_target.
    # GUARD: Only match imperative commands, NOT questions containing "open".
    # E.g. "did it open" is a question, not "open <target>".
    m = _OPEN_RE.match(clause_stripped)
    if m and not _QUESTION_PREFIX_RE.match(clause_stripped):
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

        # Guard: "minimize all windows/applications/everything" is a bulk command.
        # Do NOT map to minimize_window(title="all windows"); let orchestrator handle.
        target_l = re.sub(r"\s+", " ", target.lower()).strip()
        if re.search(r"\b(?:all\s+(?:windows|apps?|applications)|everything)\b", target_l):
            return HybridDecision(mode="llm", intents=None, reply="", confidence=0.6)

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

    # Click / press / hit / tap X -> desktop_click_uia (or native win-ctl).
    click_decision = _match_click_intent(clause_stripped)
    if click_decision is not None:
        return click_decision

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

    # High-confidence media play/pause patterns (unambiguous commands)
    if re.search(r"\b(?:hit\s+play|hit\s+pause|press\s+play|press\s+pause|play\s*pause|play/pause|play\s+(?:the\s+)?music|play\s+(?:the\s+)?media|play\s+(?:the\s+)?video|pause\s+(?:it|this|that|the\s+music|music|media|video)|resume\s+(?:it|this|that|the\s+music|music|media|video|playback)|unpause)\b", tl):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "media_play_pause", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.92,
        )

    # Lower-confidence bare "play"/"pause"/"resume" (could be ambiguous)
    if re.search(r"\b(?:pause|play|resume)\b", tl):
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

    # Normalize for lightweight keyword detection.
    normalized = _normalize_text_for_routing(raw)

    # =====================================================================
    # HIGH-PRIORITY: Verification / status queries about the LAST action.
    # MUST run before multi-intent extraction to prevent "open" keyword trap.
    # "did the last thing you open actually open successfully" -> get_recent_events
    # "did that work" -> get_recent_events
    # "what just happened" -> get_recent_events
    # =====================================================================
    if _VERIFICATION_QUERY_RE.match(raw):
        logger.info("[ROUTER] matched=verification_query tool=get_recent_events reason=status/verification question")
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_recent_events", "args": {"limit": 5}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # =====================================================================
    # HIGH-PRIORITY: Foreground / app context queries.
    # "what app am I currently using" -> get_window_context
    # "where am I" -> get_window_context
    # MUST run before multi-intent to avoid splitting on keyword "open".
    # =====================================================================
    if _WINDOW_CONTEXT_RE.match(raw):
        logger.info("[ROUTER] matched=window_context_query tool=get_window_context reason=foreground/app query")
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_window_context", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # =====================================================================
    # Phase 16 (pre-split): "click <target> and type <text>"
    # Must run BEFORE multi-intent extraction, which would split on "and".
    # =====================================================================
    cat_decision_early = _match_click_and_type(raw)
    if cat_decision_early is not None:
        return cat_decision_early

    # =====================================================================
    # MULTI-INTENT EXTRACTION (Phase 17): Before any single-intent early
    # exits, check if the utterance contains MULTIPLE tool segments.
    # E.g. "What can you do? Can you tell me the time?"
    #   -> [get_capabilities, get_time]
    # E.g. "Open notepad and then tell me something cool, but then what is the time?"
    #   -> [open_target(notepad), get_time] + leftover="tell me something cool"
    # Only activates when the text genuinely looks like multi-intent.
    # =====================================================================
    if looks_multi_intent(raw) or _MI_CONNECTORS_RE.search(raw) or len(_scan_tool_spans(raw)) >= 2:
        mi_result = extract_multi_intents(raw)
        if mi_result is not None:
            mi_intents, mi_leftover = mi_result
            if len(mi_intents) >= 2 or (len(mi_intents) >= 1 and mi_leftover):
                logger.info(
                    f"[ROUTER] multi_intent extracted {len(mi_intents)} tools, "
                    f"leftover={mi_leftover[:50] if mi_leftover else '(none)'!r}"
                )
                return HybridDecision(
                    mode="tool_plan",
                    intents=mi_intents,
                    reply=(f"__LEFTOVER__:{mi_leftover}" if mi_leftover else ""),
                    confidence=0.93,
                )

    # =====================================================================
    # CAPABILITIES / HELP: single-intent early exit.
    # Must run before needs_reasoning() which would swallow "help".
    # =====================================================================
    if _is_capabilities_query(raw):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_capabilities", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # Time queries: handle fragments and multi-sentence utterances.
    # Must run before needs_reasoning() so short fragments like "Time." don't
    # get treated as conversational streaming reply-only.
    is_creative = _is_creative_request(raw)
    has_time_keyword = bool(
        normalized
        and (
            _TIME_KEYWORDS_ANYWHERE_RE.search(normalized)
            or _TIME_REQUEST_ANYWHERE_RE.search(normalized)
        )
    )
    is_time_fragment = bool(normalized and _looks_like_time_fragment(normalized))

    # - Strong time-keyword phrases ("what's the time", "current time") should route
    #   to tools even if the utterance also contains creative requests.
    # - Bare fragments ("time") should NOT override creative content requests.
    if has_time_keyword or (is_time_fragment and not is_creative):
        leftover = ""
        if has_time_keyword and is_creative:
            leftover = _extract_leftover_around_time(raw, normalized)
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_time", "args": {}, "continue_on_error": False}],
            reply=(f"__LEFTOVER__:{leftover}" if leftover else ""),
            confidence=0.95,
        )

    # Date/day-of-week queries: handle fragments and multi-sentence utterances.
    # Must run before needs_reasoning() so "What is today?" doesn't get routed
    # to LLM reply-only streaming.
    has_date_keyword = bool(
        normalized
        and (
            _DATE_KEYWORDS_ANYWHERE_RE.search(normalized)
            or _DATE_REQUEST_ANYWHERE_RE.search(normalized)
        )
    )
    is_date_fragment = bool(normalized and _looks_like_date_fragment(normalized))

    # - Strong date/day phrases should route to tools even if the utterance also
    #   contains creative requests.
    # - Bare fragments ("today", "date") should NOT override creative content requests.
    if has_date_keyword or (is_date_fragment and not is_creative):
        leftover = ""
        if has_date_keyword and is_creative:
            leftover = _extract_leftover_around_date(raw, normalized)
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "get_time", "args": {}, "continue_on_error": False}],
            reply=(f"__LEFTOVER__:{leftover}" if leftover else ""),
            confidence=0.95,
        )

    # Window perception queries are safe, deterministic, and should NOT be swallowed
    # by the generic reasoning-question heuristic.
    # Route to describe_screen (Phase 14 ground truth) for richer output.
    if _WHATS_ON_MY_SCREEN_RE.match(raw):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "describe_screen", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.93,
        )
    if _LIST_OPEN_WINDOWS_RE.match(raw):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "list_open_windows", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.93,
        )

    # Phase 14: Desktop Ground Truth early-exit routes
    if _WHATS_ON_SCREEN_DEEP_RE.match(raw):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "describe_screen", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.93,
        )
    m = _BUTTON_CHECK_RE.match(raw)
    if m:
        btn_name = (m.group(1) or m.group(2) or m.group(3) or m.group(4) or "").strip().rstrip("?. ")
        if btn_name:
            args = {
                "text": btn_name,
                "method": "uia",
                "control_type": "Button",
                "match_mode": "exact",
            }
            return HybridDecision(
                mode="tool_plan",
                intents=[{"tool": "ui_find_text", "args": args, "continue_on_error": False}],
                reply="",
                confidence=0.93,
            )
    if _INSTALL_CHECK_RE.match(raw):
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "install_succeeded_check", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.93,
        )

    # Phase 15a: Deterministic UI-state tool queries (recent events, open windows)
    # These patterns are defined in ui_state_patterns and route to specific tools.
    # get_recent_events is a pure event-log lookup — no screen perception needed.
    from wyzer.core.ui_state_patterns import is_ui_state_tool_query as _ui_tool_q
    _ui_tool = _ui_tool_q(raw)
    if _ui_tool is not None:
        logger.info(f"[ROUTER] matched=ui_state_tool_query tool={_ui_tool} reason=deterministic UI query")
        _tool_args: dict = {}
        if _ui_tool == "get_recent_events":
            # Extract numeric limit from the query (e.g. "last five actions" → 5)
            _limit = _extract_event_limit(raw)
            if _limit is not None:
                _tool_args["limit"] = _limit
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": _ui_tool, "args": _tool_args, "continue_on_error": False}],
            reply="",
            confidence=0.95,
        )

    # Phase 15: Broad UI-state queries -> describe_screen (perceive first)
    if _is_ui_state_query(raw):
        logger.info("[ROUTER] matched=ui_state_query tool=describe_screen reason=broad UI question")
        return HybridDecision(
            mode="tool_plan",
            intents=[{"tool": "describe_screen", "args": {}, "continue_on_error": False}],
            reply="",
            confidence=0.93,
            _needs_perception=True,
        )

    # Phase 17: Agent-grade perception-first gate.
    # If the query needs perception, tag the decision so the orchestrator
    # knows to run the agent micro-loop.
    from wyzer.core.ui_state_patterns import needs_perception_first as _needs_pf
    _pf_flag = _needs_pf(raw)

    # Phase 16: "click <target> and type <text>" — deterministic fast-path
    # Must run BEFORE the generic click intent to avoid consuming the target.
    cat_decision = _match_click_and_type(raw)
    if cat_decision is not None:
        cat_decision._needs_perception = cat_decision._needs_perception or _pf_flag
        return cat_decision

    # Phase 16b: "type <text>" / "write <text>" — direct typing into focused field
    # Must run BEFORE generic click intent so "type hello" doesn't fall through.
    type_decision = _match_type_direct(raw)
    if type_decision is not None:
        type_decision._needs_perception = type_decision._needs_perception or _pf_flag
        return type_decision

    # Phase 14c: Click / press commands -> desktop_click_uia
    click_decision = _match_click_intent(raw)
    if click_decision is not None:
        click_decision._needs_perception = click_decision._needs_perception or _pf_flag
        return click_decision

    # =====================================================================
    # Phase 14b: Broad screen-description & element-verify via normalized text.
    # These use phrase-containment (not anchored regex) so they catch natural
    # speech like "Oh, what's on my screen? Can you describe it?"
    # MUST run BEFORE needs_reasoning() to prevent LLM fallthrough.
    # =====================================================================
    screen_decision = _match_screen_describe_intent(normalized)
    if screen_decision is not None:
        screen_decision._needs_perception = True  # always for screen describe
        return screen_decision

    element_decision = _match_verify_element_intent(normalized)
    if element_decision is not None:
        element_decision._needs_perception = True  # always for element verify
        return element_decision

    # Try multi-intent parsing: handle mixed tool+LLM utterances like
    # "open notepad and tell me a story" without getting blocked by needs_reasoning().
    if looks_multi_intent(raw):
        try:
            from wyzer.core.multi_intent_parser import parse_multi_intent_with_fallback, parse_multi_intent_partial

            result = parse_multi_intent_with_fallback(raw)
            if result is not None:
                intents, confidence = result
                return HybridDecision(mode="tool_plan", intents=intents, reply="", confidence=confidence, _needs_perception=_pf_flag)

            # Try partial parsing: tool intents + leftover text for LLM
            partial_result = parse_multi_intent_partial(raw)
            if partial_result is not None:
                intents, leftover_text, confidence = partial_result
                if intents and leftover_text:
                    return HybridDecision(
                        mode="tool_plan",
                        intents=intents,
                        reply=f"__LEFTOVER__:{leftover_text}",
                        confidence=confidence,
                        _needs_perception=_pf_flag,
                    )
                elif intents:
                    return HybridDecision(mode="tool_plan", intents=intents, reply="", confidence=confidence, _needs_perception=_pf_flag)
        except Exception:
            # If multi-intent parser fails, fall back to the normal logic below.
            pass

    # Check for reasoning/explanation questions - these go to LLM when we couldn't
    # deterministically extract any tool intents.
    if needs_reasoning(raw):
        logger.debug("[ROUTER] no_match -> LLM (needs_reasoning=True)")
        return HybridDecision(mode="llm", intents=None, reply="", confidence=0.85, _needs_perception=_pf_flag)

    result = _decide_single_clause(raw)
    result._needs_perception = result._needs_perception or _pf_flag
    if result.mode == "tool_plan":
        tool_name = result.intents[0]["tool"] if result.intents else "?"
        logger.debug(f"[ROUTER] matched=single_clause tool={tool_name} reason=_decide_single_clause")
    else:
        logger.debug("[ROUTER] no_match -> LLM (single_clause fallthrough)")
    return result
