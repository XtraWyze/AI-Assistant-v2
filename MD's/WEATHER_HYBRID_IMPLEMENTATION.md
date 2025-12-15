# Weather Bypass Implementation with Hybrid Scoring

## Summary
Implemented weather/temperature/forecast query detection to bypass LLM and route directly to the `get_weather_forecast` tool using hybrid scoring, similar to how "open" commands are handled.

## Changes Made

### 1. **wyzer/core/hybrid_router.py**

#### Added Weather Pattern Recognition
```python
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
```

#### Added Weather Detection Logic
Inserted after time query check in `_decide_single_clause()`:
- Detects any weather-related query using `_WEATHER_RE` pattern
- Extracts location when present using regex: `\b(?:in|for|at|near)\s+(.+?)(?:\?|$)`
- Filters out common non-location words (it, this, here, there, outside)
- Routes to `get_weather_forecast` tool with high confidence (0.92)
- Location is passed as optional argument when extracted

## Features

### Supported Query Patterns
- Basic: "what's the weather", "weather", "forecast", "temperature"
- Temperature variations: "temp", "is it hot", "how cold", "is it warm"
- Forecasts: "forecast", "what's the forecast"
- Condition checks: "is it raining", "will it snow", "is it cold"
- Location-specific: "weather in [location]", "temperature for [location]", "forecast at [location]"

### Example Behavior

| Query | Route | Tool | Location |
|-------|-------|------|----------|
| "what's the weather" | tool_plan | get_weather_forecast | (none) |
| "weather in new york" | tool_plan | get_weather_forecast | new york |
| "temperature in london" | tool_plan | get_weather_forecast | london |
| "is it raining" | tool_plan | get_weather_forecast | (none) |
| "forecast for paris" | tool_plan | get_weather_forecast | paris |

### Confidence Levels
- Weather queries: **0.92** (high confidence, deterministic routing)
- Comparable to: open commands (0.90), time queries (0.95)

## Integration Points

### Tool Used
- **Tool Name:** `get_weather_forecast`
- **Location:** `wyzer/tools/get_weather_forecast.py`
- **Arguments:**
  - `location` (optional): City/location name (e.g., "Seattle", "Paris, France")
  - `days` (optional): Number of forecast days (1-14, default 3)
  - `units` (optional): Unit system (metric/imperial/fahrenheit/celsius)

### Bypass Mechanism
Unlike LLM-based weather queries which require language understanding:
- **Before:** "What's the weather in New York?" → LLM processes → Tool execution
- **After:** "What's the weather in New York?" → Regex match → Direct tool execution (0.92 confidence)

### No LLM Call
Weather queries completely bypass the LLM for:
- Reduced latency (direct tool execution)
- Deterministic behavior (consistent results)
- Lower resource usage (no LLM inference needed)

## Testing

Test coverage added in `scripts/test_hybrid_router.py`:
```python
{
    "text": "what's the weather",
    "expect_route": "tool_plan",
    "expect_tool": "get_weather_forecast",
    "expect_llm_calls": 0,
    "expect_args": {},
},
{
    "text": "weather in new york",
    "expect_route": "tool_plan",
    "expect_tool": "get_weather_forecast",
    "expect_llm_calls": 0,
    "expect_args": {"location": "new york"},
},
```

## Architecture Alignment

Follows the established hybrid routing pattern:
1. **Pattern Recognition:** Regex-based detection of obvious commands
2. **Location Extraction:** Intelligent parameter extraction from user input
3. **Tool Plan Generation:** Direct intents array with tool and arguments
4. **Confidence Scoring:** High confidence (0.92) for deterministic routing
5. **LLM Bypass:** No LLM call for detected weather queries

## Files Modified
- `wyzer/core/hybrid_router.py` - Added weather pattern and detection logic
- `scripts/test_hybrid_router.py` - Added weather test cases

## Files Not Modified
- `wyzer/tools/get_weather_forecast.py` - Already supports location parameter
