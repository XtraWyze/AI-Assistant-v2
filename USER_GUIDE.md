# 🎙️ Wyzer User Guide

> **Your friendly local voice assistant for Windows!**

Welcome to Wyzer! This guide will help you get started and make the most of your new AI assistant.

---

## 🚀 Quick Start (After Installation)

### Starting Wyzer

**Option 1: Double-click the launcher**
- Just double-click `run.bat` in your Wyzer folder

**Option 2: Run from command line**
```
python run.py
```

**Option 3: Quiet mode (cleaner output)**
```
python run.py --quiet
```

### Your First Conversation

1. **Wait for startup** - Wyzer will say "Ready" when it's listening
2. **Say the wake word** - "Hey Wyzer" or just "Wyzer"
3. **Ask something** - "What time is it?"
4. **Listen to the response** - Wyzer will speak back to you!

That's it! You're using Wyzer! 🎉

---

## 🗣️ Wake Words

To get Wyzer's attention, just say:

| Wake Word | When to Use |
|-----------|-------------|
| **"Hey Wyzer"** | Most reliable - works great in noisy environments |
| **"Wyzer"** | Quick and casual when it's quiet |

💡 **Tip:** After Wyzer responds, you have ~3 seconds to ask a follow-up question without saying the wake word again!

---

## ✨ What Can Wyzer Do?

### ⏰ Time & Date
| Say This | What Happens |
|----------|--------------|
| "What time is it?" | Gets the current time |
| "What's today's date?" | Gets today's date |

### 🌤️ Weather
| Say This | What Happens |
|----------|--------------|
| "What's the weather?" | Weather for your location |
| "What's the weather in Paris?" | Weather for any city |
| "Is it going to rain?" | Weather forecast |

### 📂 Open Apps & Folders
| Say This | What Happens |
|----------|--------------|
| "Open Chrome" | Opens Google Chrome |
| "Open Spotify" | Opens Spotify |
| "Open Downloads" | Opens your Downloads folder |
| "Open Documents" | Opens your Documents folder |
| "Open Steam" | Opens Steam |

### 🪟 Control Windows
| Say This | What Happens |
|----------|--------------|
| "Minimize Chrome" | Minimizes Chrome window |
| "Maximize Spotify" | Makes Spotify fullscreen |
| "Close Notepad" | Closes Notepad |
| "Focus Discord" | Brings Discord to the front |
| "Move Chrome to monitor 2" | Moves window to second screen |

### 🎵 Media Controls
| Say This | What Happens |
|----------|--------------|
| "Pause" or "Play" | Toggle play/pause |
| "Next track" or "Skip" | Skip to next song |
| "Previous" | Go back a song |
| "What's playing?" | Shows current song info |

### 🔊 Volume Control
| Say This | What Happens |
|----------|--------------|
| "Volume up" | Increases volume |
| "Volume down" | Decreases volume |
| "Set volume to 50%" | Sets specific volume |
| "Mute" | Mutes audio |
| "Mute Spotify" | Mutes just Spotify |

### ⏱️ Timers
| Say This | What Happens |
|----------|--------------|
| "Set a timer for 5 minutes" | Starts a countdown |
| "Timer for 30 seconds" | Quick timer |
| "How much time left?" | Checks timer status |
| "Cancel the timer" | Stops the timer |

### 💾 Storage & Drives
| Say This | What Happens |
|----------|--------------|
| "How much space do I have?" | Shows drive space |
| "Open D drive" | Opens D: in Explorer |
| "Scan my drives" | Gets detailed drive info |

### 🔍 Google Search
| Say This | What Happens |
|----------|--------------|
| "Google cats" | Searches Google for cats |
| "Search for pizza recipes" | Opens Google search |

### 💻 System Info
| Say This | What Happens |
|----------|--------------|
| "What's my system info?" | CPU, RAM, disk info |
| "How many monitors?" | Display info |
| "Monitor info" | Detailed screen info |

---

## 🔗 Combine Multiple Commands!

Wyzer understands when you want to do multiple things at once:

| Say This | What Happens |
|----------|--------------|
| "Open Spotify and play music" | Opens Spotify, then plays |
| "What time is it and what's the weather" | Gets time, then weather |
| "Close Chrome and open Firefox" | Closes one, opens other |
| "Minimize Discord and focus Spotify" | Manages both windows |

**Connecting words you can use:**
- "and" - "Open Chrome and minimize Spotify"
- "then" - "Open Spotify then play music"
- "and then" - "Check the time and then open Chrome"

---

## 🛑 How to Interrupt Wyzer

If Wyzer is talking and you want to stop it:

**Just say the wake word!** - "Hey Wyzer" or "Wyzer"

This immediately stops Wyzer and gets ready for your next command. This is called "barge-in."

---

## 💬 Follow-Up Mode

After Wyzer responds, you have a few seconds to ask another question **without saying the wake word**:

```
You: "Hey Wyzer"
Wyzer: "Yes?"
You: "What time is it?"
Wyzer: "It's 3:45 PM"
You: "And what's the weather?"  ← No wake word needed!
Wyzer: "Currently sunny and 72 degrees"
```

**To exit follow-up mode, say:**
- "No"
- "That's all"
- "Stop"
- "Never mind"

---

## ⚙️ Helpful Startup Options

| Command | What It Does |
|---------|--------------|
| `python run.py` | Normal mode |
| `python run.py --quiet` | Cleaner output (less debug info) |
| `python run.py --no-hotword` | No wake word needed (always listening) |
| `python run.py --list-devices` | Shows available microphones |

### Choose Your LLM Backend

| Command | What It Does |
|---------|--------------|
| `python run.py` | Uses llama.cpp (default, recommended) |
| `python run.py --llm ollama` | Uses Ollama instead |
| `python run.py --llm off` | Tools only, no AI chat |

---

## 🎯 Tips for Best Results

### Speaking Clearly
- Speak at a normal pace
- Wait for the "listening" cue before speaking
- Pause briefly at the end of your sentence

### Common Phrases That Work Great
- "Open [app name]"
- "Close [app name]"
- "What's the [time/weather/date]?"
- "Set volume to [number] percent"
- "Timer for [number] minutes"
- "Google [search term]"

### If Wyzer Doesn't Understand
- Try rephrasing your request
- Be more specific: "Open Google Chrome" instead of just "Open browser"
- Make sure your microphone is working

---

## 🔧 Customize Your Apps

### Add Custom App Shortcuts

Create or edit `wyzer/local_library/aliases.json`:

```json
{
  "code": "C:\\Path\\To\\VSCode\\Code.exe",
  "work folder": "D:\\Work\\Projects",
  "my game": "C:\\Games\\MyGame.exe"
}
```

Now you can say "Open code" or "Open work folder"!

### Refresh the App Library

If you install new apps, tell Wyzer:
- "Refresh the library"
- "Scan for apps"

---

## ❓ Troubleshooting

### Wyzer doesn't hear me
- Check your microphone is connected
- Run `python run.py --list-devices` to see available mics
- Try speaking louder or closer to the mic

### Wyzer doesn't recognize apps
- Say "Refresh the library" to scan for new apps
- Add the app to `aliases.json` (see above)
- Use the full app name: "Open Google Chrome" not "Open Chrome"

### Responses are slow
- Try `--quiet` mode to reduce overhead
- Use a smaller Whisper model: `python run.py --model tiny`
- Make sure your LLM is running (llama.cpp or Ollama)

### No speech output
- Check your speakers/headphones are connected
- Check system volume isn't muted
- Verify Piper TTS files exist in `wyzer/assets/piper/`

---

## 🎉 You're Ready!

That's everything you need to know to start using Wyzer! Just remember:

1. **Say "Hey Wyzer"** to wake it up
2. **Speak your command** naturally  
3. **Listen to the response**
4. **Ask follow-up questions** within a few seconds (no wake word needed)
5. **Say the wake word again** anytime to interrupt

Have fun with your new voice assistant! 🚀

---

*For more technical documentation, see [WYZER_DOCUMENTATION.md](WYZER_DOCUMENTATION.md)*
