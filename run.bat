@echo off
REM Wyzer AI Assistant Launcher
REM This batch file runs the Wyzer AI Assistant

REM Run with virtual environment Python
REM Note: use_memories defaults to ON. Use --no-memories flag to disable.
venv_new\Scripts\python.exe run.py --log-level DEBUG %*

REM Pause to see any errors
pause
