#!/usr/bin/env python3
"""
Wyzer AI Assistant - Phase 1-3
Entry point for running the assistant.

Usage:
    python run.py                    # Run with hotword detection
    python run.py --no-hotword       # Run without hotword (immediate listening)
    python run.py --model medium     # Use different Whisper model
    python run.py --list-devices     # List audio devices
"""
import sys
import argparse
from wyzer.core.logger import init_logger, get_logger
from wyzer.core.config import Config
from wyzer.audio.mic_stream import MicStream


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Wyzer AI Assistant - Voice Assistant with STT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                     # Normal mode with hotword
  python run.py --no-hotword        # Test mode: immediate listening
  python run.py --model medium      # Use Whisper medium model
  python run.py --device 1          # Use specific audio device
  python run.py --list-devices      # List available audio devices
        """
    )
    
    parser.add_argument(
        "--no-hotword",
        action="store_true",
        help="Disable hotword detection (immediate listening mode)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Audio device index or name"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--whisper-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper inference (default: cpu)"
    )
    
    # LLM Brain arguments (Phase 4)
    parser.add_argument(
        "--llm",
        type=str,
        default="ollama",
        choices=["ollama", "off"],
        help="LLM mode: ollama or off (default: ollama)"
    )
    
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:latest",
        help="Ollama model name (default: llama3.1:latest)"
    )
    
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama API base URL (default: http://127.0.0.1:11434)"
    )
    
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=30,
        help="LLM request timeout in seconds (default: 30)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize logger
    init_logger(args.log_level)
    logger = get_logger()
    
    # List devices if requested
    if args.list_devices:
        MicStream.list_devices()
        return 0
    
    # Parse audio device
    audio_device = None
    if args.device:
        try:
            audio_device = int(args.device)
        except ValueError:
            logger.error(f"Invalid device index: {args.device}")
            logger.info("Use --list-devices to see available devices")
            return 1
    
    # Print startup banner
    print("\n" + "=" * 60)
    print("  Wyzer AI Assistant - Phase 4")
    print("=" * 60)
    print(f"  Whisper Model: {args.model}")
    print(f"  Whisper Device: {args.whisper_device}")
    print(f"  Hotword Enabled: {not args.no_hotword}")
    if not args.no_hotword:
        print(f"  Hotword Keywords: {', '.join(Config.HOTWORD_KEYWORDS)}")
    print(f"  LLM Mode: {args.llm}")
    if args.llm == "ollama":
        print(f"  LLM Model: {args.ollama_model}")
        print(f"  LLM URL: {args.ollama_url}")
    print(f"  Sample Rate: {Config.SAMPLE_RATE}Hz")
    print(f"  Log Level: {args.log_level}")
    print("=" * 60 + "\n")
    
    # Import assistant (after logger is initialized)
    try:
        from wyzer.core.assistant import WyzerAssistant
    except ImportError as e:
        logger.error(f"Failed to import assistant: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    
    # Create and start assistant
    try:
        assistant = WyzerAssistant(
            enable_hotword=not args.no_hotword,
            whisper_model=args.model,
            whisper_device=args.whisper_device,
            audio_device=audio_device,
            llm_mode=args.llm,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            llm_timeout=args.llm_timeout
        )
        
        logger.info("Starting assistant... (Press Ctrl+C to stop)")
        assistant.start()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
