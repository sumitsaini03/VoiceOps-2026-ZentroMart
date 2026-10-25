#!/usr/bin/env python3
"""VoiceOps 2026 ZentroMart Support Agent CLI and Batch Runner.

Usage:
    python run.py <audio_dir>
    python run.py --text "My order still hasn't arrived. Can I cancel it?"
    python run.py --interactive
"""

import argparse
import json
import os
from pathlib import Path
import sys
import dotenv

dotenv.load_dotenv()

from src.agent import ConversationalAgent
from src.voice import VoiceHandler


def process_audio_batch(audio_dir: Path, output_dir: Path) -> int:
    """Process all audio files in audio_dir.

    Saves per file:
    - transcript (.txt)
    - reply text (.txt)
    - reply audio (.wav)
    - chunks used (.json)
    One bad file never stops the batch; exits nonzero if any failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    voice_handler = VoiceHandler()

    if voice_handler.is_blocked():
        print(f"\n[WARNING] Live voice is {voice_handler.status}.")
        print("Set GNANI_API_KEY in your environment or .env to enable speech processing.")
        return 1

    audio_files = sorted(
        [f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg")]
    )

    if not audio_files:
        print(f"No audio files found in directory: {audio_dir}")
        return 0

    print(f"Found {len(audio_files)} audio file(s) in {audio_dir}. Starting batch processing...")
    agent = ConversationalAgent()
    failures = 0
    batch_summary = []

    for idx, audio_file in enumerate(audio_files, 1):
        stem = audio_file.stem
        out_wav = str(output_dir / f"{stem}_reply.wav")
        out_transcript = output_dir / f"{stem}_transcript.txt"
        out_reply = output_dir / f"{stem}_reply.txt"
        out_chunks = output_dir / f"{stem}_chunks.json"

        print(f"[{idx}/{len(audio_files)}] Processing {audio_file.name}...", end=" ", flush=True)

        try:
            transcript, reply_text, saved_audio, chunks_used = voice_handler.process_audio(
                audio_path=str(audio_file),
                agent=agent,
                output_audio_path=out_wav,
            )

            # Save per-file outputs
            out_transcript.write_text(transcript or "", encoding="utf-8")
            out_reply.write_text(reply_text or "", encoding="utf-8")
            out_chunks.write_text(json.dumps(chunks_used, indent=2), encoding="utf-8")

            batch_summary.append({
                "file": audio_file.name,
                "transcript": transcript,
                "reply": reply_text,
                "audio_saved": bool(saved_audio),
                "chunks_used": chunks_used,
                "status": "SUCCESS",
            })
            print("OK")
        except Exception as e:
            failures += 1
            print(f"FAILED ({e})")
            batch_summary.append({
                "file": audio_file.name,
                "error": str(e),
                "status": "ERROR",
            })

    # Save batch summary log
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    print(f"\nBatch completed. Summary saved to {summary_path}")
    print(f"Processed: {len(audio_files)} | Succeeded: {len(audio_files) - failures} | Failed: {failures}")

    return 1 if failures > 0 else 0


def interactive_mode():
    """Interactive text session for quick human testing and demonstrations."""
    print("=" * 60)
    print("ZentroMart Support Agent - Interactive Session")
    print("Type your message below. Type 'exit' or 'quit' to end.")
    print("=" * 60)
    agent = ConversationalAgent()

    while True:
        try:
            user_input = input("\nCustomer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Ending session. Goodbye!")
                break
            reply = agent.respond(user_input)
            print(f"Agent: {reply}")
            if agent.last_chunks_used:
                print(f"  [Retrieved chunks: {', '.join(agent.last_chunks_used)}]")
            if agent.ticket_created:
                print(f"  [Escalation Ticket: {agent.ticket_created.get('ticket_id')}]")
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break


def main():
    parser = argparse.ArgumentParser(description="VoiceOps 2026 ZentroMart Support Agent")
    parser.add_argument("audio_dir", nargs="?", help="Path to directory containing input audio files")
    parser.add_argument("--output_dir", default="sample_outputs", help="Output directory for replies (default: sample_outputs)")
    parser.add_argument("--text", help="Run a single text message through the agent")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat session")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if args.text:
        agent = ConversationalAgent()
        reply = agent.respond(args.text)
        print(f"Agent: {reply}")
        if agent.last_chunks_used:
            print(f"Chunks used: {agent.last_chunks_used}")
        return

    if args.audio_dir:
        audio_path = Path(args.audio_dir)
        if not audio_path.exists() or not audio_path.is_dir():
            print(f"Error: {args.audio_dir} is not a valid directory.")
            sys.exit(1)
        exit_code = process_audio_batch(audio_path, Path(args.output_dir))
        sys.exit(exit_code)

    parser.print_help()


if __name__ == "__main__":
    main()
