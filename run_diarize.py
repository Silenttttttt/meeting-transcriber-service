#!/usr/bin/env python3
"""Standalone diarization worker - runs as its own OS process specifically
so CUDA/VRAM is fully released when it exits, before the main service
process ever loads Whisper. `app/main.py` invokes this as a real subprocess
per diarization request (not an in-process import) - this preserves the
original CLI tool's deliberate design (see its `run_diarize.py`): loading
pyannote's pipeline and Whisper's model in the SAME process back-to-back
left stale CUDA allocations behind even after `del`-ing the first model, so
the original tool always ran diarization in a separate process that fully
exits before Whisper ever loads. This service keeps that property.

Usage:
    python run_diarize.py <wav_path>

On success: prints a JSON array of `[start_sec, end_sec, speaker_id]` to
stdout and exits 0.

On failure: prints a real error message to stderr and exits 1 - it does
NOT print "[]" on failure. A caller must be able to tell "no speakers
found" (a normal, successful empty list) apart from "diarization broke"
(a non-zero exit).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

from app.diarize_core import DiarizationError, diarize_wav


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: run_diarize.py <wav_path>", file=sys.stderr)
        sys.exit(1)

    wav_path = sys.argv[1]
    try:
        segments = diarize_wav(wav_path)
    except DiarizationError as exc:
        print(f"diarization failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print(json.dumps(segments))


if __name__ == "__main__":
    main()
