"""HTTP wrapper around the transcribe+diarize pipeline extracted from a
Discord-meeting-recorder CLI tool. Two endpoints:

  GET  /health      - liveness/readiness; never loads a model.
  POST /transcribe   - the real pipeline: two WAV uploads (remote + mic),
                        optional per-track diarization.

Design notes (see README for the full writeup):

- Dual-track (remote/mic) upload is a deliberate, preserved feature of the
  original tool, not an accident - it's built for the "two separate audio
  feeds of one conversation" shape a call-recording setup naturally
  produces, and diarization can be requested independently per track.
- Diarization runs as a subprocess (`run_diarize.py`), not an in-process
  import - see that file's docstring for why (CUDA memory isolation from
  Whisper, ported from the original tool's own design).
- Every request pays Whisper's full model-load cost - `transcribe_audio_segments`
  loads and unloads the model per call by design (ported unchanged from the
  source tool). This matters for the deployment's scale-to-zero tuning: a
  cold pod adds only process-startup time on top of that, not "cold model
  vs. warm model."
- No timing-sidecar / wall-clock reconciliation logic was ported. The
  original tool's `<wav>.timing.json` mechanism existed only to stitch
  together non-contiguous Discord voice-channel audio chunks with real
  silence gaps between them - meaningless for a generic service that just
  accepts two continuous WAV files. Segment start times returned here are
  plain in-file offsets.
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.diarize_core import assign_speakers
from app.transcriber import transcribe_audio_segments

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("meeting-transcriber")

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIARIZE = REPO_ROOT / "run_diarize.py"


def _log_gpu_info() -> None:
    """Logged once at startup - the real, checkable proof (via `kubectl logs`)
    that this pod actually has GPU device access, not just the k8s resource
    claim satisfied."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("CUDA available: %s (%.1f GB)", name, total_mem_gb)
        else:
            logger.warning("CUDA is NOT available - transcription will fall back to CPU (slow), "
                            "and diarization will fail outright (no CPU fallback).")
    except Exception:
        logger.exception("failed to query GPU info at startup")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _log_gpu_info()
    yield


app = FastAPI(
    title="Meeting Transcriber Service",
    description="Whisper transcription + pyannote speaker diarization for two-track (remote+mic) meeting recordings.",
    version="0.1.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    gpu_name: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe target. Deliberately does NOT load Whisper
    or pyannote - only confirms the process is up and reports GPU
    visibility, which is cheap (`torch.cuda.is_available()` doesn't load
    any model weights)."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        logger.exception("health check: failed to query CUDA state")
        cuda_available = False
        gpu_name = None
    return HealthResponse(status="ok", cuda_available=cuda_available, gpu_name=gpu_name)


def _run_diarize_subprocess(wav_path: str) -> list:
    """Runs `run_diarize.py` as a real subprocess against `wav_path` and
    returns its parsed `[start, end, speaker_id]` list. Raises RuntimeError
    with the subprocess's real stderr on failure - never silently returns
    an empty list, so the endpoint can turn this into a clear HTTP error
    instead of returning "successful" JSON with diarization quietly
    missing."""
    result = subprocess.run(
        [sys.executable, str(RUN_DIARIZE), wav_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        logger.error("diarization subprocess failed (exit=%s): %s", result.returncode, stderr)
        last_line = stderr.splitlines()[-1] if stderr else "unknown error (no stderr captured)"
        raise RuntimeError(last_line)
    stdout = result.stdout.strip()
    try:
        return json.loads(stdout.splitlines()[-1]) if stdout else []
    except Exception as exc:
        raise RuntimeError(f"diarization subprocess produced unparseable stdout: {stdout!r}") from exc


def _write_temp_wav(data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


@app.post("/transcribe")
async def transcribe(
    remote: UploadFile = File(..., description="Remote/other-party audio track (WAV)"),
    mic: UploadFile = File(..., description="Local microphone audio track (WAV)"),
    language: str = Form("pt", description="Whisper language code, e.g. 'pt', 'en'."),
    diarize: bool = Form(False, description="Diarize the remote track."),
    diarize_mic: bool = Form(False, description="Diarize the mic track."),
) -> dict:
    """Transcribes both tracks and, if requested, diarizes them. Returns
    real structured JSON - see README for the response shape."""
    remote_bytes = await remote.read()
    mic_bytes = await mic.read()

    tmp_paths: list[str] = []
    try:
        remote_diarization: list = []
        mic_diarization: list = []

        # Diarize BEFORE Whisper loads - the subprocess fully exits and
        # the driver reclaims VRAM before Whisper ever touches the GPU,
        # exactly like the original tool.
        if diarize:
            remote_path = _write_temp_wav(remote_bytes)
            tmp_paths.append(remote_path)
            try:
                remote_diarization = _run_diarize_subprocess(remote_path)
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=f"remote diarization failed: {exc}") from exc

        if diarize_mic:
            mic_path = _write_temp_wav(mic_bytes)
            tmp_paths.append(mic_path)
            try:
                mic_diarization = _run_diarize_subprocess(mic_path)
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=f"mic diarization failed: {exc}") from exc

        # Whisper transcription failures (a real CUDA OutOfMemoryError was
        # observed live on this cluster's shared desktop/GPU-node machine,
        # competing with other real GPU usage on the same box) previously
        # propagated as an unhandled exception, which FastAPI's default
        # handler turns into a plain-text, non-JSON 500 - failing the same
        # "always return well-formed JSON" contract diarization failures
        # already respected. Caught here the same way for the same reason.
        try:
            remote_segments = await transcribe_audio_segments(io.BytesIO(remote_bytes), language)
            mic_segments = await transcribe_audio_segments(io.BytesIO(mic_bytes), language)
        except Exception as exc:
            logger.exception("transcription failed")
            raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc

        all_segs: list = []
        for wav_sec, text in remote_segments:
            all_segs.append([wav_sec, "remote", text])
        for wav_sec, text in mic_segments:
            all_segs.append([wav_sec, "mic", text])
        all_segs.sort(key=lambda x: x[0])

        if remote_diarization:
            all_segs = assign_speakers(all_segs, remote_diarization, stream_label="remote", label_prefix="Speaker")
        if mic_diarization:
            all_segs = assign_speakers(all_segs, mic_diarization, stream_label="mic", label_prefix="Speaker")

        return {
            "language": language,
            "diarize": diarize,
            "diarize_mic": diarize_mic,
            "segments": [{"start": s, "speaker": label, "text": text} for s, label, text in all_segs],
            "diarization": {
                "remote": remote_diarization or None,
                "mic": mic_diarization or None,
            },
        }
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
