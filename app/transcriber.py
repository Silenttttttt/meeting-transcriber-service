"""Whisper transcription, ported as-is from the original CLI tool this
service was extracted from (`disc-transcripter/transcriber.py`) - no
behavior changes. Uses `openai-whisper` (imported as `whisper`), not
`faster-whisper`.

Every call loads a fresh model and unloads it again in a `finally` block
(see `_load_model`/`_unload_model`) rather than keeping one warm in memory
across requests - that's an intentional, preserved property of the
original tool (it was written to run once per CLI invocation), and it
means the marginal cost of a cold pod vs. a warm one is just process
startup, not "warm model vs. cold model": every single request pays the
full model-load cost regardless. See the top-level README for why that
matters for this service's scale-to-zero tuning.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
from io import BytesIO

import whisper

_PT_HALLUCINATIONS = re.compile(
    r"legenda[s]?\s*(por|:|[A-Z])|"
    r"sob[\s-]+títulos|"
    r"sônia ruberti|"
    r"adriana zanotto|"
    r"obrigado por assistir|"
    r"inscreva[-\s]se no canal|"
    r"deixe\s+(um\s+)?like|"
    r"^\s*abertura\s*$|"
    r"^\s*encerramento\s*$",
    re.IGNORECASE,
)

_WHISPER_KWARGS = dict(
    temperature=0,
    beam_size=5,
    best_of=5,
    fp16=True,
    no_speech_threshold=0.6,
    logprob_threshold=-1.0,
    compression_ratio_threshold=2.0,
    condition_on_previous_text=False,
    verbose=False,
)


def _load_model():
    import torch

    preferred = os.getenv("WHISPER_MODEL", "large-v3")
    # VRAM requirements (fp16): large-v3 ~6 GB, medium ~3 GB, base ~1 GB
    _VRAM_MIN = {"large-v3": 6.0, "large-v2": 6.0, "large": 6.0, "medium": 3.0, "small": 2.0, "base": 0.8}
    fallback_chain = [preferred, "medium", "base"]
    free_gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
    for model_name in fallback_chain:
        needed = _VRAM_MIN.get(model_name, 9.5)
        if free_gb >= needed:
            if model_name != preferred:
                print(f"[transcriber] VRAM low ({free_gb:.1f}GB free) - using {model_name} instead of {preferred}", flush=True)
            return whisper.load_model(model_name, device="cuda")
    print(f"[transcriber] VRAM too low for any GPU model ({free_gb:.1f}GB free) - using CPU", flush=True)
    return whisper.load_model("base", device="cpu")


def _unload_model(model):
    import gc

    import torch

    del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _transcribe_sync(path: str, language: str) -> str:
    model = _load_model()
    try:
        result = model.transcribe(path, language=language, **_WHISPER_KWARGS)
        text = " ".join(
            s["text"].strip()
            for s in result.get("segments", [])
            if s.get("text", "").strip() and not _PT_HALLUCINATIONS.search(s["text"])
        )
        return text.strip()
    finally:
        _unload_model(model)


def _transcribe_sync_segments(path: str, language: str) -> list:
    model = _load_model()
    try:
        result = model.transcribe(path, language=language, **_WHISPER_KWARGS)
        out = []
        for s in result.get("segments", []):
            text = s.get("text", "").strip()
            if text and not _PT_HALLUCINATIONS.search(text):
                out.append((s["start"], text))
        return out
    finally:
        _unload_model(model)


async def transcribe_audio(audio: BytesIO, language: str = "pt", **_) -> str:
    """Returns the full transcript as one plain string."""
    audio.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio.read())
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _transcribe_sync, tmp_path, language)
    finally:
        os.unlink(tmp_path)


async def transcribe_audio_segments(audio: BytesIO, language: str = "pt") -> list:
    """Returns `[(start_sec: float, text: str), ...]` - the real shape the
    HTTP layer builds its response from."""
    audio.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio.read())
        tmp_path = tmp.name
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _transcribe_sync_segments, tmp_path, language)
    finally:
        os.unlink(tmp_path)
