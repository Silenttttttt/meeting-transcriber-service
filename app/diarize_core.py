"""Speaker diarization via pyannote.audio, ported from the original CLI
tool's `diarize.py` - core algorithm (pipeline load, speaker/transcript
merge) is unchanged. Two deliberate differences from the source:

1. Failures are LOUD here, not silent. The original `diarize_wav()` had a
   broad `except Exception` that returned `[]` on ANY failure (missing
   token, model load error, CUDA OOM, pipeline bug, ...) so a CLI caller
   would just see "no speakers" and move on. That's a bad failure mode for
   an HTTP service: a client that explicitly asked for diarization can't
   tell "genuinely one speaker detected" apart from "diarization silently
   broke" if both return an empty/absent result. Every failure path here
   raises `DiarizationError` with a real message instead, and this module
   also logs it before raising.

2. The original's optional resemblyzer-based enrolled-speaker-name lookup
   (`speaker_profiles.identify_speakers`) is intentionally NOT ported -
   that's a separate, out-of-scope concern (voice fingerprinting /
   enrollment), not part of "transcribe + diarize." Diarized speakers here
   are always the raw pyannote IDs (`SPEAKER_00`, `SPEAKER_01`, ...) unless
   `assign_speakers` renumbers them for display.

Requires `HF_TOKEN` with access to the gated
`pyannote/speaker-diarization-3.1` model - accept the terms at
https://hf.co/pyannote/speaker-diarization-3.1 with the same account the
token belongs to, or every diarization call will fail with a 401/403 from
Hugging Face.
"""
from __future__ import annotations

import gc
import logging
import os

logger = logging.getLogger("meeting-transcriber.diarize")


class DiarizationError(RuntimeError):
    """Diarization could not be performed - the message says why. Never
    raised for "zero speakers found in genuinely quiet audio" - that's a
    normal, successful empty-list result, not an error."""


def _cuda_free() -> None:
    import torch

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _run_pipeline(hf_token: str, wav_path: str) -> list[tuple[float, float, str]]:
    """Load the pipeline, diarize, return raw segments. The pipeline itself
    is freed on function return (refcount -> 0), same as the original."""
    import torch
    import inspect

    from pyannote.audio import Pipeline

    if not torch.cuda.is_available():
        raise DiarizationError(
            "CUDA is not available in this container - diarization requires a GPU "
            "(there is no CPU fallback here; see README)."
        )

    # pyannote.audio renamed this kwarg across versions: older releases (and,
    # confirmed empirically, 3.4.0 - what actually resolves from this
    # project's `pyannote.audio>=3.3.2,<4.0` pin alongside the pinned torch
    # version) take `use_auth_token`; some other releases take `token`
    # instead. Inspecting the real installed signature rather than hardcoding
    # either name means this keeps working across whichever version a given
    # build actually resolves, instead of silently breaking on an
    # unexpected-keyword-argument TypeError the way a hardcoded guess did the
    # first time this was deployed.
    sig = inspect.signature(Pipeline.from_pretrained)
    if "token" in sig.parameters:
        auth_kwargs = {"token": hf_token}
    elif "use_auth_token" in sig.parameters:
        auth_kwargs = {"use_auth_token": hf_token}
    else:
        raise DiarizationError(
            "installed pyannote.audio's Pipeline.from_pretrained() has neither a "
            "'token' nor a 'use_auth_token' parameter - incompatible version installed"
        )
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **auth_kwargs)
    pipeline.to(torch.device("cuda"))
    result_raw = pipeline(wav_path)

    # pyannote >=3.3 wraps the result in a DiarizeOutput; older versions
    # return an Annotation directly.
    if hasattr(result_raw, "speaker_diarization"):
        annotation = result_raw.speaker_diarization
    elif hasattr(result_raw, "itertracks"):
        annotation = result_raw
    else:
        raise DiarizationError(f"unexpected pyannote pipeline output type: {type(result_raw)}")

    return [(turn.start, turn.end, spk) for turn, _, spk in annotation.itertracks(yield_label=True)]


def diarize_wav(wav_path: str) -> list[tuple[float, float, str]]:
    """Returns `[(start_sec, end_sec, speaker_id), ...]` sorted by start.

    Raises `DiarizationError` on any real failure - callers MUST handle
    this explicitly. Unlike the original CLI tool, this never silently
    returns `[]` to mean "something went wrong."
    """
    try:
        import pyannote.audio  # noqa: F401
    except ImportError as exc:
        raise DiarizationError("pyannote.audio is not installed") from exc

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise DiarizationError(
            "HF_TOKEN is not set - diarization requires a Hugging Face token with access to "
            "pyannote/speaker-diarization-3.1"
        )

    try:
        segments = _run_pipeline(hf_token, wav_path)
    except DiarizationError:
        raise
    except Exception as exc:
        logger.exception("diarization pipeline failed")
        raise DiarizationError(f"diarization pipeline failed: {exc}") from exc
    finally:
        try:
            _cuda_free()
        except Exception:
            logger.warning("failed to free CUDA memory after diarization", exc_info=True)

    segments.sort(key=lambda x: x[0])
    return segments


def _nearest_speaker(wc_sec: float, diarization: list[tuple[float, float, str]]) -> str | None:
    """Return the speaker ID overlapping wc_sec, or the nearest one within
    3 seconds if none overlaps exactly."""
    for start, end, spk in diarization:
        if start <= wc_sec <= end:
            return spk
    best_spk, best_dist = None, float("inf")
    for start, end, spk in diarization:
        dist = min(abs(start - wc_sec), abs(end - wc_sec))
        if dist < best_dist:
            best_dist, best_spk = dist, spk
    return best_spk if best_dist < 3.0 else None


def assign_speakers(segs: list, diarization: list, stream_label: str, label_prefix: str) -> list:
    """Replace `stream_label` entries in `segs` with per-speaker labels.

    Unknown `SPEAKER_XX` IDs are numbered sequentially by first appearance
    IN THE TRANSCRIPT (not pyannote's internal speaker count), so a
    ghost/noise speaker that never actually matches a transcribed segment
    is ignored rather than reserving a number.

    segs:         `[[wc_sec, label, text], ...]`
    diarization:  output of `diarize_wav()`
    stream_label: which label to replace, e.g. "remote" or "mic"
    label_prefix: base name for unknown speakers, e.g. "Speaker"
    """
    if not diarization:
        return segs

    raw_matches = []
    seen_order: list[str] = []
    for wc_sec, label, _text in segs:
        spk = _nearest_speaker(wc_sec, diarization) if label == stream_label else None
        raw_matches.append(spk)
        if spk and spk not in seen_order:
            seen_order.append(spk)

    unknown = [s for s in seen_order if s.startswith("SPEAKER_")]
    spk_display = {s: s for s in seen_order if not s.startswith("SPEAKER_")}
    if len(unknown) == 1:
        spk_display[unknown[0]] = label_prefix
    else:
        for i, s in enumerate(unknown, 1):
            spk_display[s] = f"{label_prefix} {i}"

    result = []
    for (wc_sec, label, text), spk in zip(segs, raw_matches):
        if label == stream_label and spk:
            label = spk_display.get(spk, label)
        result.append([wc_sec, label, text])
    return result
