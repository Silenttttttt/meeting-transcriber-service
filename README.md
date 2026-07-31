# Meeting Transcriber Service

A small, GPU-backed HTTP service that transcribes a two-track meeting
recording (a "remote" track and a "mic" track - e.g. the other
participants' audio and your own microphone, from any call-recording setup
that keeps them separate) with [OpenAI Whisper](https://github.com/openai/whisper),
and can optionally identify individual speakers per track with
[pyannote.audio](https://github.com/pyannote/pyannote-audio)'s speaker
diarization pipeline.

Extracted from a personal Discord-bot project's transcription core, with
everything Discord-specific (the bot itself, voice-fingerprint speaker
enrollment, note summarization) left behind. What's here is generic:
upload two WAV files, get back a timestamped, speaker-labeled transcript.

## Why two tracks?

Recording a call as two separate mono tracks (what you hear vs. what you
say) rather than one mixed-down stereo/mono file is a deliberate,
preserved design choice, not an artifact of the extraction - it's what
lets diarization be requested independently per track (usually you only
need to diarize the "remote" track, since you already know who "you" are
on the mic track), and it keeps Whisper's transcription of overlapping
speech cleaner than trying to separate it after the fact from a single
mixed file.

## Running it

```bash
docker build -t meeting-transcriber .
docker run --rm --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_xxx \
  meeting-transcriber
```

Requires an NVIDIA GPU and the NVIDIA Container Toolkit on the host -
there is no CPU fallback for diarization (see "GPU requirement" below).
Whisper transcription *can* fall back to CPU if VRAM is too low for the
configured model (see `app/transcriber.py::_load_model`), but that's a
degraded-mode safety net, not something to rely on for real throughput.

## API

### `GET /health`

Liveness/readiness probe. Never loads Whisper or pyannote - just confirms
the process is up and reports GPU visibility:

```json
{"status": "ok", "cuda_available": true, "gpu_name": "NVIDIA GeForce RTX ..."}
```

### `POST /transcribe`

`multipart/form-data`:

| Field          | Type | Required | Default | Meaning                                  |
|----------------|------|----------|---------|-------------------------------------------|
| `remote`       | file | yes      | -       | Remote/other-party track (WAV)            |
| `mic`          | file | yes      | -       | Local microphone track (WAV)              |
| `language`     | text | no       | `pt`    | Whisper language code (`en`, `pt`, ...)   |
| `diarize`      | bool | no       | `false` | Diarize the `remote` track                |
| `diarize_mic`  | bool | no       | `false` | Diarize the `mic` track                   |

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "remote=@remote.wav" \
  -F "mic=@mic.wav" \
  -F "language=en" \
  -F "diarize=true"
```

Response:

```json
{
  "language": "en",
  "diarize": true,
  "diarize_mic": false,
  "segments": [
    {"start": 0.42, "speaker": "mic", "text": "Hey, can you hear me?"},
    {"start": 3.10, "speaker": "Speaker 1", "text": "Yeah, loud and clear."}
  ],
  "diarization": {
    "remote": [[0.0, 4.8, "SPEAKER_00"], [5.1, 9.3, "SPEAKER_01"]],
    "mic": null
  }
}
```

- `segments` is the merged, chronological transcript across both tracks.
  `speaker` is the fixed track name (`"remote"`/`"mic"`) unless that track
  was diarized, in which case it's replaced with the matched speaker label
  (an enrolled/raw pyannote ID renumbered as `"Speaker 1"`, `"Speaker 2"`,
  ... in order of first appearance in the transcript - see
  `app/diarize_core.py::assign_speakers`).
- `diarization.remote`/`diarization.mic` are the raw pyannote output for
  each track that was actually diarized (`[start_sec, end_sec,
  speaker_id]`), or `null` if that track wasn't diarized.

If diarization is requested and fails (missing/invalid `HF_TOKEN`, model
load error, CUDA OOM, etc.), the request fails with **HTTP 500** and a real
error message - it never silently degrades to "diarization just didn't
happen." See "Failure behavior" below.

## GPU requirement

Diarization (`diarize`/`diarize_mic`) has **no CPU fallback**:
`app/diarize_core.py` hardcodes `pipeline.to(torch.device("cuda"))` and
raises immediately if `torch.cuda.is_available()` is false. This matches
the source tool's own design (pyannote's pipeline on CPU is unusably slow
for anything beyond a toy clip) - it was just made an explicit, loud error
here instead of a silent empty result.

Whisper transcription has a soft VRAM-aware fallback chain
(`large-v3 -> medium -> base`, then CPU as a last resort - see
`_load_model`), preserved unchanged from the source tool.

## Hugging Face gated model (diarization prerequisite)

`pyannote/speaker-diarization-3.1` is a gated model. Before `HF_TOKEN` will
work:

1. Log into huggingface.co with the account the token belongs to.
2. Visit https://hf.co/pyannote/speaker-diarization-3.1 and accept the
   terms (and, if prompted, the terms of the segmentation model it depends
   on, `pyannote/segmentation-3.0`).
3. Generate a read-scoped token at https://hf.co/settings/tokens and set
   it as `HF_TOKEN`.

Without this, every diarization request fails with a clear 401/403-derived
`DiarizationError` - it's a real, configured prerequisite, not a bug.

## Failure behavior (a deliberate change from the source tool)

The original CLI tool's diarization function caught every exception and
returned an empty list - reasonable for a one-off script where a human
reads the output and can tell "no speakers" apart from "something broke"
from context. That's a bad failure mode for an HTTP API: a client that
explicitly asked for diarization has no way to distinguish "diarization
found one speaker" from "diarization silently failed" if both come back
looking the same. Every failure path in `app/diarize_core.py` and
`run_diarize.py` now raises/exits loudly with a real message, which
`app/main.py` turns into an HTTP 500 with that message as the detail.

## Architecture notes

- **Subprocess-isolated diarization.** `run_diarize.py` runs as its own OS
  process (invoked via `subprocess.run`, not imported in-process) so its
  CUDA context - and pyannote's pipeline VRAM - is fully released when the
  process exits, before Whisper ever loads in the main service process.
  This is ported directly from the source tool's own design; it exists
  because loading pyannote's pipeline and Whisper's model back-to-back in
  the same process left VRAM allocations behind that `del` + `empty_cache`
  alone didn't fully reclaim.
- **No timing-sidecar reconciliation.** The source tool had a
  `<wav>.timing.json` mechanism to convert in-file timestamps to
  wall-clock time across non-contiguous Discord voice-channel audio chunks
  with real silence gaps between them. That concept doesn't apply to a
  generic two-continuous-WAV-file upload, so it wasn't ported - `start`
  timestamps here are plain in-file offsets in seconds.
- **No per-request model caching.** `transcribe_audio_segments` loads a
  fresh Whisper model and unloads it again after every call (ported
  unchanged from the source tool - it was written for one-shot CLI runs).
  This means every `/transcribe` call pays the full model-load cost
  regardless of whether the pod is "warm" or just cold-started - a cold
  pod only adds process-startup time (torch/CUDA init, a few seconds) on
  top of that, not "cold model vs. warm model." This is why the deployment
  can reasonably scale to zero without the cold-start penalty being much
  worse than steady-state per-request latency.
- **Pre-baked Whisper weights.** The Docker image downloads and caches the
  configured `WHISPER_MODEL` at build time (see the Dockerfile), so a
  freshly scheduled pod doesn't need to download several GB from OpenAI's
  CDN before it can serve its first real request. Trade-off: a larger
  image (~3 GB extra for `large-v3`) and a rebuild whenever `WHISPER_MODEL`
  changes, in exchange for consistent cold-start latency.

## What was deliberately left out of this extraction

This service only covers transcription + diarization. Left behind, on
purpose:

- The Discord voice bot itself (recording, per-user audio capture).
- Voice-fingerprint speaker enrollment/identification (a separate,
  resemblyzer-based concern - diarized speakers here are always the raw
  pyannote/renumbered IDs, never a real enrolled name).
- Meeting-note summarization (a separate, LLM-based concern).

## Configuration

| Env var         | Required                | Meaning                                            |
|-----------------|--------------------------|-----------------------------------------------------|
| `HF_TOKEN`      | only if diarizing        | Hugging Face token with access to the gated model  |
| `WHISPER_MODEL` | no (defaults `large-v3`) | Whisper model size                                 |

## License

MIT - see [LICENSE](LICENSE).
