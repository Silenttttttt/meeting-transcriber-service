# syntax=docker/dockerfile:1

# Official PyTorch image with a matching CUDA/cuDNN runtime already baked
# in - chosen over a bare nvidia/cuda base so torch/torchaudio's CUDA build
# is guaranteed compatible with the driver stack it ships against, instead
# of hoping a separately-pip-installed torch wheel happens to match.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# ffmpeg is a hard runtime dependency of openai-whisper (it shells out to
# it to decode/resample audio); git is needed by pip to fetch a couple of
# pyannote.audio's own git-based transitive dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the Whisper model into the image instead of downloading it on
# first cold start.
#
# Tradeoff (see README "Deployment notes" for the full writeup): this adds
# ~3 GB to the image for large-v3, and re-baking means rebuilding the image
# whenever WHISPER_MODEL changes. The alternative - downloading on first
# request - would otherwise turn EVERY cold start after a scale-to-zero
# into a multi-GB download from openai's CDN, which is a much worse
# experience for a service whose whole point is scaling to zero when idle.
# Loaded on CPU here on purpose: the Docker build environment has no GPU,
# and `whisper.load_model` only needs a device to move weights onto, not to
# actually run anything - CPU is enough to trigger the download+cache.
ARG WHISPER_MODEL=large-v3
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN python -c "import os, whisper; whisper.load_model(os.environ['WHISPER_MODEL'], device='cpu')"

COPY app/ ./app/
COPY run_diarize.py .

EXPOSE 8000

# Doesn't load any model - see app/main.py's /health handler.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
