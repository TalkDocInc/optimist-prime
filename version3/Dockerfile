# Dockerfile
FROM python:3.11-slim

# Set architecture hints (for Apple Silicon host compatibility)
ARG TARGETPLATFORM=linux/amd64

# Install system dependencies (including audio stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    libasound2 \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies (MPS-aware PyTorch via pip)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        ollama \
        openai-whisper \
        pyannote.audio==3.1.0 \
        gradio==4.35.0 \
        scipy \
        numpy \
        librosa \
        reportlab \
        requests \
        sympy \
        scikit-learn

# Copy app code
COPY karma_compass.py .

# Create required directories
RUN mkdir -p /app/reports /app/profiles

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=""  # Optional — set via --build-arg or runtime
ENV OLLAMA_MODEL="gemma4:e4b"

# Run app (with volume mounts for persistence)
CMD ["python", "karma_compass.py"]
