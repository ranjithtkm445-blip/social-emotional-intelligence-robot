FROM python:3.10-slim

WORKDIR /app

# System libraries required by opencv, mediapipe, sounddevice/portaudio, librosa,
# and pyttsx3 (which shells out to espeak-ng for TTS on Linux, unlike Windows' sapi5)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Without --server.headless, Streamlit's first-run email opt-in prompt blocks on stdin
# forever in a container (no TTY to answer it), which looks like a silent startup hang.
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
