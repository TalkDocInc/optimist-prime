# dynamic_recorder.py
import webrtcvad
import pyaudio
import wave
import threading
import time
import os
import json
import requests
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException
import uvicorn
import queue
from contextlib import contextmanager

# ——— CONFIG ———
SILENCE_THRESHOLD_SEC = 30  # auto-stop after this many seconds of silence
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_MS = 30  # 30ms frames (1 frame = 480 samples at 16kHz)
SILENCE_FRAMES_REQ = int(SILENCE_THRESHOLD_SEC * 1000 / FRAME_MS)

RECORDINGS_DIR = Path(os.environ.get("KARMA_RECORDINGS_DIR", os.path.expanduser("~/KarmaRecordings")))
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# FastAPI server for HTTP control
app = FastAPI(title="Dynamic Recorder")
recording_active = False
recording_thread = None
stop_event = threading.Event()
frames_queue = queue.Queue()
recording_start_time = None

# Global silence counter (reset on speech)
silent_frames = 0


@contextmanager
def vad_context():
    """Safe VAD context manager."""
    vad = webrtcvad.Vad(2)  # Aggressiveness: 2 (balanced)
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=int(RATE * FRAME_MS / 1000)
    )
    try:
        yield vad, audio, stream
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def _record_loop():
    """Background loop: records until silence threshold or /stop is called."""
    global recording_active, silent_frames, frames_queue, recording_start_time
    recording_start_time = datetime.now().isoformat()

    with vad_context() as (vad, audio, stream):
        print("[Dynamic Recorder] Starting recording...")
        silent_frames = 0

        try:
            while not stop_event.is_set():
                frame = stream.read(int(RATE * FRAME_MS / 1000), exception_on_overflow=False)
                is_speech = vad.is_speech(frame, RATE)

                if is_speech:
                    frames_queue.put(frame)
                    silent_frames = 0  # reset counter
                else:
                    silent_frames += 1
                    if silent_frames > SILENCE_FRAMES_REQ:
                        break  # auto-stop on silence

                # Check for manual /stop
                if stop_event.is_set():
                    break

        except Exception as e:
            print(f"[Dynamic Recorder] Recording error: {e}")
        finally:
            recording_active = False
            # Save recording
            _save_recording()


def _save_recording():
    """Save frames to WAV and auto-trigger karma_compass analysis."""
    global frames_queue
    frames = list(frames_queue.queue)
    if not frames:
        print("[Dynamic Recorder] No frames collected.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RECORDINGS_DIR / f"call_{timestamp}.wav"

    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # pyaudio.paInt16 = 2 bytes
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    print(f"[Dynamic Recorder] Saved: {filepath}")

    # Auto-trigger karma analysis
    _trigger_analysis(filepath, timestamp)


def _trigger_analysis(filepath: Path, timestamp: str):
    """Call karma_compass.py analysis via HTTP or subprocess."""
    # Option 1: HTTP call (if karma_compass app is running on port 7860)
    try:
        # Prepare form-data (Gradio expects file upload)
        with open(filepath, "rb") as f:
            files = {"audio": ("recording.wav", f, "audio/wav")}
            data = {
                "s1_name": "speaker_1",
                "s2_name": "speaker_2",
                "notes": f"Dynamic recording from {timestamp}"
            }
            resp = requests.post(
                "http://localhost:7860/analyze",  # custom endpoint (see below)
                files=files,
                data=data,
                timeout=60
            )
        if resp.status_code == 200:
            print(f"[Dynamic Recorder] Analysis triggered: {resp.json()}")
        else:
            print(f"[Dynamic Recorder] Gradio analyze failed: {resp.status_code}")
    except Exception as e:
        print(f"[Dynamic Recorder] Error triggering analysis: {e}")
        print("[Dynamic Recorder] Fallback: print analysis-ready JSON")
        # Optional: print for manual analysis
        print(f"\n[MANUAL NEXT STEP] Run: python karma_compass.py --audio {filepath}")


# ——— FASTAPI ENDPOINTS ———
@app.post("/start_recording")
def start_recording():
    global recording_active, recording_thread, stop_event
    if recording_active:
        raise HTTPException(status_code=400, detail="Recording already in progress")
    recording_active = True
    stop_event.clear()
    frames_queue.queue.clear()
    recording_thread = threading.Thread(target=_record_loop, daemon=True)
    recording_thread.start()
    return {
        "status": "recording started",
        "silent_threshold_sec": SILENCE_THRESHOLD_SEC,
        "recordings_dir": str(RECORDINGS_DIR)
    }


@app.post("/stop_recording")
def stop_recording():
    global recording_active
    if not recording_active:
        raise HTTPException(status_code=400, detail="No active recording")
    stop_event.set()
    return {
        "status": "recording stopped",
        "message": "Will save shortly once silence threshold is met or buffer flushed"
    }


@app.get("/status")
def status():
    return {
        "recording_active": recording_active,
        "silent_frames_remaining": max(0, SILENCE_FRAMES_REQ - silent_frames),
        "start_time": recording_start_time
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ——— CLI ENTRY POINT ———
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dynamic Recorder with VAD")
    parser.add_argument("--port", type=int, default=8001, help="FastAPI port")
    args = parser.parse_args()

    print("[Dynamic Recorder] Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)