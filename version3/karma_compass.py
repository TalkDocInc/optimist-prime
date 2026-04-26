# karma_compass.py
import torch
import whisper
import ollama
import numpy as np
import re
import itertools
from scipy.stats import poisson
from pyannote.audio import Pipeline
import sqlite3
from datetime import datetime
import json
import os
import gradio as gr

# ——— CONFIG ———
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Required for Pyannote (get free at huggingface.co/settings/tokens)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
DB_PATH = "karma_trajectory.db"

# Use Apple Silicon GPU (MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Karma Compass] Using Device: {device}")


# In karma_compass.py (FastAPI layer)
from fastapi import FastAPI, UploadFile, File
import tempfile
import gradio as gr

# Add FastAPI app (if not already present)
app = FastAPI(title="Karma Compass API")

@app.post("/analyze")
async def analyze_recording(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    # Reuse your existing analyze_audio logic
    result = analyze_audio(tmp_path, "call_s1", "call_s2", "Dynamic recording analysis")
    return json.loads(result)

# ——— PRIME ENGINE (Your Core Innovation) ———
class PrimeComplexityEngine:
    def __init__(self):
        # Pre-compute primes for a-z
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
        self.char_map = {chr(i + 97): self.primes[i] for i in range(26)}
        self.baseline_lambda = 0.08  # Expected prime-sum density in random speech

    def is_prime(self, n: int) -> bool:
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def get_word_prime_value(self, word: str) -> int:
        clean_word = re.sub(r'[^a-z]', '', word.lower())
        return sum(self.char_map.get(c, 0) for c in clean_word)

    def calculate_structural_complexity(self, text: str) -> tuple:
        """
        Implements the 'Prefix-Sum' and 'Combinatorial Window' logic.
        Checks: W1, W1+W2, W1+W2+W3... and deep subsets in recent window.
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words: return 0.0, 0

        word_values = [self.get_word_prime_value(w) for w in words]
        prime_hits = 0
        total_tests = 0

        # 1. Prefix Sums (The 'Building' pattern: W1, W1+W2, W1+W2+W3...)
        current_prefix_sum = 0
        for val in word_values:
            current_prefix_sum += val
            total_tests += 1
            if self.is_prime(current_prefix_sum):
                prime_hits += 1

        # 2. Deep Combinatorial Windows (Limited to 10 words to prevent O(2^n) explosion)
        window_size = min(len(word_values), 10)
        recent_words = word_values[-window_size:]
        for r in range(2, window_size):
            for subset in itertools.combinations(recent_words, r):
                total_tests += 1
                if self.is_prime(sum(subset)):
                    prime_hits += 1

        density = prime_hits / total_tests if total_tests > 0 else 0.0
        return density, total_tests

    def calculate_p_value(self, observed_density: float, total_tests: int) -> float:
        """
        Calculates the probability that the observed prime density
        is a result of random chance (Poisson-based significance).
        """
        k = int(observed_density * total_tests)
        if k == 0: return 1.0
        try:
            # Probability of seeing k or more successes in a Poisson distribution
            # based on the natural speech lambda.
            p_val = 1 - poisson.cdf(k, self.baseline_lambda * total_tests)
            return max(0.0, min(1.0, p_val))  # Clamp [0,1]
        except:
            return 1.0

# ——— AUDIO INTELLIGENCE PIPELINE ———
class AudioIntelligencePipeline:
    def __init__(self, hf_token: str = HF_TOKEN):
        print("[Karma Compass] Initializing M4 Pro Optimized Pipeline...")
        # 1. Pyannote Diarization (CPU-only: MPS support unstable)
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=hf_token
            )
        except Exception as e:
            print(f"[Karma Compass] Warning: Pyannote init failed: {e}")
            self.diarization_pipeline = None
        # 2. Whisper Transcription (Running on MPS/GPU)
        self.whisper_model = whisper.load_model("base", device="mps")
        # 3. Pattern Engine
        self.prime_engine = PrimeComplexityEngine()

    def process_session(self, audio_file: str, ollama_model: str = OLLAMA_MODEL) -> dict:
        print(f"[Karma Compass] Analyzing Audio: {audio_file}")
        # Step 1: Diarization (Who spoke when)
        if self.diarization_pipeline:
            diarization = self.diarization_pipeline(audio_file)
        else:
            # Fallback: assume 2 speakers split by time
            diarization = type('obj', (object,), {
                'itertracks': lambda self: iter([
                    (type('obj', (object,), {'start': 0, 'end': 0.5}), None, "S1"),
                    (type('obj', (object,), {'start': 0.5, 'end': 1.0}), None, "S2")
                ])
            })()

        # Step 2: Transcription (What was said)
        result = self.whisper_model.transcribe(audio_file)
        segments = result['segments']

        # Step 3: Map Transcription to Speakers (time-aligned)
        speaker_data = {"S1": [], "S2": []}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_label = "S1" if "00" in str(speaker) else "S2"
            # Find text segments that fall within this speaker's time window
            segment_text = ""
            for seg in segments:
                if seg['start'] >= turn.start and seg['end'] <= turn.end:
                    segment_text += " " + seg['text']
            if segment_text.strip():
                speaker_data[speaker_label].append(segment_text.strip())

        # Step 4: Prime Analysis for both speakers
        final_report = {}
        for speaker, texts in speaker_data.items():
            full_text = " ".join(texts)
            if not full_text: 
                final_report[speaker] = {"error": "No speech detected"}
                continue
            density, tests = self.prime_engine.calculate_structural_complexity(full_text)
            p_val = self.prime_engine.calculate_p_value(density, tests)
            final_report[speaker] = {
                "Speech": full_text,
                "Prime_Density": round(density, 5),
                "P_Value_Randomness": f"{p_val:.10f}",
                "Conscious_Effort_Detected": p_val < 0.05,
                "total_tests": tests,
                "prime_hits": int(density * tests)
            }

        # Step 5: Semantic Reciprocity via Ollama Embedding (if 2 speakers)
        if len(final_report) == 2 and "error" not in final_report.get("S1", {}) and "error" not in final_report.get("S2", {}):
            try:
                s1_text = final_report["S1"]["Speech"]
                s2_text = final_report["S2"]["Speech"]
                print("[Karma Compass] Calculating Semantic Alignment via Gemma...")
                emb_s1 = ollama.embed(model=ollama_model, input=s1_text)['embeddings'][0]
                emb_s2 = ollama.embed(model=ollama_model, input=s2_text)['embeddings'][0]
                similarity = np.dot(emb_s1, emb_s2) / (
                    np.linalg.norm(emb_s1) * np.linalg.norm(emb_s2)
                )
                final_report["GLOBAL_ALIGNMENT"] = round(similarity, 4)
                final_report["COGNITIVE_RECIPROCITY"] = self._reciprocity_score(final_report["S1"], final_report["S2"])
            except Exception as e:
                print(f"[Karma Compass] Embedding error: {e}")
                final_report["GLOBAL_ALIGNMENT"] = None

        return final_report

    def _reciprocity_score(self, s1: dict, s2: dict) -> float:
        """Simple reciprocity: product of conscious effort flags + embedding similarity."""
        effort1 = s1.get("Conscious_Effort_Detected", False)
        effort2 = s2.get("Conscious_Effort_Detected", False)
        if effort1 and effort2:
            return 0.7 + (s2.get("GLOBAL_ALIGNMENT", 0.5) or 0.5) * 0.3
        elif effort1 or effort2:
            return 0.4 + (s2.get("GLOBAL_ALIGNMENT", 0.5) or 0.5) * 0.2
        else:
            return 0.1 + (s2.get("GLOBAL_ALIGNMENT", 0.5) or 0.5) * 0.1

# ——— LONGITUDINAL TRACKING (SQLite) ———
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user1_id TEXT NOT NULL,
            user2_id TEXT NOT NULL,
            reciprocity REAL NOT NULL,
            ethical_score REAL,
            karma_estimate REAL NOT NULL,
            s1_prime_density REAL,
            s2_prime_density REAL,
            s1_p_val REAL,
            s2_p_val REAL,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(
    user1_id: str,
    user2_id: str,
    reciprocity: float,
    ethical_score: float,
    karma: float,
    s1_density: float,
    s2_density: float,
    s1_p_val: float,
    s2_p_val: float,
    notes: str = ""
):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO interactions 
            (timestamp, user1_id, user2_id, reciprocity, ethical_score, karma_estimate, 
             s1_prime_density, s2_prime_density, s1_p_val, s2_p_val, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user1_id,
            user2_id,
            reciprocity,
            ethical_score,
            karma,
            s1_density,
            s2_density,
            s1_p_val,
            s2_p_val,
            notes
        ))
        conn.commit()
    finally:
        conn.close()

# ——— VOICE CLONING DETECTION (3-feature heuristic) ———
class VoiceCloningDetector:
    def __init__(self):
        self.threshold = 0.65  # >65% spoof probability → block

    def detect_spoof(self, audio_path: str) -> tuple:
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, duration=5)  # 5s sample
            if len(y) < sr:
                return False, 0.0, "Audio too short"

            # 1. Zero-crossing rate (voice = spiky, synth = flat)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_var = np.var(zcr)

            # 2. Spectral flatness (synth = flat = low variance)
            spec_flat = librosa.feature.spectral_flatness(y=y)[0]
            spec_flat_var = np.var(spec_flat)

            # 3. Spectral centroid variance (real voice = richer harmonics = higher variance)
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_var = np.var(spec_centroid)

            # Combine into spoof score
            spoof_score = (
                0.4 * (spec_flat_var < 0.001) +
                0.3 * (zcr_var < 0.0005) +
                0.3 * (centroid_var < 2000)
            )

            is_spoofed = spoof_score > self.threshold
            details = f"spec_flat_var: {spec_flat_var:.6f}, zcr_var: {zcr_var:.6f}, centroid_var: {centroid_var:.0f}"
            return is_spoofed, spoof_score, details
        except Exception as e:
            return False, 0.0, f"Error: {e}"

# ——— MAIN APP (Gradio UI) ———
def analyze_audio(audio_path, s1_name, s2_name, notes):
    if not audio_path:
        raise gr.Error("No audio file provided.")
    # Voice cloning check
    spoof_detector = VoiceCloningDetector()
    is_spoofed, score, details = spoof_detector.detect_spoof(audio_path)
    if is_spoofed:
        raise gr.Error(f"⚠️ Voice cloning detected (confidence: {score:.1%}). Refusing to analyze.")

    # Initialize pipeline
    pipeline = AudioIntelligencePipeline(hf_token=HF_TOKEN)

    # Process session
    report = pipeline.process_session(audio_path, ollama_model=OLLAMA_MODEL)

    # Extract key scores
    reciprocity = report.get("COGNITIVE_RECIPROCITY", 0.0)
    s1_density = report.get("S1", {}).get("Prime_Density", 0.0)
    s2_density = report.get("S2", {}).get("Prime_Density", 0.0)
    s1_p_val = report.get("S1", {}).get("P_Value_Randomness", "1.0")
    s2_p_val = report.get("S2", {}).get("P_Value_Randomness", "1.0")
    
    # Ethical scoring via Ollama (low-temperature summary)
    s1_text = report.get("S1", {}).get("Speech", "")
    s2_text = report.get("S2", {}).get("Speech", "")
    try:
        ethical_prompt = f"Rate ethical alignment 0-10 based on: S1='{s1_text}' S2='{s2_text}'. Respond with: <score: X>"
        ethical_resp = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": ethical_prompt}])
        import re
        m = re.search(r'<score:\s*(\d+)>', ethical_resp['message']['content'])
        ethical_score = float(m.group(1)) / 10.0 if m else 0.5
    except:
        ethical_score = 0.5

    # Composite karma
    karma = (reciprocity * 0.4 + ethical_score * 0.6) * (1.0 / (1.0 + abs(report.get("S1", {}).get("P_Value_Randomness", 0.5))))

    # Log to DB
    try:
        log_interaction(
            s1_name, s2_name,
            reciprocity, ethical_score, karma,
            s1_density, s2_density, s1_p_val, s2_p_val,
            notes
        )
    except Exception as e:
        print(f"[Karma Compass] DB error: {e}")

    # Build final report
    output_report = {
        "Cognitive Reciprocity": f"{reciprocity:.3f}",
        "Ethical Score": f"{ethical_score:.3f}",
        "Karma Estimate": f"{karma:.3f}",
        "Speaker 1": {
            "Prime Density": f"{s1_density:.5f}",
            "P-Value": s1_p_val,
            "Conscious Effort": report["S1"].get("Conscious_Effort_Detected", False),
            "Total Tests": report["S1"].get("total_tests", 0),
            "Prime Hits": report["S1"].get("prime_hits", 0)
        },
        "Speaker 2": {
            "Prime Density": f"{s2_density:.5f}",
            "P-Value": s2_p_val,
            "Conscious Effort": report["S2"].get("Conscious_Effort_Detected", False),
            "Total Tests": report["S2"].get("total_tests", 0),
            "Prime Hits": report["S2"].get("prime_hits", 0)
        },
        "Global Alignment": report.get("GLOBAL_ALIGNMENT"),
        "Full Text": f"S1: {s1_text}\n\nS2: {s2_text}"
    }

    return json.dumps(output_report, indent=2)

# ——— GRADIO UI ———
with gr.Blocks(title="Karma Compass") as demo:
    gr.Markdown("## 🌱 Karma Compass: Cognitive & Ethical Reciprocity Engine")
    gr.Markdown("Upload a call recording (MP3/WAV) → Analyze prime-pattern reciprocity + semantic alignment.")

    with gr.Row():
        audio_in = gr.Audio(sources=["upload"], type="filepath", label="Call Recording (MP3/WAV)")
        with gr.Column():
            s1_name = gr.Textbox(label="Speaker 1 ID", value="user1")
            s2_name = gr.Textbox(label="Speaker 2 ID", value="user2")
            notes = gr.Textbox(label="Context/Notes", placeholder="e.g., first meeting, brainstorm")
            btn_analyze = gr.Button("⚡ Analyze Reciprocity", variant="primary")

    output = gr.Textbox(label="Analysis Report", lines=20)

    # ——— KARMA TRAJECTORY ———
    with gr.Row():
        tracker_name = gr.Textbox(label="User ID for Trajectory")
        btn_refresh = gr.Button("📊 Refresh Trajectory")

    trajectory_plot = gr.Plot(label="Karma Over Time")
    traj_table = gr.DataFrame(label="Recent Interactions")

    def update_trajectory(name):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT timestamp, karma_estimate FROM interactions 
            WHERE user1_id = ? OR user2_id = ?
            ORDER BY timestamp
        """, (name, name))
        rows = c.fetchall()
        conn.close()
        if not rows:
            return None, []

        # Parse times + karma
        times = [r[0] for r in rows]
        karmas = [r[1] for r in rows]
        try:
            from datetime import datetime as dt
            times = [dt.fromisoformat(t) for t in times]
        except:
            times = list(range(len(karmas)))

        # Plot
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(times, karmas, marker='o', linestyle='-', color='#2c7bb6')
        ax.set_xlabel("Time")
        ax.set_ylabel("Karma Estimate")
        ax.set_title(f"Karma Trajectory for {name}")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        fig.tight_layout()

        # Table
        table_data = [["Timestamp", "Karma"]]
        for t, k in zip(times, karmas):
            table_data.append([t.isoformat() if hasattr(t, 'isoformat') else str(t), f"{k:.3f}"])

        return fig, table_data

    btn_refresh.click(
        update_trajectory,
        inputs=[tracker_name],
        outputs=[trajectory_plot, traj_table]
    )

    # ——— EVENT LISTENER ———
    btn_analyze.click(
        analyze_audio,
        inputs=[audio_in, s1_name, s2_name, notes],
        outputs=[output]
    )

# ——— LAUNCH ———
init_db()
demo.launch(share=False)

