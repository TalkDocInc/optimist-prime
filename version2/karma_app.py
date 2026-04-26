# karma_app.py
import gradio as gr
import numpy as np
import os
import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
import time

# Local imports (ensure `core/` is in PYTHONPATH)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

from audio_processor import AudioProcessor
from prime_scorer import all_prime_substring_sums, word_to_prime_sum, is_prime
from reciprocity_engine import (
    cognitive_reciprocity_score,
    load_or_build_norm_stats,
    speaker_prime_profile,
    reciprocal_pattern_score
)

# ——— CONFIG ———
MODEL_NAME = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434"

# ——— DB MANAGEMENT ———
DB_PATH = "karma_trajectory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            speaker1_name TEXT,
            speaker2_name TEXT,
            reciprocity REAL,
            ethical_score REAL,
            karma_estimate REAL,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(s1_name: str, s2_name: str, reciprocity: float, ethical_score: float, karma: float, notes: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO interactions (timestamp, speaker1_name, speaker2_name, reciprocity, ethical_score, karma_estimate, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), s1_name, s2_name, reciprocity, ethical_score, karma, notes)
    )
    conn.commit()
    conn.close()

def get_trajectory(speaker: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, karma_estimate FROM interactions 
        WHERE speaker1_name = ? OR speaker2_name = ?
        ORDER BY timestamp
    """, (speaker, speaker))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return [], []
    times = [r[0] for r in rows]
    karmas = [r[1] for r in rows]
    # Convert ISO strings to datetime for plotting
    from datetime import datetime as dt
    try:
        times = [dt.fromisoformat(t) for t in times]
    except:
        times = list(range(len(karmas)))
    return times, karmas

# ——— ETHICAL ALIGNMENT SCORING (via Gemma) ———
class EthicalScorer:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.prompt_template = """Analyze the ethical alignment between two speakers based on their dialogue. Rate ethical alignment on 0–10, considering: compassion, truthfulness, long-term benefit, fairness, humility, and mutual uplift.

Speaker 1: "{s1}"
Speaker 2: "{s2}"

Provide:
1. A 1-sentence analysis
2. A numeric score (0–10, where 10 = maximum ethical reciprocity)
Respond ONLY with: "<analysis>" followed by "<score: X>" """

    def score(self, s1: str, s2: str) -> (str, float):
        # Use Ollama (fallback to naive heuristic if unavailable)
        try:
            import requests
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a wise, compassionate ethicist."},
                        {"role": "user", "content": self.prompt_template.format(s1=s1, s2=s2)}
                    ],
                    "options": {"temperature": 0.1, "num_ctx": 4096}
                },
                timeout=15
            )
            if resp.status_code == 200:
                result = resp.json()["message"]["content"].strip()
                # Parse score from text (look for "<score: X>" or just a number)
                import re
                m = re.search(r'<score:\s*(\d+(?:\.\d+)?)>', result)
                if m:
                    score = float(m.group(1)) / 10.0  # normalize to 0–1
                else:
                    # Fallback: scan any number 0–10
                    nums = re.findall(r'(\d+(?:\.\d+)?)', result)
                    for n in nums:
                        try:
                            val = float(n)
                            if 0 <= val <= 10:
                                score = val / 10.0
                                break
                        except: pass
                    else:
                        score = 0.5  # neutral default
                # Extract analysis (everything before score tag)
                analysis = result.split("<score:")[0].strip() if "<score:" in result else result
                return analysis, score
            else:
                return "⚠️ Ollama unreachable — using heuristic.", self._heuristic_score(s1, s2)
        except Exception as e:
            return f"⚠️ Ollama error: {e}", self._heuristic_score(s1, s2)

    def _heuristic_score(self, s1: str, s2: str) -> float:
        # Simple ethical proxy: reciprocal keywords + prime-pattern match
        s1_subs = all_prime_substring_sums(s1)
        s2_subs = all_prime_substring_sums(s2)
        recip = reciprocal_pattern_score(s1, s2)
        # Ethical keywords (you can expand this!)
        ethical_terms = {
            "compassion", "fair", "just", "share", "sustain", "uplift", "wise", "honest", "help", "together"
        }
        def count_ethical(text):
            return sum(1 for w in text.lower().split() if w.strip(".,?!") in ethical_terms)
        eth1, eth2 = count_ethical(s1), count_ethical(s2)
        eth_ratio = min(eth1/eth2, eth2/eth1) if eth1 > 0 and eth2 > 0 else 0
        return min(1.0, 0.4 * recip + 0.4 * (eth1 + eth2) / 10 + 0.2 * eth_ratio)

# ——— UI LOGIC ———
audio_proc = AudioProcessor()
ethical_scorer = EthicalScorer()
norm_stats = load_or_build_norm_stats()

def process_speech(s1_audio, s2_audio, s1_name, s2_name, notes):
    if not s1_audio or not s2_audio:
        raise gr.Error("Please provide both audio recordings.")
    
    # Transcribe (batch for speed)
    _, s1_text = audio_proc.process_file(s1_audio)
    _, s2_text = audio_proc.process_file(s2_audio)
    
    # Prime-based cognitive reciprocity
    rec_result = cognitive_reciprocity_score(s1_text, s2_text, norm_stats)
    reciprocity = rec_result["reciprocity_score"]

    # Ethical alignment
    analysis, ethical = ethical_scorer.score(s1_text, s2_text)

    # Composite karma: cognitive + ethical × norm-deviation penalty
    k1_z = abs(rec_result["speaker1_profile"]["z_score"])
    k2_z = abs(rec_result["speaker2_profile"]["z_score"])
    deviation_penalty = 1.0 / (1.0 + k1_z + k2_z)  # higher z = more "effortful", lower penalty
    karma = (reciprocity * 0.4 + ethical * 0.6) * deviation_penalty

    # Log to DB
    log_interaction(s1_name, s2_name, reciprocity, ethical, karma, notes)

    # Generate visualization data
    t1, k1 = get_trajectory(s1_name)
    t2, k2 = get_trajectory(s2_name)
    combined = (t1 + t2, k1 + k2)

    return {
        "cognitive_reciprocity": reciprocity,
        "ethical_insight": analysis,
        "ethical_score": ethical,
        "karma_score": karma,
        "s1_profile": rec_result["speaker1_profile"],
        "s2_profile": rec_result["speaker2_profile"],
        "trajectory_data": combined
    }

# ——— UI LAYOUT ———
with gr.Blocks(title="Karma Compass", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌱 Karma Compass — Find friends who multiply positive impact")
    gr.Markdown("Record conversations → See *cognitive* & *ethical* alignment → Track long-term karma trajectory.")

    with gr.Tabs():
        # 1️⃣ REAL-TIME INTERACTION
        with gr.TabItem("🤝 Record & Analyze"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Speaker 1")
                    s1_rec = gr.Audio(sources=["microphone"], type="filepath", label="Record speaker 1 (up to 30s)")
                    s1_name = gr.Textbox(label="Speaker 1 Name", placeholder="e.g., Alex")
                with gr.Column():
                    gr.Markdown("### Speaker 2")
                    s2_rec = gr.Audio(sources=["microphone"], type="filepath", label="Record speaker 2")
                    s2_name = gr.Textbox(label="Speaker 2 Name", placeholder="e.g., Sam")
            
            notes = gr.Textbox(label="Context/Notes (optional)", placeholder="e.g., first meeting, project brainstorm")

            btn_analyze = gr.Button("⚡ Analyze Alignment", variant="primary")
            
            with gr.Column():
                gr.Markdown("### 📊 Results")
                with gr.Row():
                    rec_score = gr.Number(label="Cognitive Reciprocity", precision=3)
                    eth_score = gr.Number(label="Ethical Score", precision=3)
                    karma_out = gr.Number(label="Karma Estimate", precision=3)
                analysis_out = gr.Textbox(label="Ethical Insight (LLM)", lines=4)
                
                gr.Markdown("#### Speaker Profiles")
                with gr.Row():
                    s1_prof = gr.JSON(label="Speaker 1 Prime Profile")
                    s2_prof = gr.JSON(label="Speaker 2 Prime Profile")

        # 2️⃣ LONG-TERM KARMA TRAJECTORY
        with gr.TabItem("📈 Karma Tracker"):
            gr.Markdown("Track your *mutual karma* across interactions. Higher = more synergistic, aligned impact.")
            tracker_name = gr.Textbox(label="Enter Speaker Name to View", placeholder="e.g., Alex")
            btn_refresh = gr.Button("🔄 Refresh Trajectory", variant="secondary")
            
            plot = gr.Plot(label="Karma Over Time")
            traj_table = gr.DataFrame(label="Recent Interactions (SQL)", max_rows=10)
            
            gr.Markdown("*Data stored locally in `karma_trajectory.db`. No external servers.*")

        # 3️⃣ PROFILE UPLOAD & BASELINE
        with gr.TabItem("👤 Speaker Baseline"):
            gr.Markdown("Upload a short clip of a speaker to build their *prime-profile* and compare against norm.")
            baseline_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload audio (≤60s)")
            profile_name = gr.Textbox(label="Name for Profile", placeholder="e.g., Maya (baseline)")
            btn_profile = gr.Button("🔍 Build Baseline Profile", variant="secondary")
            profile_out = gr.JSON(label="Baseline Prime Profile")

    # ——— EVENT LISTENERS ———
    def analyze_click(s1, s2, s1n, s2n, notes):
        try:
            res = process_speech(s1, s2, s1n, s2n, notes)
            return (
                res["cognitive_reciprocity"],
                res["ethical_score"],
                res["karma_score"],
                res["ethical_insight"],
                res["s1_profile"],
                res["s2_profile"],
                # For plot
                render_trajectory(res["trajectory_data"])
            )
        except Exception as e:
            return [f"⚠️ Error: {e}"] + [None]*6

    def refresh_trajectory(name):
        t, k = get_trajectory(name)
        return render_trajectory((t, k))

    def build_profile(audio, name):
        if not audio or not name:
            return {"error": "Please provide audio and name."}
        try:
            _, text = audio_proc.process_file(audio)
            prof = speaker_prime_profile(text, norm_stats)
            # Save to file
            os.makedirs("profiles", exist_ok=True)
            with open(f"profiles/{name}.json", "w") as f:
                json.dump(prof, f, indent=2, default=str)
            return prof
        except Exception as e:
            return {"error": str(e)}

    btn_analyze.click(
        fn=analyze_click,
        inputs=[s1_rec, s2_rec, s1_name, s2_name, notes],
        outputs=[rec_score, eth_score, karma_out, analysis_out, s1_prof, s2_prof, plot]
    )
    btn_refresh.click(
        fn=refresh_trajectory,
        inputs=[tracker_name],
        outputs=[plot]
    )
    btn_profile.click(
        fn=build_profile,
        inputs=[baseline_audio, profile_name],
        outputs=[profile_out]
    )

    # ——— CUSTOM VISUALIZATION (MPS-OPTIMIZED) ———
    def render_trajectory(data):
        t, k = data
        import matplotlib.pyplot as plt
        if len(k) == 0:
            return None
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
        # Convert times to numeric indices if not datetime
        if not isinstance(t[0], datetime):
            x = list(range(len(k)))
        else:
            x = t
            plt.xticks(rotation=45)
        ax.plot(x, k, marker='o', linestyle='-', color='#2c7bb6')
        ax.set_xlabel("Time")
        ax.set_ylabel("Karma Estimate")
        ax.set_title("Karma Trajectory")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

# ——— LAUNCH ———
init_db()
demo.launch(share=False)  # Set share=True for public link (e.g., Gradio Spaces)