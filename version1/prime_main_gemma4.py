import torch
import whisper
import ollama
import numpy as np
import re
import itertools
from pyannote.audio import Pipeline
from scipy.stats import poisson
from scipy.special import factorial

# Use Apple Silicon GPU (MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {device}")

class PrimeComplexityEngine:
    def __init__(self):
        # Pre-compute primes for a-z
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 7_3, 79, 83, 89, 97, 101]
        self.char_map = {chr(i + 97): self.primes[i] for i in range(26)}
        self.baseline_lambda = 0.08  # Expected prime-sum density in random speech

    def is_prime(self, n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def get_word_prime_value(self, word):
        clean_word = re.sub(r'[^a-z]', '', word.lower())
        return sum(self.char_map.get(c, 0) for c in clean_word)

    def calculate_structural_complexity(self, text):
        """
        Implements the 'Prefix-Sum' and 'Combinatorial Window' logic.
        Checks: W1, W1+W2, W1+W2+W3... and truncated power-sets.
        """
        words = re.findall(r'\b[a-int-a-z]+\b', text.lower())
        if not words: return 0.0, 0.0
        
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
        # This looks for hidden patterns in the recent word history
        window_size = min(len(word_values), 10)
        recent_words = word_values[-window_size:]
        
        # Check subsets of the recent window
        for r in range(2, window_size):
            for subset in itertools.combinations(recent_words, r):
                total_tests += 1
                if self.is_prime(sum(subset)):
                    prime_hits += 1

        density = prime_hits / total_tests if total_tests > 0 else 0
        return density, total_tests

    def calculate_p_value(self, observed_density, total_tests):
        """
        Calculates the probability that the observed prime density 
        is a result of random chance (Poisson-based significance).
        """
        k = int(observed_density * total_tests)
        if k == 0: return 1.0
        # Probability of seeing k or more successes in a Poisson distribution
        # based on the natural speech lambda.
        try:
            # Using the Poisson survival function (1 - CDF)
            p_val = 1 - poisson.cdf(k, self.baseline_lambda * total_tests)
            return p_val
        except:
            return 1.0

class AudioIntelligencePipeline:
    def __init__(self, hf_token, ollama_model="gemma:2b"):
        print("Initializing M4 Pro Optimized Pipeline...")
        
        # 1. Pyannote Diarization (Running on CPU as MPS support for Pyannote is unstable)
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        )
        
        # 2. Whisper Transcription (Running on MPS/GPU)
        self.whisper_model = whisper.load_model("base", device="mps")
        
        # 3. Pattern & Semantic Engines
        self.prime_engine = PrimeComplexityEngine()
        self.ollama_model = ollama_model

    def process_session(self, audio_file):
        print(f"Analyzing Audio: {audio_file}")
        
        # Step 1: Diarization (Who spoke when)
        diarization = self.diarization_pipeline(audio_file)
        
        # Step 2: Transcription (What was said)
        result = self.whisper_model.transcribe(audio_file)
        segments = result['segments']
        
        # Step 3: Map Transcription to Speakers
        speaker_data = {"S1": [], "S2": []}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_label = "S1" if speaker == "spk00" else "S2"
            
            # Find text segments that fall within this speaker's time window
            segment_text = ""
            for seg in segments:
                if seg['start'] >= turn.start and seg['end'] <= turn.end:
                    segment_text += " " + seg['text']
            
            if segment_text.strip():
                speaker_data[speaker_label].append(segment_text.strip())

        # Step 4: Analysis for both speakers
        final_report = {}
        
        for speaker, texts in speaker_data.items():
            full_text = " ".join(texts)
            if not full_text: continue
            
            density, tests = self.prime_engine.calculate_structural_complexity(full_text)
            p_val = self.prime_engine.calculate_p_value(density, tests)
            
            final_report[speaker] = {
                "Speech": full_text,
                "Prime_Density": round(density, 5),
                "P_Value_Randomness": f"{p_val:.10f}",
                "Conscious_Effort_Detected": p_val < 0.05
            }

        # Step 5: Semantic Reciprocity (Using Speaker 1 as the anchor)
        if len(final_report) == 2:
            s1_text = final_report["S1"]["Speech"]
            s2_text = final_report["S2"]["Speech"]

            
            print("Calculating Semantic Alignment via Gemma...")
            emb_s1 = ollama.embed(model=self.ollama_model, input=s1_text)['embeddings'][0]
            emb_s2 = ollama.embed(model=self.ollama_model, input=s2_text)['embeddings'][0]
            
            similarity = np.dot(emb_s1, emb_s2) / (
                np.linalg.norm(emb_s1) * np.linalg.norm(emb_s2)
            )
            final_report["GLOBAL_ALIGNMENT"] = round(similarity, 4)

        return final_report

# --- MAIN EXECUTION ---
if __if_name_ == "__main__":
    # Replace with your HuggingFace Token
    HF_TOKEN = "your_hf_token_here"
    
    pipeline = AudioIntelligencePipeline(hf_token=HF_TOKEN)
    
    # Replace with your actual recorded audio file
    try:
        report = pipeline.process_session("input_audio.wav")
        
        print("\n" + "="*50)
        print("        KARMIC COGNITION ANALYSIS REPORT")
        print("="*50)
        for speaker, data in report.items():
            print(f"\n[{speaker}]")
            for k, v in data.items():
                print(        f"  {k:25}: {v}")
        print("="*50)
        
    except Exception as e:
        print(f"Pipeline Failure: {e}")