import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import subprocess
import os
import time
import torch
import soundfile as sf
import numpy as np
import asyncio
import nest_asyncio
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set page config for better UI
st.set_page_config(
    page_title="English Accent Detector",
    # page_icon="🎙️", # Removed to avoid MemoryError
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .accent-result {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        background-color: black;
    }
    .confidence-bar {
        height: 20px;
        background-color: #4CAF50;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Accent Classification Setup ---
# Updated model_id to the SpeechBrain model
# Updated model_id to the SpeechBrain model
model_id = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"

# Removed dtype and transformers model loading
# model_loaded = False # Will be managed by try/except

# Removed chunking function as SpeechBrain's classify_file might handle it
# def process_audio_in_chunks(audio_path, chunk_duration=30):
#     # ... (function removed) ...

classifier = None # Initialize classifier outside the try block
model_loaded = False # Initialize model_loaded flag

# Initialize event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Set torchaudio backend
# torchaudio.set_audio_backend("soundfile")

with st.spinner(f"Loading accent classification model: {model_id}..."):
    try:
        classifier = foreign_class(
            source=model_id,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier"
        )
        st.success(f"✅ Accent classification model `{model_id}` loaded successfully.")
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load accent classification model `{model_id}`: {e}")
        st.warning("Please ensure you have the required libraries (speechbrain, torchaudio) and that the model ID is correct.")
        model_loaded = False

# Main UI
st.title("🎙️ English Accent Detector")
st.info("""
⚠️ **Warning:** This tool analyzes only the first few seconds of the audio/video to detect the accent. The results may be inaccurate, especially for similar accents (e.g., Indian English vs US English). The SpeechBrain model used is open-source and has limitations. For professional results, we recommend using a commercial speech-to-text service with accent detection.
""")
st.markdown("""
This tool analyzes English accents from video URLs. It can identify various English accents including:
- British English (en-GB)
- American English (en-US)
- Australian English (en-AU)
- Indian English (en-IN)
- And more...

Simply paste a video URL (e.g., YouTube, Loom, or direct MP4 link) and the tool will analyze the speaker's accent.
""")

# Sample video URL for testing
st.markdown("""
**Sample video for testing:**
```
https://www.youtube.com/watch?v=dQw4w9WgXcQ
```
""")

video_url = st.text_input("Enter Video URL", placeholder="https://...")

if st.button("Analyze Accent", type="primary"):
    if not model_loaded or classifier is None:
        st.error("Cannot proceed because the accent classification model failed to load.")
    elif not video_url:
        st.warning("Please enter a video URL.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Ensure output_audio_path is defined before the try block for finally clause
        output_audio_path = None

        try:
            # --- Audio Extraction ---
            timestamp = int(time.time())
            output_audio_path = f"candidate_audio_{timestamp}.wav"

            status_text.text("Downloading and extracting audio...")
            progress_bar.progress(20)

            command = [
                "yt-dlp",
                "-x",
                "--audio-format", "wav",
                "--no-check-certificate",
                "--output", output_audio_path,
                "--postprocessor-args", "ffmpeg:-t 30",
                video_url
            ]

            # Added timeout for subprocess to prevent hanging
            process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
            progress_bar.progress(50)

            if not os.path.exists(output_audio_path):
                raise FileNotFoundError("Audio file was not created successfully")

            status_text.text("Analyzing accent...")
            progress_bar.progress(70)

            # --- Accent Classification using EncoderClassifier ---
            out_prob, score, index, text_lab = classifier.classify_file(output_audio_path)
            # Convert probabilities to list
            out_prob = out_prob.squeeze().tolist() if hasattr(out_prob, 'squeeze') else list(out_prob)
            progress_bar.progress(90)

            # Get the list of all possible labels from the model's hparams
            all_labels = classifier.hparams.label_encoder.ind2lab

            # Combine labels and scores into a list of dictionaries for consistent processing
            valid_results = []
            if len(out_prob) == len(all_labels):
                valid_results = [{'label': all_labels[i], 'score': out_prob[i]} for i in range(len(all_labels))]

            if valid_results:
                # The best result is already given by text_lab and score, but we re-find it for consistency
                # and to ensure the display logic uses the processed valid_results list.
                best = max(valid_results, key=lambda x: x["score"])
                predicted_label, predicted_conf = best["label"], best["score"]

                progress_bar.progress(100)
                status_text.text("Analysis complete!")

                st.subheader("Analysis Results")
                
                # Display the main result in a prominent way
                st.markdown(f"""
                <div class="accent-result">
                    <h3>Predicted Accent: {predicted_label}</h3>
                    <p>Confidence: {predicted_conf*100:.1f}%</p>
                    <div class="confidence-bar" style="width: {predicted_conf*100}%"></div>
                </div>
                """, unsafe_allow_html=True)

                # Display all results in a sorted table
                st.subheader("All Accent Probabilities")
                sorted_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)
                
                for res in sorted_results:
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span>{res['label']}</span>
                            <span>{res['score']*100:.1f}%</span>
                        </div>
                        <div class="confidence-bar" style="width: {res['score']*100}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No valid accent classification scores were obtained.")
                st.write("Raw pipeline output (out_prob):", out_prob)
                st.write("Expected labels (all_labels):", all_labels)

        except FileNotFoundError:
            st.error("Error: 'yt-dlp' or 'ffmpeg' command not found.")
            st.warning("Please ensure you have yt-dlp and ffmpeg installed and accessible in your system's PATH.")
            st.markdown("""
            Installation instructions:
            - [FFmpeg](https://ffmpeg.org/download.html)
            - [yt-dlp](https://github.com/yt-dlp/yt-dlp#installation)
            """)
        except subprocess.TimeoutExpired:
            st.error("Video download/audio extraction timed out after 5 minutes.")
            st.warning("The video might be too long or there might be network issues.")
        except subprocess.CalledProcessError as e:
            st.error(f"Error during video download/audio extraction: {e}")
            st.text("Command output:")
            st.text(e.stdout)
            st.text("Command errors:")
            st.text(e.stderr)
        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            st.warning("Ensure the video contains clear English speech.")
        finally:
            # Clean up the downloaded audio file if it was created
            if output_audio_path and os.path.exists(output_audio_path):
                os.remove(output_audio_path)
                st.info(f"Cleaned up temporary audio file: {output_audio_path}")
            # Always clear progress bar and status text
            progress_bar.empty()
            status_text.empty() 
