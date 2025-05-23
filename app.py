import streamlit as st
import subprocess
import os
import time # Import time for a simple unique filename
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, pipeline
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

import torch # Often needed by transformers

# --- Accent Classification Setup ---
# Load the pre-trained model
model_id = "dima806/speech-accent-classification"
# Based on previous errors and the model type, sticking with explicit Wav2Vec2 classes might be safer
# feat_ext = AutoFeatureExtractor.from_pretrained(model_id)
# model    = AutoModelForAudioClassification.from_pretrained(model_id)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_loaded = False # Initialize model_loaded to False
try:
    # Attempt to load the model components
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model    = Wav2Vec2ForSequenceClassification.from_pretrained(model_id, torch_dtype=dtype)
    # Create the audio classification pipeline
    clf = pipeline(
        "audio-classification",
        model=model,
        feature_extractor=feat_ext,
        return_all_scores=True,
        torch_dtype=dtype, # Pass dtype to pipeline
        device=0 if torch.cuda.is_available() else -1 # Use GPU if available, otherwise CPU (-1)
    )
    st.success("Accent classification model loaded successfully.")
    model_loaded = True # Set to True only if all components load

except Exception as e:
    st.error(f"Failed to load accent classification model: {e}")
    st.warning("Please ensure you have the 'transformers', 'soundfile', and 'torch' libraries installed and that the model ID is correct.")
    model_loaded = False # Ensure it's False on failure


st.title("English Accent Detector")

st.write("Enter a public video URL (e.g., Loom or direct MP4 link) to analyze the speaker's English accent.")

video_url = st.text_input("Video URL")

if st.button("Analyze Accent"):
    if not model_loaded:
        st.error("Cannot proceed because the accent classification model failed to load.")
    elif not video_url:
         st.warning("Please enter a video URL.")
    else:
        st.info("Processing video and analyzing accent. This may take a moment...")

        # --- Audio Extraction ---
        # Generate a unique filename based on timestamp
        timestamp = int(time.time())
        output_audio_path = f"candidate_audio_{timestamp}.wav"

        # Use yt-dlp to download and extract audio as WAV
        # Requires yt-dlp and ffmpeg to be installed and in PATH
        try:
            st.info(f"Downloading audio to {output_audio_path}...")
            # Command to download and convert to WAV using yt-dlp
            # -x: extract audio
            # --audio-format wav: specify format
            # -o: output filename template
            # Use --no-check-certificate for potentially problematic URLs, remove if not needed
            command = [
                "yt-dlp",
                "-x",
                "--audio-format", "wav",
                "--no-check-certificate",
                "--output", output_audio_path,
                video_url
            ]
            # Added timeout for subprocess to prevent hanging
            process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300) # 5 minutes timeout
            st.text("yt-dlp output:")
            st.text(process.stdout)
            st.text("yt-dlp errors:")
            st.text(process.stderr)
            st.success(f"Audio extracted successfully to {output_audio_path}")

            # --- Accent Classification ---
            st.info("Analyzing accent...")
            try:
                # Expected output structure for this model seems to be: [{"label": "...", "score": ...}, {"label": "...", "score": ...}, ...]
                pipeline_results = clf(output_audio_path)

                st.write("Debugging: Raw pipeline output:")
                st.write(pipeline_results) # Print the raw output here

                # Directly process the pipeline_results list
                if pipeline_results and isinstance(pipeline_results, list):

                    st.write("Debugging: Raw pipeline_results structure and types:")
                    for i, item in enumerate(pipeline_results):
                        st.write(f"  Item {i}: Type = {type(item)}, Content = {item}")

                    # Filter out any non-dictionary or incomplete items more strictly
                    valid_results = [res for res in pipeline_results if isinstance(res, dict) and "label" in res and isinstance(res["label"], str) and "score" in res and isinstance(res["score"], (int, float))]

                    st.write(f"Debugging: Found {len(valid_results)} valid result dictionaries after filtering.")


                    if valid_results:
                        # Choose the most probable accent from valid results
                        # Ensure the max function operates only on valid dictionaries
                        best = max(valid_results, key=lambda x: x["score"])
                        label, conf = best["label"], best["score"]

                        st.subheader("Analysis Results")
                        # The model seems to output 'Foreign' or 'Native'. We need to map 'Native' to an English accent type if possible, or just report 'Native English Speaker'.
                        # For now, let's just report the label as is from the model, as per the model card's likely labels.
                        # If the task requires specific English accent types (British, American, etc.), this model might not be granular enough.
                        # Given the model outputs 'Foreign' and 'Native', 'Native' likely implies a native English speaker without differentiating specific regional accents.
                        display_label = "Native English Speaker" if label == "Native" else label # Improve display for 'Native'

                        st.metric("Predicted Accent", display_label, f"{conf*100:.1f}% Confidence")


                        # Optional: Display all valid scores
                        st.write("All scores:")
                        # Sort the valid results before displaying
                        for res in sorted(valid_results, key=lambda x: x["score"], reverse=True):
                            # Adjust display for 'Native' in the full list as well
                            display_item_label = "Native English Speaker" if res['label'] == "Native" else res['label']
                            st.text(f"- {display_item_label}: {res['score']*100:.1f}%")
                    else:
                        st.warning("No valid accent classification scores were obtained from the pipeline output after filtering.")
                        st.write("Raw pipeline output was:", pipeline_results) # Already printed above

                else:
                    st.error("Unexpected format from accent classification pipeline output.")
                    st.write("Raw pipeline output was:", pipeline_results) # Already printed above


            except Exception as e:
                st.error(f"Error during accent classification: {e}")
                st.write(f"Details: {e}")
                st.write("Ensure the video contains clear English speech and that the model is appropriate or try a different video.")


            finally:
                # Clean up the downloaded audio file
                if os.path.exists(output_audio_path):
                    os.remove(output_audio_path)
                    st.info(f"Cleaned up temporary audio file: {output_audio_path}")

        except FileNotFoundError:
            st.error("Error: 'yt-dlp' or 'ffmpeg' command not found.")
            st.warning("Please ensure you have yt-dlp and ffmpeg installed and accessible in your system's PATH.")
            st.markdown("Instructions for installing ffmpeg: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)")
            st.markdown("Instructions for installing yt-dlp: [https://github.com/yt-dlp/yt-dlp#installation](https://github.com/yt-dlp/yt-dlp#installation)")
        except subprocess.TimeoutExpired:
             st.error(f"Video download/audio extraction timed out after 5 minutes.")
             st.warning("The video might be too long or there might be network issues.")
             if os.path.exists(output_audio_path):
                 os.remove(output_audio_path)
                 st.info(f"Cleaned up temporary audio file: {output_audio_path}")
        except subprocess.CalledProcessError as e:
            st.error(f"Error during video download/audio extraction: {e}")
            st.text("Command output:")
            st.text(e.stdout)
            st.text("Command errors:")
            st.text(e.stderr)
        except Exception as e:
            st.error(f"An unexpected error occurred during video processing: {e}") 