# English Accent Detector

⚠️ **Warning:** This tool analyzes only the first few seconds of the audio/video to detect the accent. The results may be inaccurate, especially for similar accents (e.g., Indian English vs US English). The SpeechBrain model used is open-source and has limitations. For professional results, we recommend using a commercial speech-to-text service with accent detection.

This is a simple Streamlit application to download audio from a public video URL and classify the speaker's English accent using a pre-trained Hugging Face model.

## Prerequisites

1.  **Python 3.7+**
2.  **ffmpeg**: This command-line tool is required by `yt-dlp` for audio extraction. Follow the installation instructions for your operating system: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
3.  **Git** (optional, for cloning this repository)

## Setup

1.  Clone the repository (if applicable) or save the files `app.py` and `requirements.txt`.
2.  Navigate to the project directory in your terminal.
3.  Create a virtual environment (recommended):
    ```bash
    python -m venv .venv
    ```
4.  Activate the virtual environment:
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
5.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  Make sure your virtual environment is activated.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  A new tab should open in your web browser with the application interface.

## Usage

1.  Paste the public URL of a video containing spoken English into the text box.
2.  Click the "Analyze Accent" button.
3.  The application will download the audio, process it, and display the predicted English accent and a confidence score.

## Notes

*   This tool relies on the `furkanakkurt/english-accent-classification` model from Hugging Face, which supports a limited number of English accents (e.g., en-US, en-GB, en-AU, en-IN).
*   The quality of the accent detection depends heavily on the quality of the audio and the specific model used.
*   Ensure the video URL is publicly accessible and does not require authentication.
*   The first time you run the app, the Hugging Face model will be downloaded, which may take some time depending on your internet connection. 