# DubFlow

**DubFlow** is an AI dubbing application that automatically translates and dubs videos into 11 Indian languages. It features a robust backend pipeline leveraging Gemini 2.5 pro ASR, Gemini 2.5 pro MT and Google Cloud TTS and modern, responsive "Warm SaaS" UI.

![DubFlow Dashboard](image%20copy.png)

## Features

-   **Multi-Language Support**: Dub content into Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Assamese, and Odia.
-   **End-to-End Pipeline**: Automated ASR (Transcription), MT (Translation), and TTS (Speech Synthesis).
-   **Interactive Dashboard**:
    -   **Video Library**: View and manage all your dubbed videos.
    -   **Smart Player**: Watch videos with synchronized, auto-scrolling transcripts (Source vs Target).
    -   **Real-time Progress**: Visualizer for the AI pipeline steps.
-   **Persistence**: Automatically saves task history and restores it on server restart.
-   **Auto-Cleanup**: Automatically syncs the dashboard with the file system, removing deleted videos.

## Technology Stack

### Frontend
-   **Framework**: [React 18](https://react.dev/)
-   **Build Tool**: [Vite](https://vitejs.dev/)
-   **Styling**: [Tailwind CSS](https://tailwindcss.com/) + Custom CSS for glassmorphism and animations.
-   **Icons**: [Lucide React](https://lucide.dev/)
-   **HTTP Client**: [Axios](https://axios-http.com/)

### Backend
-   **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
-   **Server**: [Uvicorn](https://www.uvicorn.org/)
-   **Audio Processing**: `ffmpeg`, `pydub`
-   **AI Models**:
    -   **ASR & MT**: Google Gemini 2.5 Pro (via `google.generativeai`)
    -   **TTS**: Google Cloud Text-to-Speech

## AI Pipeline Architecture

The core of DubFlow is an advanced orchestration of **Gemini 2.5 Pro** and **Google Cloud TTS**, handled by `google_pipeline2.py`. Here is the step-by-step workflow:

### 1. Audio Preprocessing
-   **Extraction**: Uses `ffmpeg` to extract audio from the uploaded video.
-   **Chunking**: Splits the audio into **5-minute chunks** to optimize processing for the ASR model.

### 2. Advanced ASR (Automatic Speech Recognition)
-   **Model**: **Gemini 2.5 Pro**.
-   **Transcription**: Converts audio chunks into a structured JSON format, which is then compiled into a master SRT file.
-   **Smart Prompting**:
    -   Removes filler words.
    -   Marks silences explicitly.
    -   Performs grammar correction.
    -   Fixes segmentation logic.
    -   Handles multiple speaker diarization.
-   **Verification**:
    -   **Hallucination Check**: Implements a retry backoff logic if timestamp hallucinations are detected.
    -   **Final Validation**: Verifies timestamps one last time before passing data to the translation layer.

### 3. Context-Aware MT (Machine Translation)
-   **Model**: **Gemini 2.5 Pro**.
-   **Word Budget Method**: Translates text while strictly adhering to the time constraints of the original segment.
-   **Technical Terms**: Prompts the model to **keep technical terms in English** (e.g., "API", "Database").
-   **Adaptive Translation**:
    -   **Elaboration**: Expands short translations to fill time.
    -   **Summarization**: Condenses long translations to fit time.

### 4. Precision TTS (Text-to-Speech) & Sync
-   **Model**: **Google Cloud TTS** (Neural2 voices).
-   **Duration Loop**:
    1.  Calculates the estimated spoken duration of the translated text.
    2.  **Feedback Loop**: If the duration deviation is **> 0.5 seconds**, the text is sent back to the MT layer for re-elaboration or re-summarization.
    3.  **Fine-Tuning**: If the deviation is **≤ 0.5 seconds**, the audio speed is adjusted by up to **±15%** to achieve perfect sync.
-   **Result**: Audio that matches the speaker's lip movements as closely as possible without sounding artificial.

### 5. Final Assembly
-   **Mixing**: Merges the new dubbed audio track with the original video.
-   **Ducking**: Lowers the volume of the original background audio when speech is present, ensuring the dub is clear while retaining ambient sound.

##  Installation & Run

### Prerequisites
-   Node.js & npm
-   Python 3.11+
-   `ffmpeg` installed and added to PATH.
-   Google Gemini API Key

### 1. Backend Setup
```bash
cd Backend
# Create virtual environment (optional but recommended)
python -m venv venv
# Activate venv (Windows: venv\Scripts\activate, Mac/Linux: source venv/bin/activate)

# Install dependencies
pip install fastapi uvicorn python-multipart google-generativeai google-cloud-texttospeech pydub requests python-dotenv

# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run Server
python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Frontend Setup
```bash
cd Frontend
# Install dependencies
npm install

# Run Dev Server
npm run dev
```

Access the app at `http://localhost:5173`.

## 📂 Project Structure

```
DubFlow/
├── Backend/
│   ├── server.py              # FastAPI application & endpoints
│   ├── google_pipeline2.py    # Core AI pipeline logic
│   ├── history.json           # Task persistence
│   ├── Video out/             # Generated dubbed videos
│   ├── SRT/                   # Generated transcripts
│   └── ...
├── Frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React component (Dashboard, Player, Config)
│   │   ├── App.css            # Global styles & animations
│   │   └── ...
│   ├── vite.config.js         # Vite configuration
│   └── ...
└── README.md                  # This file
```

### Done by Abhishek Anand
