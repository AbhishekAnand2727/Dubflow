# DubFlow

**DubFlow** is an advanced AI dubbing automation platform that seamlessly translates and dubs videos into 11 Indian languages. It leverages a state-of-the-art pipeline orchestrated with **Google Gemini Pro 2.5** (for ASR and Adaptive Translation) and **Google Cloud TTS** (Neural2/Journey/Chirp), wrapped in a modern React + FastAPI architecture.

![DubFlow Dashboard](image.png)

## 1. Project Overview

DubFlow solves the complex challenge of video dubbing by automating not just translation, but **synchronization**. The system ensures that the dubbed audio matches the timing of the original speaker, preventing the common "voiceover drift" found in basic tools. It features a "Warm SaaS" UI for managing jobs, editing transcripts, and monitoring real-time progress.

---

## 2. Technology Stack

### Frontend
- **Framework:** React 18 (Vite)
- **Styling:** TailwindCSS (with extensive custom animations and glassmorphism)
- **Icons:** Lucide React
- **State Management:** React Hooks (`useState`, `useEffect`, `useRef`)
- **API Interaction:** Axios
- **Key Components:**
  - `TranscriptViewer`: Interactive segment editor with Merge/Split/Delete capabilities.
  - `PlayerState`: Synchronization of video playback with transcript.
  - `ProcessingState`: Real-time job status polling.

### Backend
- **Server:** FastAPI (Python 3.10+)
- **Database:** SQLite (with Thread-local connection management)
- **Core Processing:**
  - **ASR:** Google **Gemini Pro 2.5** (Multimodal) / Whisper (Legacy/Alternative)
  - **Translation (MT):** Google **Gemini Pro 2.5** (Context-aware, Adaptive)
  - **TTS:** Google Cloud Text-to-Speech (Neural2 / Journey / Chirp voices)
  - **Audio Processing:** PyDub (pydub), FFmpeg
- **Concurrency:** `ThreadPoolExecutor` from `concurrent.futures`

---

## 3. The Dubbing Pipeline (`google_pipeline2.py`)

The pipeline operates in 4 major stages:

### Stage 1: Analysis & ASR
1.  **Extraction:** Audio is extracted from the video using FFmpeg.
2.  **Transcription:** The audio is sent to **Gemini Pro 2.5** to generate a timestamped SRT with speaker diarization.
3.  **Cleaning:** A double-pass cleaning strategy removes hallucinated speaker labels (e.g., `[Speaker 1]`) from the source text.

### Stage 2: Adaptive Translation (The "Brain")
This is the most complex component. It doesn't just translate; it **fits** the translation to the time constraint.
- **WPM Calibration:** Each language has a specific "Words Per Minute" setting (e.g., Hindi: 210, English: 180).
- **Feedback Loop:**
  1.  Generate initial translation.
  2.  Estimate duration based on WPM.
  3.  Compare with `target_duration`.
  4.  **If too long (> 500ms diff):** trigger `summarize` mode ("Make it shorter").
  5.  **If too short (> 500ms diff):** trigger `elaborate` mode ("Add filler words/detail").
  6.  **Panic Mode:** If 2-3 retries fail, strictly truncate or stretch speed.

### Stage 3: TTS Generation
- **Voice Selection:** Matches gender/speaker to specific Google TTS voices.
- **Safeguards:** 
  - **Hallucination Check:** `clean_text_for_tts` removes tags from the *translated* text just before TTS generation to prevent the AI voice from reading metadata.
  - **Caching:** Hashes text+voice+speed to avoid regenerating identical audio.

### Stage 4: Audio Assembly
- **Sequential Assembly:** Segments are appended one by one.
  - If a segment finishes *early*, silence is inserted to match the next start time.
  - If a segment runs *late* (overlap), the next segment starts immediately after, pushing the timeline slightly but **guaranteeing no audio collision**.
- **Mixing:** The final dub track is merged with the video using FFmpeg.

---

## 4. Key Challenges & Solutions

### A. ASR Hallucinations & Artifacts
- **Issue:** ASR models often include "[Speaker 1]" or metadata in the spoken text.
- **Fix:** Implemented a **Double-Pass Cleaning Strategy**:
  1.  **Source Loop:** `while` loop removes *all* leading speaker tags from source text.
  2.  **Target Safety Net:** Cleaning logic applied to the *translated* text before TTS catch hallucinations.

### B. Synchronization & "The Buffer"
- **Issue:** Translations were often too long or too short, causing "rushed" speech or awkward silence.
- **Solution:** **Simplification**. 
  - Tuned **WPM** (Words Per Minute) per language to match reality (e.g., English lowered to 180).
  - Used a constant feedback buffer to drive the adaptive translation engine.

### C. Overlaps
- **Issue:** Independent segments often overlapped when the previous one ran long.
- **Fix:** Switched to **Sequential Assembly**, where audio is placed relative to the previous clip's actual end time, ensuring 0% overlap.

---

## 5. Features

### Interactive Editor
- **Transcript View:** Edit Source and Target text.
- **Timeline Controls:** Edit Start/End timestamps.
- **Operations:**
  - **Merge:** Combine two segments into one.
  - **Split:** (New!) Click a word to split a segment precisely at that point.
  - **Delete/Restore:** Toggle exclusion of segments.

### Dashboard
- **History:** View all past jobs.
- **Status:** Real-time progress updates.
- **Download:** Get final video or SRTs.

---

## 6. Installation & Run

### 1. Backend Setup
```bash
cd Backend
python -m venv venv
# Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env
python -m uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Frontend Setup
```bash
cd Frontend
npm install
npm run dev
```

Access the app at `http://localhost:5173`.
