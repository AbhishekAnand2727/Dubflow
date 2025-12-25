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

### Advanced Video Player \u0026 Expert Review Interface

![DubFlow Video Player](image%20copy.png)

The video player window provides a professional-grade interface for language experts to review and refine dubbing quality:

#### **Dual Transcript View**
- **Side-by-Side Display:** Source and target language transcripts are displayed simultaneously for easy comparison.
- **Auto-Scroll Sync:** Transcripts automatically scroll and highlight the current segment as the video plays, maintaining perfect synchronization.
- **Real-Time Playback:** Click any segment to instantly jump to that point in the video.

#### **Expert Editing Tools**
Language experts have full control to refine translations and ensure quality:

1. **Direct Text Editing:** Both source and target text can be edited inline with live preview.
2. **Timestamp Adjustment:** Precise control over segment start/end times for perfect synchronization.
3. **Segment Operations:** Merge, split, or delete segments as needed for optimal phrasing.

#### **Live ASR Microphone 🎤**
A powerful real-time speech recognition tool for rapid corrections:
- **One-Click Recording:** Language experts can speak directly to update segment text.
- **Language-Aware:** Automatically recognizes the target language for accurate transcription.
- **Instant Updates:** Spoken input immediately replaces or appends to the segment text.

#### **Gemini AI Translation Assistant 🤖**
Built-in AI-powered translation refinement:
- **Customizable Prompts:** Modify the translation behavior via the prompt bar at the top (e.g., \"Use casual conversational language, not formal\").
- **Context-Aware:** Re-generates translations with full context understanding.
- **One-Click Retry:** Click the brain icon (🧠) next to any segment to regenerate its translation with your custom instructions.

#### **Multi-Speaker Voice Selection 🎭**
When the system detects multiple speakers in the video:
- **Automatic Detection:** Speaker diarization identifies different speakers (Speaker 1, Speaker 2, etc.).
- **Voice Customization:** Each speaker gets their own voice dropdown menu.
- **Gender \u0026 Style:** Choose from Neural2, Journey, and Chirp3 voices with gender labels (Male/Female).
- **Persistent Settings:** Selected voices are saved and applied across all segments for that speaker.

#### **Intelligent Redubbing**
After language experts make changes, the redubbing process is highly optimized:

- **Smart Change Detection:** Only modified segments are regenerated, reusing cached audio for unchanged parts.
- **Voice Consistency:** Speaker-specific voice selections are preserved and applied automatically.
- **Instant Updates:** Changes appear immediately after clicking \"Update Dub\".
- **Multi-Speaker Output:** Final video maintains distinct voices for each speaker, creating natural-sounding conversations.

**Example Workflow:**
1. Review the initial dub in the video player
2. Identify a segment that needs refinement
3. Use the live ASR mic to speak a better translation, or
4. Manually edit the text, or  
5. Use the Gemini brain tool with a custom prompt
6. Select appropriate voices for Speaker 2 and Speaker 3
7. Click \"Update Dub\" → Only changed segments regenerate
8. Download the final multi-speaker dubbed video

This end-to-end expert review workflow ensures dubbing quality that rivals professional studios, while maintaining the speed and efficiency of AI automation.

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
