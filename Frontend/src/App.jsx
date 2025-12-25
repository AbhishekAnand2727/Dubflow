// Removed hardcoded constants
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Pause, RotateCcw, Volume2, Globe, Clock, RefreshCw, ChevronLeft, Download, Upload, Trash2, Wand2, Mic, Settings, Loader2, CheckCircle2, AlertCircle, Sparkles, Languages, Activity, Scissors, LayoutTemplate, Terminal, X, FileAudio, FileVideo } from 'lucide-react';

const SERVER_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8002";
const API_BASE = `${SERVER_URL}/api`;

const CONSTANTS = {
  GAP_THRESHOLD: 1,
  SCROLL_THRESHOLD: 50,
  TEXTAREA_CHARS_PER_ROW: 35,
};

function App() {
  const [view, setView] = useState('home');
  const [history, setHistory] = useState([]);
  const [activeTask, setActiveTask] = useState(null);

  // Local State
  const [state, setState] = useState('config'); // config | processing | success
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [taskId, setTaskId] = useState(null);
  const [result, setResult] = useState(null);
  const [transcripts, setTranscripts] = useState({ source: [], target: [] });
  const [pipelineStatus, setPipelineStatus] = useState({ step: 'Initializing', progress: 0, message: '' });
  const [toast, setToast] = useState(null); // { message, type: 'success'|'error'|'info' }
  const toastTimeoutRef = useRef(null);

  const showToast = (message, type = 'info') => {
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    setToast({ message, type });
    toastTimeoutRef.current = setTimeout(() => setToast(null), 3000);
  };

  const [settings, setSettings] = useState({
    inputLang: 'English',
    outputLang: 'Hindi', // Default
    voice: '',
    speed: 1.0,
    durationLimit: null,
    wpm: 150 // Default fallback
  });

  // Global Config Data
  const [supportedLanguages, setSupportedLanguages] = useState([]);
  const [languageDefaults, setLanguageDefaults] = useState({}); // Map: Name -> Default WPM
  const [allVoices, setAllVoices] = useState([]);

  useEffect(() => {
    fetchConfig();
    fetchHistory();
  }, []);

  // Refresh history whenever we navigate back to home
  useEffect(() => {
    if (view === 'home') {
      fetchHistory();
    }
  }, [view]);

  const fetchConfig = async () => {
    try {
      // 1. Try to get everything from /voices (New Backend)
      const voiceRes = await axios.get(`${API_BASE}/voices`);

      let langs = voiceRes.data.languages || [];
      const voices = voiceRes.data.voices || [];

      // Restore settings from LocalStorage
      try {
        const savedSettings = JSON.parse(localStorage.getItem('dubflow_settings') || '{}');
        if (savedSettings.outputLang) {
          setSettings(prev => ({ ...prev, ...savedSettings }));
        }
      } catch (e) {
        console.warn("Failed to parse saved settings:", e);
      }

      // 2. Fallback: If languages missing (Old Backend), fetch from /languages
      if (langs.length === 0) {
        console.warn("Backend missing 'languages' in /voices. Fetching /languages fallback...");
        try {
          const fallbackRes = await axios.get(`${API_BASE}/languages`);
          // Map string list to object structure if needed, or handle strings
          // The new code expects objects with {name, default_wpm}
          // So we adjust:
          langs = (fallbackRes.data.languages || []).map(name => ({ name, default_wpm: 150 }));
        } catch (e) {
          console.error("Fallback /languages failed", e);
        }
      }

      // Setup Defaults Map
      const defaults = {};
      langs.forEach(l => defaults[l.name] = l.default_wpm);
      defaults['English'] = 180; // FORCE OVERRIDE: User requested 180 for English
      defaults['Tamil'] = 160;   // FORCE OVERRIDE: User requested revert to old value
      defaults['Telugu'] = 160;  // FORCE OVERRIDE: Grouping with Tamil
      defaults['Kannada'] = 160; // FORCE OVERRIDE: Grouping with Tamil
      defaults['Malayalam'] = 160;// FORCE OVERRIDE: Grouping with Tamil
      setLanguageDefaults(defaults);

      setSupportedLanguages(langs.map(l => l.name));
      setAllVoices(voices);
    } catch (err) {
      console.error("Failed to fetch config", err);
    }
  };

  // ... rest (keep fetchHistory logic merged or minimal change) ...

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API_BASE}/history`);
      setHistory(res.data);
    } catch (err) {
      console.error("Failed to fetch history", err);
    }
  };

  // Helper to persist settings to localStorage
  const updateSettings = (updates) => {
    setSettings(prev => {
      const next = { ...prev, ...updates };
      try {
        localStorage.setItem('dubflow_settings', JSON.stringify(next));
      } catch (e) {
        console.warn("Failed to save settings:", e);
      }
      return next;
    });
  };

  // --- Handlers ---

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      const validTypes = ['video/', 'audio/'];
      if (validTypes.some(type => droppedFile.type.startsWith(type))) {
        setFile(droppedFile);
      } else {
        showToast("Please upload a valid video or audio file.", "error");
      }
    }
  };

  const [resumeJobId, setResumeJobId] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState(null);
  const [originalFilename, setOriginalFilename] = useState(null); // Added state

  const handleUpload = async () => {
    if (!file) return;

    try {
      // 1. Upload File
      const formData = new FormData();
      formData.append('file', file);

      const uploadRes = await axios.post(`${API_BASE}/upload`, formData, {
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percent);
        }
      });

      const { filename, resume_job_id, original_name, deduplicated } = uploadRes.data;
      setUploadedFilename(filename);
      setOriginalFilename(original_name);

      if (deduplicated) {
        showToast("File recognized! Instant upload.", "success");
      }

      if (resume_job_id) {
        // Found an incomplete job -> Prompt user
        setResumeJobId(resume_job_id);
        return;
      }

      // No resume -> Start Dubbing immediately
      startDubbing(filename);
      setUploadProgress(0); // Reset for next upload

    } catch (err) {
      console.error(err);
      showToast("Error uploading file.", "error");
    }
  };

  const startDubbing = async (fname) => {
    try {
      // Validate voice selection
      if (!settings.voice) {
        showToast("Please select a voice before starting.", "error");
        setState('config');
        return;
      }

      // Immediate feedback to user
      setState('processing');
      setPipelineStatus({ step: 'Initializing', progress: 0, message: 'Starting job on server...' });
      setActiveTask(null); // Clear any previous task context

      const dubRes = await axios.post(`${API_BASE}/dub`, {
        filename: fname,
        input_lang: settings.inputLang,
        output_lang: settings.outputLang,
        target_voice: settings.voice,
        speed: settings.speed,
        duration_limit: settings.durationLimit,
        wpm: settings.wpm
      });

      setTaskId(dubRes.data.task_id);

      if (!dubRes.data.task_id) {
        console.error("CRITICAL: No task_id returned from server!", dubRes);
        showToast("Error: Server did not return a Task ID. Check console.", "error");
        setState('config');
        return;
      }
      console.log("Job Started. Task ID:", dubRes.data.task_id);

      setResumeJobId(null); // Clear
    } catch (err) {
      console.error(err);
      showToast("Error starting dubbing.", "error");
      setState('config'); // Revert state on error
    }
  };

  const handleResume = async () => {
    if (!resumeJobId) return;
    try {
      // Call Resume Endpoint
      await axios.post(`${API_BASE}/resume`, { job_id: resumeJobId });

      setTaskId(resumeJobId);
      setState('processing');
      setResumeJobId(null);
    } catch (err) {
      console.error("Resume failed", err);
      showToast("Failed to resume task. Starting fresh instead is recommended.", "error");
    }
  };

  // --- Polling for Status ---
  // --- Polling for Status ---
  // --- Polling for Status ---
  useEffect(() => {
    let interval;
    if (state === 'processing' && taskId) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`${API_BASE}/status/${taskId}`);
          const data = res.data;

          setPipelineStatus({
            step: data.step,
            progress: data.progress,
            message: data.message
          });

          if (data.status === 'completed') {
            console.log("Polling: Task completed. Fetching Result...");
            clearInterval(interval);
            fetchResult(taskId);
          } else if (data.status === 'failed') {
            clearInterval(interval);
            showToast(`Task Failed: ${data.message}`, "error");
            setState('config');
          }
        } catch (err) {
          console.error("Polling error", err);
        }
      }, 2000); // Increased to 2000ms
    }
    return () => clearInterval(interval);
  }, [state, taskId]);

  const fetchResult = async (id) => {
    try {
      console.log(`Fetching result for ${id}...`);
      // alert(`Debug: Fetching result for ${id}`);
      const res = await axios.get(`${API_BASE}/result/${id}`);
      console.log("Result received:", res.data);
      setResult(res.data);
      setTranscripts({
        source: res.data.source_segments,
        target: res.data.target_segments
      });
      // Update history to include the new task immediately
      fetchHistory();

      // Find the task details from history (or construct temporary one)
      const taskRes = await axios.get(`${API_BASE}/status/${id}`);
      setActiveTask(taskRes.data);

      console.log("Setting active task and switching view...");
      // alert("Debug: Switching to Player View");
      setState('success');
      setView('player');
    } catch (err) {
      console.error("fetchResult Error:", err);
      showToast("Failed to fetch results.", "error");
    }
  };

  // --- Components ---

  const Header = () => (
    <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-orange-100 shadow-sm">
      <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
        <div
          className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
          onClick={() => {
            setView('home');
            setActiveTask(null);
            setState('config'); // Reset state to safe default for new actions
          }}
          role="button"
          aria-label="Go to Home"
        >
          <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center shadow-lg shadow-orange-500/20 backdrop-blur-sm overflow-hidden">
            <img src="/logo.png" alt="Logo" className="w-full h-full object-cover" />
          </div>
          <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-600 to-amber-600">
            DubFlow
          </span>
        </div>
        <div className="flex items-center gap-4">
          {/* Show explicit Dashboard button if not on home */}
          {view !== 'home' && (
            <button
              onClick={() => {
                setView('home');
                setActiveTask(null);
                setState('config');
              }}
              className="text-sm font-bold text-slate-500 hover:text-orange-600 flex items-center gap-2 transition-colors"
            >
              <LayoutTemplate size={16} />
              Dashboard
            </button>
          )}
        </div>
      </div>
    </header>
  );

  const HomeState = () => (
    <div className="max-w-6xl mx-auto mt-8 px-6 animate-in fade-in duration-500">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-black text-slate-900">Your Dubs</h1>
          <p className="text-slate-500 font-medium">Library of translated videos</p>
        </div>
        <button
          onClick={() => {
            setView('new');
            setState('config');
            setFile(null);
            setResult(null);
            setActiveTask(null);
          }}
          className="px-6 py-3 rounded-xl font-bold text-white bg-orange-600 hover:bg-orange-700 transition-colors shadow-lg shadow-orange-500/20 flex items-center gap-2"
        >
          <Wand2 size={20} /> New Dub
        </button>
      </div>

      {history.length === 0 ? (
        <div className="text-center py-20 bg-white/50 rounded-3xl border border-dashed border-slate-300">
          <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4 text-orange-500">
            <Wand2 size={32} />
          </div>
          <h3 className="text-xl font-bold text-slate-900">No dubs yet</h3>
          <p className="text-slate-500 mt-2">Create your first dubbed video to see it here.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {history.map((task) => (
            <div
              key={task.id}
              onClick={(e) => loadTask(task.id, e)}
              className="group relative bg-white border border-slate-100 rounded-2xl p-4 shadow-sm hover:shadow-xl transition-all hover:-translate-y-1 overflow-hidden cursor-pointer"
            >
              {/* Thumbnail */}
              <div
                className="aspect-video bg-slate-900 rounded-xl mb-4 overflow-hidden relative"
              >
                <img
                  src={`${SERVER_URL}/api/thumbnail/${task.id}`}
                  onError={(e) => e.target.style.display = 'none'}
                  className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-12 h-12 rounded-full bg-white/20 backdrop-blur-md flex items-center justify-center text-white shadow-lg group-hover:scale-110 transition-transform">
                    <Play fill="currentColor" size={20} />
                  </div>
                </div>

                {/* Status Badge */}
                <div className="absolute top-2 left-2">
                  {task.status === 'completed' && <span className="px-2 py-1 rounded-md bg-green-500/90 text-white text-xs font-bold backdrop-blur-sm shadow-sm">Completed</span>}
                  {task.status === 'processing' && <span className="px-2 py-1 rounded-md bg-amber-500/90 text-white text-xs font-bold backdrop-blur-sm shadow-sm animate-pulse">Processing</span>}
                  {task.status === 'failed' && <span className="px-2 py-1 rounded-md bg-red-500/90 text-white text-xs font-bold backdrop-blur-sm shadow-sm">Failed</span>}
                </div>

                {/* Language Flag (Target) */}
                <div className="absolute bottom-2 right-2 px-2 py-1 rounded-md bg-black/60 text-white text-xs font-bold backdrop-blur-sm border border-white/10">
                  {task.output_lang}
                </div>

                {/* Delete Button - Inside Video Box */}
                <button
                  type="button"
                  onClick={(e) => handleDelete(e, task.id)}
                  className="absolute top-2 right-2 p-2 bg-slate-200 hover:bg-red-200 text-slate-500 hover:text-red-600 rounded-full transition-colors z-[60] shadow-sm border border-slate-300 cursor-pointer opacity-0 group-hover:opacity-100"
                  title="Delete Video"
                >
                  <Trash2 size={16} />
                </button>
              </div>

              {/* Info */}
              <div>
                <h3 className="font-bold text-slate-800 truncate leading-tight" title={task.original_filename || task.filename}>
                  {task.original_filename || task.filename}
                </h3>
                <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                  <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-orange-500" /> {task.input_lang}</span>
                  <span>→</span>
                  <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-amber-500" /> {task.output_lang}</span>
                </div>
                <div className="mt-4 text-xs text-slate-400 font-medium">
                  {(() => {
                    try {
                      const dateStr = task.created_at || task.timestamp; // Support both
                      if (!dateStr) return "Unknown Date";
                      // Handle SQLite format "YYYY-MM-DD HH:MM:SS" -> "YYYY-MM-DDTHH:MM:SS" for Safari/Chrome consistency
                      const safeDate = dateStr.replace(" ", "T");
                      return new Date(safeDate).toLocaleDateString(undefined, {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      });
                    } catch (e) {
                      return "Invalid Date";
                    }
                  })()}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const handleDelete = async (e, id) => {
    e.stopPropagation(); // Just stop propagation, sufficient for React

    if (!window.confirm("Are you sure you want to delete this dub? This action cannot be undone.")) return;

    try {
      await axios.delete(`${API_BASE}/delete/${id}`);
      setHistory(prev => prev.filter(t => t.id !== id));

      if (activeTask && activeTask.id === id) {
        setResult(null);
        setActiveTask(null);
      }
    } catch (err) {
      console.error("Delete failed", err);
      showToast("Failed to delete task.", "error");
    }
  };

  const loadTask = async (id, e) => {
    // Ultimate Safety: If this click came from the Delete button, IGNORE IT.
    if (e && e.target && e.target.closest('button')) {
      return;
    }

    if (e) {
      e.stopPropagation();
      e.preventDefault();
    }

    // Check status first
    const task = history.find(t => t.id === id);
    if (task && task.status === 'processing') {
      setActiveTask(task);
      setTaskId(task.id);
      setState('processing');
      setView('new');
      return;
    }

    try {
      const res = await axios.get(`${API_BASE}/result/${id}`);
      setResult(res.data);
      setTranscripts({
        source: res.data.source_segments,
        target: res.data.target_segments
      });
      // FIX: Ensure activeTask always has id (history.find can return undefined)
      setActiveTask(task || { id, ...res.data });
      setView('player');
    } catch (err) {
      console.error(err);
      showToast("Failed to load task.", "error");
    }
  };

  const ConfigState = () => (
    <div className="max-w-3xl mx-auto mt-12 space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <button
        onClick={() => setView('home')}
        className="flex items-center gap-2 text-slate-500 hover:text-orange-600 transition-colors font-bold mb-4"
      >
        <ChevronLeft size={20} /> Back to Dashboard
      </button>
      <div className="text-center space-y-2">
        <h1 className="text-5xl font-black text-slate-900 drop-shadow-sm tracking-tight">
          Indic Video Dubbing
        </h1>
        <p className="text-xl text-slate-600 font-medium max-w-lg mx-auto">
          Translate and lip-sync your content into 11 Indian languages instantly.
        </p>
      </div>

      {/* Upload Zone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className={`
          relative group cursor-pointer
          border-3 border-dashed rounded-3xl p-12
          transition-all duration-300 ease-out backdrop-blur-sm
          ${file
            ? 'border-orange-500 bg-orange-50/80 shadow-xl shadow-orange-500/10'
            : 'border-slate-300 bg-white/60 hover:border-orange-400 hover:bg-white/80 hover:shadow-lg hover:shadow-orange-500/5'}
        `}
      >
        <input
          id="file-upload"
          name="file-upload"
          type="file"
          className="absolute inset-0 opacity-0 cursor-pointer"
          onChange={(e) => setFile(e.target.files[0])}
          accept="video/*,audio/*"
        />
        <div className="flex flex-col items-center gap-4 text-center">
          <div className={`
            w-20 h-20 rounded-2xl flex items-center justify-center shadow-md
            transition-colors duration-300
            ${file ? 'bg-orange-100 text-orange-600' : 'bg-white text-slate-400 group-hover:text-orange-500'}
          `}>
            {file ? (file.type.startsWith('audio') ? <FileAudio size={40} /> : <FileVideo size={40} />) : <Upload size={40} />}
          </div>
          <div>
            <h3 className="text-xl font-bold text-slate-900">
              {file ? file.name : "Drop your video or audio here"}
            </h3>
            <p className="text-slate-500 mt-1 font-medium">
              {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "or click to browse"}
            </p>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      <div className="bg-white/90 backdrop-blur-md rounded-3xl shadow-xl shadow-orange-900/5 p-8 border border-white/50">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Languages size={16} className="text-orange-500" /> Source Language
            </label>
            <select
              className="w-full p-3 rounded-xl bg-slate-50 border-2 border-transparent focus:border-orange-500 focus:ring-0 text-slate-900 font-semibold transition-all"
              value={settings.inputLang}
              onChange={(e) => updateSettings({ inputLang: e.target.value })}
            >
              {supportedLanguages.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Languages size={16} className="text-orange-500" /> Target Language
            </label>
            <select
              className="w-full p-3 rounded-xl bg-slate-50 border-2 border-transparent focus:border-orange-500 focus:ring-0 text-slate-900 font-semibold transition-all"
              value={settings.outputLang}
              onChange={(e) => {
                const newLang = e.target.value;
                const newDefaultWpm = languageDefaults[newLang] || 150;
                // User Request: Hindi defaults to 0.85x speed
                const newSpeed = newLang === 'Hindi' ? 0.85 : 1.0;
                updateSettings({ outputLang: newLang, wpm: newDefaultWpm, speed: newSpeed });
              }}
            >
              {supportedLanguages.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Mic size={16} className="text-orange-500" /> Voice Model
            </label>
            <select
              className="w-full p-3 rounded-xl bg-slate-50 border-2 border-transparent focus:border-orange-500 focus:ring-0 text-slate-900 font-semibold transition-all"
              value={settings.voice}
              onChange={(e) => updateSettings({ voice: e.target.value })}
            >
              <option value="">Select a Voice</option>
              {(() => {
                const voices = allVoices.filter(v => v.lang === settings.outputLang);
                const grouped = voices.reduce((acc, v) => {
                  if (!acc[v.category]) acc[v.category] = [];
                  acc[v.category].push(v);
                  return acc;
                }, {});

                return Object.entries(grouped).map(([category, catVoices]) => (
                  <optgroup key={category} label={category}>
                    {catVoices.map(v => (
                      <option key={v.id} value={v.id}>
                        {v.name} ({v.gender})
                      </option>
                    ))}
                  </optgroup>
                ));
              })()}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Activity size={16} className="text-orange-500" /> Target Words Per Minute ({settings.wpm})
            </label>
            <input
              type="range"
              min="50" max="250" step="1"
              className="w-full accent-orange-600 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              value={settings.wpm}
              onChange={(e) => updateSettings({ wpm: parseInt(e.target.value) })}
              title="Higher WPM = Faster Speech. Lower WPM = Slower Speech."
            />
            <div className="flex justify-between text-xs text-slate-400 font-medium px-1">
              <span>Slow (50)</span>
              <span>Fast (250)</span>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Clock size={16} className="text-orange-500" /> Duration Limit (Sec)
            </label>
            <input
              type="number"
              placeholder="Full Video (Default)"
              className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-orange-500 outline-none transition-all font-medium text-slate-700"
              value={settings.durationLimit || ''}
              onChange={(e) => updateSettings({ durationLimit: e.target.value ? parseInt(e.target.value) : null })}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Activity size={16} className="text-orange-500" /> TTS Speed ({settings.speed}x)
            </label>
            <input
              type="range"
              min="0.5" max="1.5" step="0.05"
              className="w-full accent-orange-600 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              value={settings.speed}
              onChange={(e) => updateSettings({ speed: parseFloat(e.target.value) })}
              title="Controls the speaking rate of the TTS voice."
            />
            <div className="flex justify-between text-xs text-slate-400 font-medium px-1">
              <span>0.5x</span>
              <span>1.0x</span>
              <span>1.5x</span>
            </div>
          </div>
        </div>



        <button
          onClick={handleUpload}
          disabled={!file || uploadProgress > 0}
          className={`
            w-full mt-8 py-4 rounded-2xl font-bold text-lg flex items-center justify-center gap-3
            transition-all duration-300 transform
            ${file
              ? 'bg-gradient-to-r from-orange-600 to-amber-500 text-white shadow-xl shadow-orange-500/30 hover:shadow-orange-500/50 hover:scale-[1.02] active:scale-[0.98]'
              : 'bg-slate-100 text-slate-400 cursor-not-allowed'}
          `}
        >
          {uploadProgress > 0 && uploadProgress < 100 ? (
            <>
              <Loader2 size={24} className="animate-spin" />
              Uploading... {uploadProgress}%
            </>
          ) : uploadProgress === 100 ? (
            <>
              <Loader2 size={24} className="animate-spin" />
              Verifying...
            </>
          ) : (
            <>
              <Wand2 size={24} />
              {file ? 'Dub Video Now' : 'Select a file to start'}
            </>
          )}
        </button>
      </div>
    </div >
  );

  /* Removed inner ProcessingState */

  /* --- Transcript Viewer with Expert Tools --- */
  const TranscriptViewer = ({ sourceSegments, targetSegments, currentTime, onUpdate, onSegmentClick, onMerge, onAdd, onDelete, deletedIndices, availableVoices, speakerOverrides, onVoiceOverride, inputLang, outputLang, onSplit, edits, onEdit }) => {
    const containerRef = useRef(null);
    const activeRef = useRef(null);
    const [editingTimestamp, setEditingTimestamp] = useState(null);
    const [activeMicSegment, setActiveMicSegment] = useState(null);
    // Removed internal edits state
    const [loadingSegment, setLoadingSegment] = useState(null); // { index, type: 'brain' | 'mic' }
    const [geminiPrompt, setGeminiPrompt] = useState("");
    const [splittingIndex, setSplittingIndex] = useState(null); // Index of segment currently being split
    const recognitionRef = useRef(null);

    // Removed duplicate handleEdit

    // Cleanup Speech Recognition on unmount
    useEffect(() => {
      return () => {
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
      };
    }, []);

    useEffect(() => {
      if (activeRef.current && containerRef.current) {
        activeRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, [currentTime]);

    // Handle timestamp editing
    const handleTimestampEdit = (index, field, value) => {
      // Validate SRT timestamp format: HH:MM:SS,mmm
      const timestampRegex = /^\d{2}:\d{2}:\d{2},\d{3}$/;
      if (!timestampRegex.test(value)) {
        showToast('Invalid timestamp format. Use HH:MM:SS,mmm', "error");
        return; // Fix: Stop execution
      }

      // Update the segment
      // This is handled by onUpdate prop, which updates localTranscripts in PlayerState
      if (onUpdate) {
        onUpdate('timestamp', index, { field, value });
      }

      setEditingTimestamp(null);
    };

    // --- Gemini Helper ---
    const handleBrainClick = async (index, type) => {
      // Logic:
      // If Target Box (type='target') clicked -> Translate Source Text -> Target Lang
      // If Source Box (type='source') clicked -> Translate Target Text -> Source Lang (Back Translation?)
      // User Request: "If the text in the original segment is highlighted then it needs to translate it to the original language."
      // Interpreted: Highlighted text in one box -> Replace/Append with translation to OTHER box?
      // Or simple: Just re-translate the WHOLE segment based on context.

      // Let's implement: Context-Aware Re-translation of the WHOLE segment for now, or text selection if possible.
      // Getting selection from specific textarea is tricky in React loop without refs for each.
      // Let's stick to: "Click Brain next to Target -> Re-translate Source to Target using Prompt".

      const sourceText = edits.source[index] !== undefined ? edits.source[index] : sourceSegments[index].text;

      // If Brain next to Source is clicked... maybe "Translate Target to Source"?
      // Let's assume Right Side Buttons are for the TARGET mainly (Expert Review).
      // But user said "2 buttons to the right".
      // Let's put buttons on the FAR RIGHT.

      // Mode: Translate Source -> Target (Standard Helper)
      setLoadingSegment({ index, type: 'target' });
      try {
        const res = await axios.post(`${API_BASE}/tool/translate`, {
          text: sourceText,
          source_lang: inputLang || "English",
          target_lang: outputLang || "Hindi",
          prompt: geminiPrompt
        });
        onEdit('target', index, res.data.translated); // Use onEdit prop
      } catch (err) {
        console.error(err);
        showToast("Gemini Error: " + err.message, "error");
      } finally {
        setLoadingSegment(null);
      }
    };

    // --- Real-Time Speech Recognition (Web Speech API) ---
    const toggleMic = async (index) => {
      // Stop (If already active)
      if (activeMicSegment === index) {
        console.log("Mic: Stopping...");
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
        setActiveMicSegment(null);
        return;
      }

      // Start (New Recording)
      // If another mic is active, stop it first
      if (activeMicSegment !== null) {
        showToast("Another microphone is active. Stop it first.", "error");
        return;
      }

      try {
        // Check for Web Speech API support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
          showToast("Speech Recognition not supported in this browser. Use Chrome/Edge.", "error");
          return;
        }

        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;

        // Configure recognition
        const langCode = getLangCode(outputLang);
        console.log("🎤 Speech Recognition Config:", { outputLang, langCode });
        recognition.lang = langCode;
        recognition.continuous = true; // Keep listening
        recognition.interimResults = true; // Show live results

        let finalTranscript = '';

        recognition.onstart = () => {
          console.log("✅ Recognition started successfully, lang:", recognition.lang);
        };

        recognition.onresult = (event) => {
          console.log("📝 Got result event:", event);
          let interimTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            console.log(`Result ${i}:`, {
              transcript,
              isFinal: event.results[i].isFinal,
              confidence: event.results[i][0].confidence
            });
            if (event.results[i].isFinal) {
              finalTranscript += transcript + ' ';
            } else {
              interimTranscript += transcript;
            }
          }

          // Update the text field in real-time
          const currentText = edits.target[index] !== undefined ? edits.target[index] : (targetSegments[index]?.text || "");
          const baseText = currentText.split('[SPEAKING]')[0].trim(); // Remove old interim marker
          const newText = (baseText + ' ' + finalTranscript + (interimTranscript ? `[SPEAKING: ${interimTranscript}]` : '')).trim();
          console.log("💬 Updating text:", newText);
          onEdit('target', index, newText); // Use onEdit prop
        };

        recognition.onend = () => {
          console.log("⏹️ Recognition ended. Final transcript:", finalTranscript);
          // Just clean up interim markers, keep final transcript
          const currentText = edits.target[index] !== undefined ? edits.target[index] : (targetSegments[index]?.text || "");
          const cleanedText = currentText.replace(/\[SPEAKING:.*?\]/g, '').trim();

          // Only update if there's actually a change (to preserve final transcript)
          if (cleanedText !== currentText) {
            onEdit('target', index, cleanedText); // Use onEdit prop
          }

          setActiveMicSegment(null);
          console.log("Speech recognition stopped. Final text preserved.");
        };

        recognition.onerror = (event) => {
          console.error("❌ Speech recognition error:", event.error, event);
          if (event.error === 'no-speech') {
            // Ignore no-speech errors, just continue
          } else {
            showToast(`Speech recognition error: ${event.error}`, "error");
          }
          setActiveMicSegment(null);
        };

        recognition.start();
        setActiveMicSegment(index);
        console.log("Real-time speech recognition started");

      } catch (err) {
        console.error("Speech Recognition Error:", err);
        showToast("Could not start speech recognition: " + err.message, "error");
        setActiveMicSegment(null);
      }
    };

    // Convert milliseconds or SRT timestamp to SRT format (HH:MM:SS,mmm)
    const formatTimestamp = (time) => {
      // If already in SRT format, return as-is
      if (typeof time === 'string' && time.includes(':')) {
        return time;
      }

      // Convert milliseconds to SRT format
      const ms = typeof time === 'number' ? time : parseInt(time);
      const hours = Math.floor(ms / 3600000);
      const minutes = Math.floor((ms % 3600000) / 60000);
      const seconds = Math.floor((ms % 60000) / 1000);
      const milliseconds = ms % 1000;

      return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')},${String(milliseconds).padStart(3, '0')}`;
    };

    // Helper to extract speaker number from text
    const extractSpeaker = (text) => {
      if (!text) return null;
      // Match ALL speaker tags and get the LAST one (correct speaker)
      // E.g., "[Speaker 1] [Speaker 2] text" -> Speaker 2 is correct
      const matches = text.match(/\[Speaker (\d+)\]/g);
      if (!matches || matches.length === 0) return null;
      const lastMatch = matches[matches.length - 1];
      const speakerNum = lastMatch.match(/\[Speaker (\d+)\]/);
      return speakerNum ? parseInt(speakerNum[1]) : null;
    };

    // Helper to strip speaker tags from display text
    const stripSpeakerTags = (text) => {
      if (!text) return text;
      return text.replace(/\[Speaker \d+\]\s*/g, '').trim();
    };

    // Helper not strictly needed for MediaRecorder ASR as backend handles it, but good to keep if we switch back
    // Map Lang Name to Code (Simple fallback)
    const getLangCode = (name) => {
      const map = { 'Hindi': 'hi-IN', 'Tamil': 'ta-IN', 'Malayalam': 'ml-IN', 'English': 'en-US', 'Telugu': 'te-IN', 'Kannada': 'kn-IN', 'Marathi': 'mr-IN', 'Gujarati': 'gu-IN', 'Bengali': 'bn-IN', 'Punjabi': 'pa-IN' };
      return map[name] || 'en-US';
    };

    return (
      <div className="flex flex-col h-full">
        {/* Top Bar: Prompt Config */}
        <div className="p-4 border-b border-orange-100 bg-orange-50/50 flex items-center gap-3">
          <Wand2 size={18} className="text-orange-600" />
          <input
            type="text"
            placeholder="Custom Instructions (e.g., 'Use casual conversational language, not formal')"
            className="flex-1 bg-white border border-orange-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-orange-500 outline-none"
            value={geminiPrompt}
            onChange={(e) => setGeminiPrompt(e.target.value)}
          />
        </div>

        <div ref={containerRef} className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
          {sourceSegments.length === 0 ? (
            <div className="text-center py-20 text-slate-400">
              <Activity size={48} className="mx-auto mb-4 opacity-50" />
              <p className="font-medium">No segments to display</p>
            </div>
          ) : sourceSegments.map((seg, i) => {
            const start = parseTime(seg.start);
            const end = parseTime(seg.end);
            const isActive = currentTime >= start && currentTime <= end;

            const sourceText = edits.source[i] !== undefined ? edits.source[i] : seg.text;
            const targetText = edits.target[i] !== undefined ? edits.target[i] : (targetSegments[i]?.text || "");
            const isDeleted = !!deletedIndices[i];

            // Extract speaker and clean display text
            const speaker = extractSpeaker(sourceText);
            const displaySourceText = stripSpeakerTags(sourceText);
            const displayTargetText = stripSpeakerTags(targetText);

            // Get available voices: Target language first, then 4-5 English backup options
            const outputLang = activeTask?.output_lang || result?.output_lang || settings.outputLang;
            const targetLangVoices = availableVoices.filter(v => v.lang === outputLang);
            const englishVoices = availableVoices.filter(v => v.lang === 'English').slice(0, 5); // Limit to 5 English voices
            const relevantVoices = [...targetLangVoices, ...englishVoices];

            // Detect gap with next segment
            const nextSeg = sourceSegments[i + 1];
            const hasGap = nextSeg && parseTime(nextSeg.start) - end > CONSTANTS.GAP_THRESHOLD;
            const gapDuration = hasGap ? parseTime(nextSeg.start) - end : 0;

            return (
              <React.Fragment key={i}>
                <div
                  ref={isActive ? activeRef : null}
                  onClick={() => !isDeleted && onSegmentClick && onSegmentClick(seg)}
                  className={`
                    grid grid-cols-[1fr_auto_1fr_auto] gap-4 p-6 rounded-2xl transition-all duration-500 items-start group cursor-pointer
                    ${isActive
                      ? 'bg-orange-50 scale-[1.01] shadow-lg shadow-orange-500/10 border-orange-200'
                      : 'hover:bg-slate-50 border border-transparent hover:border-slate-100 hover:shadow-md'}
                    ${isDeleted ? 'opacity-40 grayscale decoration-slice line-through' : ''}
                `}
                  title="Click to jump to this segment"
                >
                  {/* Source Text */}
                  <div className="space-y-2">
                    {/* Editable Timestamp Display */}
                    <div className="flex items-center gap-2 text-xs font-mono mb-2" onClick={(e) => e.stopPropagation()}>
                      <Clock size={12} className="text-slate-400" />

                      {/* Start Time */}
                      {editingTimestamp?.index === i && editingTimestamp?.field === 'start' ? (
                        <input
                          type="text"
                          className="bg-white border border-orange-300 rounded px-1 py-0.5 w-28 focus:ring-1 focus:ring-orange-500 outline-none text-xs"
                          defaultValue={formatTimestamp(seg.start)}
                          onClick={(e) => e.stopPropagation()}
                          onBlur={(e) => handleTimestampEdit(i, 'start', e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleTimestampEdit(i, 'start', e.target.value)}
                          autoFocus
                        />
                      ) : (
                        <span
                          className="text-slate-600 cursor-pointer hover:text-orange-600 hover:underline"
                          onClick={(e) => { e.stopPropagation(); setEditingTimestamp({ index: i, field: 'start' }); }}
                          title="Click to edit"
                        >
                          {formatTimestamp(seg.start)}
                        </span>
                      )}

                      <span className="text-slate-300">→</span>

                      {/* End Time */}
                      {editingTimestamp?.index === i && editingTimestamp?.field === 'end' ? (
                        <input
                          type="text"
                          className="bg-white border border-orange-300 rounded px-1 py-0.5 w-28 focus:ring-1 focus:ring-orange-500 outline-none text-xs"
                          defaultValue={formatTimestamp(seg.end)}
                          onClick={(e) => e.stopPropagation()}
                          onBlur={(e) => handleTimestampEdit(i, 'end', e.target.value)}
                          onKeyDown={(e) => e.key === 'Enter' && handleTimestampEdit(i, 'end', e.target.value)}
                          autoFocus
                        />
                      ) : (
                        <span
                          className="text-slate-600 cursor-pointer hover:text-orange-600 hover:underline"
                          onClick={(e) => { e.stopPropagation(); setEditingTimestamp({ index: i, field: 'end' }); }}
                          title="Click to edit"
                        >
                          {formatTimestamp(seg.end)}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 mb-1">
                      {speaker ? (
                        <>
                          <span className="text-xs font-bold text-orange-600 uppercase tracking-wider">Speaker {speaker}</span>
                          {speaker >= 2 && relevantVoices.length > 0 && (
                            <select
                              className="text-xs bg-white border border-orange-200 rounded px-2 py-0.5 text-slate-700 focus:ring-2 focus:ring-orange-500 outline-none cursor-pointer"
                              value={speakerOverrides[`Speaker ${speaker}`] || ""}
                              onChange={(e) => onVoiceOverride && onVoiceOverride(`Speaker ${speaker}`, e.target.value)}
                              onClick={(e) => e.stopPropagation()}
                            >
                              <option value="">Default Voice</option>
                              {relevantVoices.map(v => (
                                <option key={v.id} value={v.id}>{v.name} - {v.gender} ({v.lang})</option>
                              ))}
                            </select>
                          )}
                        </>
                      ) : (
                        <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Speaker 1</span>
                      )}
                    </div>
                    <textarea
                      className={`w-full bg-transparent border-none resize-none focus:ring-0 p-0 text-lg font-medium leading-relaxed ${isActive ? 'text-slate-900' : 'text-slate-500'}`}
                      value={displaySourceText}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => onEdit('source', i, e.target.value)}
                      rows={Math.max(1, Math.ceil(sourceText.length / CONSTANTS.TEXTAREA_CHARS_PER_ROW))}
                      disabled={isDeleted}
                    />
                  </div>

                  {/* Word Split Selection Mode */}
                  {splittingIndex === i && (
                    <div className="absolute inset-0 bg-white/95 backdrop-blur-sm z-20 flex flex-col items-center justify-center p-4 rounded-2xl animate-in fade-in zoom-in-95 duration-200 border-2 border-orange-200 shadow-xl">
                      <h4 className="text-sm font-bold text-orange-600 mb-3 bg-orange-50 px-3 py-1 rounded-full uppercase tracking-wider">Click word to split before it</h4>
                      <div className="flex flex-wrap gap-2 justify-center max-h-full overflow-y-auto w-full">
                        {sourceText.split(/\s+/).map((word, wordIdx) => (
                          <button
                            key={wordIdx}
                            onClick={(e) => { e.stopPropagation(); onSplit(i, wordIdx); setSplittingIndex(null); }}
                            className="px-2 py-1 rounded-md bg-slate-100 hover:bg-orange-500 hover:text-white transition-all text-sm font-medium border border-slate-200 hover:border-orange-600 hover:scale-105 active:scale-95"
                            disabled={wordIdx === 0} // Can't split before first word
                          >
                            {word}
                          </button>
                        ))}
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); setSplittingIndex(null); }}
                        className="mt-4 text-xs text-slate-400 hover:text-slate-600 underline"
                      >
                        Cancel
                      </button>
                    </div>
                  )}

                  {/*Center Actions (Delete + Merge) */}
                  <div className="flex flex-col items-center gap-2 pt-8 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => { e.stopPropagation(); onDelete(i); }}
                      className={`p-1.5 rounded-full transition-colors ${isDeleted ? 'bg-red-100 text-red-600' : 'text-slate-300 hover:text-red-500 hover:bg-slate-100'}`}
                      title={isDeleted ? "Restore" : "Delete Segment"}
                    >
                      {isDeleted ? <RefreshCw size={14} /> : <Trash2 size={14} />}
                    </button>

                    {/* Merge Button - only show if not last segment */}
                    {i < sourceSegments.length - 1 && !isDeleted && (
                      <button
                        onClick={(e) => { e.stopPropagation(); onMerge && onMerge(i); }}
                        className="p-1.5 rounded-full text-slate-300 hover:text-blue-500 hover:bg-slate-100 transition-colors"
                        title="Merge with next segment"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 5v14M5 12h14"></path>
                        </svg>
                      </button>
                    )}

                    {/* Split Button - Adjacent to Merge */}
                    {!isDeleted && (
                      <div className="relative group/split">
                        <button
                          onClick={(e) => { e.stopPropagation(); setSplittingIndex(splittingIndex === i ? null : i); }}
                          className={`p-1.5 rounded-full transition-colors ${splittingIndex === i ? 'bg-orange-100 text-orange-600' : 'text-slate-300 hover:text-orange-500 hover:bg-slate-100'}`}
                          title="Split Segment"
                        >
                          <Scissors size={14} className={splittingIndex === i ? "rotate-90" : ""} />
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Target Text */}
                  <div className="space-y-2">
                    <textarea
                      className={`w-full bg-transparent border-none resize-none focus:ring-0 p-0 text-lg font-medium leading-relaxed ${isActive ? 'text-orange-700' : 'text-slate-400'}`}
                      value={displayTargetText}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => onEdit('target', i, e.target.value)}
                      rows={Math.max(1, Math.ceil(targetText.length / CONSTANTS.TEXTAREA_CHARS_PER_ROW))}
                      dir="auto"
                      disabled={isDeleted}
                      placeholder="Translate..."
                    />
                  </div>

                  {/* Right Actions (Expert Tools) */}
                  <div className="flex flex-col gap-2 pt-2 opacity-100">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleBrainClick(i, 'target'); }}
                      disabled={loadingSegment?.index === i}
                      className={`p-2 rounded-lg transition-all ${loadingSegment?.index === i ? 'bg-orange-100 text-orange-600 animate-pulse' : 'bg-white border border-slate-200 text-slate-400 hover:text-orange-600 hover:border-orange-200 hover:shadow-sm'}`}
                      title="Re-translate with Gemini (Context Aware)"
                    >
                      {loadingSegment?.index === i ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}
                    </button>

                    <button
                      onClick={(e) => { e.stopPropagation(); toggleMic(i); }}
                      disabled={loadingSegment?.index === i && loadingSegment?.type === 'mic'}
                      className={`p-2 rounded-lg transition-all ${activeMicSegment === i ? 'bg-red-500 text-white shadow-md animate-pulse' : 'bg-white border border-slate-200 text-slate-400 hover:text-red-600 hover:border-red-200 hover:shadow-sm'}`}
                      title="Live Dictation (ASR)"
                    >
                      {(loadingSegment?.index === i && loadingSegment?.type === 'mic') ? <Loader2 size={16} className="animate-spin" /> : <Mic size={16} />}
                    </button>
                  </div>
                </div>

                {/* Gap Detection - Add Segment Button */}
                {hasGap && (
                  <div className="flex items-center justify-center py-4">
                    <div className="flex-1 border-t border-dashed border-slate-300"></div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        const gapStart = end;
                        const gapEnd = parseTime(nextSeg.start);
                        if (onAdd) onAdd(i, gapStart, gapEnd);
                      }}
                      className="px-4 py-2 bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-lg hover:from-orange-600 hover:to-orange-700 transition-all shadow-md hover:shadow-lg flex items-center gap-2 text-sm font-medium"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <line x1="5" y1="12" x2="19" y2="12" />
                      </svg>
                      Add Segment ({gapDuration.toFixed(1)}s gap)
                    </button>
                    <div className="flex-1 border-t border-dashed border-slate-300"></div>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    );
  };

  const parseTime = (timeStr) => {
    if (typeof timeStr === 'number') return timeStr / 1000;
    if (!timeStr) return 0;

    try {
      const [h, m, s] = timeStr.split(':');
      // Handle cases where comma might be present or not
      const [sec, ms] = s.split(',');
      return parseInt(h, 10) * 3600 + parseInt(m, 10) * 60 + parseInt(sec, 10) + (ms ? parseInt(ms, 10) / 1000 : 0);
    } catch (e) {
      console.warn("Invalid time format:", timeStr);
      return 0;
    }
  };

  const PlayerState = () => {
    const [currentTime, setCurrentTime] = useState(0);
    const [videoRefreshKey, setVideoRefreshKey] = useState(Date.now());
    const videoRef = useRef(null);
    const savedPlaybackTime = useRef(0); // To persist time across source switches

    // Update refresh key when result changes (e.g. after regeneration)
    useEffect(() => {
      setVideoRefreshKey(Date.now());
    }, [result]);

    const [edits, setEdits] = useState({ source: {}, target: {} }); // Store uncommitted edits - moved up from TranscriptViewer
    const [localTranscripts, setLocalTranscripts] = useState(transcripts);
    const [isRegenerating, setIsRegenerating] = useState(false);
    const [deletedIndices, setDeletedIndices] = useState({});
    const [availableVoices, setAvailableVoices] = useState([]);
    const [speakerOverrides, setSpeakerOverrides] = useState({});
    const [showOriginal, setShowOriginal] = useState(false); // Audio toggle state
    const pollIntervalRef = useRef(null); // For cleanup

    // Cleanup polling on unmount
    useEffect(() => {
      return () => {
        if (pollIntervalRef.current) {
          clearInterval(pollIntervalRef.current);
        }
      };
    }, []);

    // Fetch voices on mount
    useEffect(() => {
      axios.get(`${API_BASE}/voices`).then(res => setAvailableVoices(res.data.voices)).catch(console.error);
    }, []);

    // Sync local transcripts when global transcripts change (e.g. after regeneration)
    useEffect(() => {
      setLocalTranscripts(transcripts);
      setEdits({ source: {}, target: {} }); // Clear edits on new load
    }, [transcripts]);

    // Handler for edits coming from TranscriptViewer
    const handleEdit = (type, index, value) => {
      setEdits(prev => ({
        ...prev,
        [type]: {
          ...prev[type],
          [index]: value
        }
      }));
    };

    const handleTranscriptUpdate = (type, index, value) => {
      const newTranscripts = { ...localTranscripts };

      if (type === 'timestamp') {
        // Handle timestamp edits
        const { field, value: timeValue } = value;
        newTranscripts.source[index] = { ...newTranscripts.source[index], [field]: timeValue };
        if (newTranscripts.target[index]) {
          newTranscripts.target[index] = { ...newTranscripts.target[index], [field]: timeValue };
        }
      } else if (type === 'source') {
        newTranscripts.source[index] = { ...newTranscripts.source[index], text: value };
      } else if (type === 'target') {
        newTranscripts.target[index] = { ...newTranscripts.target[index], text: value };
      }

      setLocalTranscripts(newTranscripts);
    };

    const handleDelete = (index) => {
      setDeletedIndices(prev => ({
        ...prev,
        [index]: !prev[index] // Toggle delete state
      }));
    };

    // Seek video to segment time
    const handleSegmentClick = (segment) => {
      if (videoRef.current) {
        const startTime = parseTime(segment.start);
        videoRef.current.currentTime = startTime;
        videoRef.current.play();
      }
    };

    // Merge segments - handle in parent component
    const handleMerge = (index) => {
      if (index >= localTranscripts.source.length - 1) {
        showToast('Cannot merge last segment', 'error');
        return;
      }

      if (!window.confirm('Merge this segment with the next one? This will permanently remove the second segment.')) return;

      const current = localTranscripts.source[index];
      const next = localTranscripts.source[index + 1];

      // Combine text
      const mergedSourceText = (edits.source[index] !== undefined ? edits.source[index] : (current.text || '')) + ' ' + (edits.source[index + 1] !== undefined ? edits.source[index + 1] : (next.text || ''));
      const mergedTargetText = (edits.target[index] !== undefined ? edits.target[index] : (localTranscripts.target[index]?.text || '')) + ' ' + (edits.target[index + 1] !== undefined ? edits.target[index + 1] : (localTranscripts.target[index + 1]?.text || ''));

      // Create new segment arrays (Deep copy to avoid reference issues)
      const newSourceSegments = [...localTranscripts.source];
      const newTargetSegments = [...localTranscripts.target];

      // Update first segment
      newSourceSegments[index] = {
        ...current,
        text: mergedSourceText,
        end: next.end
      };

      if (newTargetSegments[index]) {
        newTargetSegments[index] = {
          ...newTargetSegments[index],
          text: mergedTargetText,
          end: next.end
        };
      }

      // Remove the second segment
      newSourceSegments.splice(index + 1, 1);
      newTargetSegments.splice(index + 1, 1);

      // Recursive check for deletedIndices shift
      // If we delete index+1, all indices > index+1 shift down by 1.
      // And we need to remove index+1 from deletedIndices if present.
      const newDeletedIndices = {};
      Object.keys(deletedIndices).forEach(key => {
        const k = parseInt(key);
        if (k === index + 1) return; // Drop the deleted one
        if (k > index + 1) {
          newDeletedIndices[k - 1] = deletedIndices[k];
        } else {
          newDeletedIndices[k] = deletedIndices[k];
        }
      });
      setDeletedIndices(newDeletedIndices);

      // Shift edits: indices > index shift down by 1
      const newEdits = { source: { ...edits.source }, target: { ...edits.target } };

      // Clean up merged indices
      delete newEdits.source[index];
      delete newEdits.source[index + 1];
      delete newEdits.target[index];
      delete newEdits.target[index + 1];

      // Shift remaining edits
      const shiftEdits = (editMap) => {
        const newMap = {};
        Object.keys(editMap).forEach(key => {
          const k = parseInt(key);
          if (k < index) newMap[k] = editMap[k]; // Before: Keep
          else if (k > index + 1) newMap[k - 1] = editMap[k]; // After: Shift down
        });
        return newMap;
      };

      setEdits({
        source: shiftEdits(edits.source),
        target: shiftEdits(edits.target)
      });

      setLocalTranscripts({
        source: newSourceSegments,
        target: newTargetSegments
      });
    };

    const handleAddSegment = (index, gapStart, gapEnd) => {
      // gapStart and gapEnd are in SECONDS (from calculation in Viewer) convert to MS?
      // Wait, parseTime returns Seconds.
      // The segments use HH:MM:SS,mmm string format.
      // We need to convert gapStart/End back to string format for initial state?
      // Or just use the formatting helper.

      // Helper to format seconds to HH:MM:SS,mmm
      const formatTime = (secs) => {
        const ms = Math.floor(secs * 1000);
        const date = new Date(ms);
        const h = Math.floor(ms / 3600000).toString().padStart(2, '0');
        const m = Math.floor((ms % 3600000) / 60000).toString().padStart(2, '0');
        const s = Math.floor((ms % 60000) / 1000).toString().padStart(2, '0');
        const mil = (ms % 1000).toString().padStart(3, '0');
        return `${h}:${m}:${s},${mil}`;
      };

      const newSegment = {
        start: formatTime(gapStart),
        end: formatTime(gapEnd),
        text: "New Segment",
        speaker: "Speaker 1"
      };

      const newTargetSegment = {
        start: formatTime(gapStart),
        end: formatTime(gapEnd),
        text: "Translation",
        speaker: "Speaker 1"
      };

      const newSource = [...localTranscripts.source];
      const newTarget = [...localTranscripts.target];

      // Insert at index + 1
      newSource.splice(index + 1, 0, newSegment);
      newTarget.splice(index + 1, 0, newTargetSegment);

      // Shift deleted indices
      const newDeletedIndices = {};
      Object.keys(deletedIndices).forEach(key => {
        const k = parseInt(key);
        if (k > index) {
          newDeletedIndices[k + 1] = deletedIndices[k];
        } else {
          newDeletedIndices[k] = deletedIndices[k];
        }
      });
      setDeletedIndices(newDeletedIndices);

      // Shift edits: indices > index shift up by 1
      const shiftEditsForAdd = (editMap) => {
        const newMap = {};
        Object.keys(editMap).forEach(key => {
          const k = parseInt(key);
          if (k <= index) newMap[k] = editMap[k];
          else newMap[k + 1] = editMap[k]; // Shift everything after gap
        });
        return newMap;
      };

      setEdits({
        source: shiftEditsForAdd(edits.source),
        target: shiftEditsForAdd(edits.target)
      });

      setLocalTranscripts({
        source: newSource,
        target: newTarget
      });
    };

    // Split segment logic
    const handleSplit = (index, wordIndex) => {
      const seg = localTranscripts.source[index];
      const targetSeg = localTranscripts.target[index];
      const words = (edits.source[index] !== undefined ? edits.source[index] : (seg.text || "")).split(/\s+/);

      if (words.length < 2) {
        showToast("Cannot split single word segment", "error");
        return;
      }

      // Calculate split point
      // wordIndex is the index of the FIRST word of the SECOND segment
      // e.g. "Hello World" -> click "World" (index 1) -> Seg 1: "Hello", Seg 2: "World"

      const text1 = words.slice(0, wordIndex).join(" ");
      const text2 = words.slice(wordIndex).join(" ");

      // Calculate time split
      const start = parseTime(seg.start);
      const end = parseTime(seg.end);
      const duration = end - start;

      // Rough estimate based on word count ratio
      const ratio = wordIndex / words.length;
      const splitTime = start + (duration * ratio);

      // Format time helper
      const formatTime = (secs) => {
        const ms = Math.floor(secs * 1000);
        const date = new Date(ms);
        const h = Math.floor(ms / 3600000).toString().padStart(2, '0');
        const m = Math.floor((ms % 3600000) / 60000).toString().padStart(2, '0');
        const s = Math.floor((ms % 60000) / 1000).toString().padStart(2, '0');
        const mil = (ms % 1000).toString().padStart(3, '0');
        return `${h}:${m}:${s},${mil}`;
      };

      const splitTimeStr = formatTime(splitTime);

      // Create new segments
      const newSeg1 = { ...seg, text: text1, end: splitTimeStr };
      const newSeg2 = { ...seg, text: text2, start: splitTimeStr }; // Inherit speaker etc

      const newTargetSeg1 = { ...targetSeg, text: "", end: splitTimeStr }; // Clear target? Or try to split? Clearing is safer.
      const newTargetSeg2 = { ...targetSeg, text: "", start: splitTimeStr };

      // Update state
      const newSource = [...localTranscripts.source];
      const newTarget = [...localTranscripts.target];

      newSource.splice(index, 1, newSeg1, newSeg2);
      newTarget.splice(index, 1, newTargetSeg1, newTargetSeg2);

      // Shift deleted indices
      // Everything > index shifts by +1
      const newDeletedIndices = {};
      Object.keys(deletedIndices).forEach(key => {
        const k = parseInt(key);
        if (k > index) {
          newDeletedIndices[k + 1] = deletedIndices[k];
        } else if (k < index) {
          newDeletedIndices[k] = deletedIndices[k];
        } else {
          // Index itself was split. If it was deleted, should both new ones be deleted?
          // Assuming we only split active segments usually.
          if (deletedIndices[k]) {
            newDeletedIndices[k] = true;
            newDeletedIndices[k + 1] = true;
          }
        }
      });
      setDeletedIndices(newDeletedIndices);

      // Shift edits: indices > index shift up by 1
      const shiftEditsUp = (editMap) => {
        const newMap = {};
        Object.keys(editMap).forEach(key => {
          const k = parseInt(key);
          if (k < index) newMap[k] = editMap[k];
          else if (k > index) newMap[k + 1] = editMap[k];
          // At index (k==index), we split it. The old edit is likely invalid for the new split segments.
          // So we drop edit[index].
        });
        return newMap;
      };

      setEdits({
        source: shiftEditsUp(edits.source),
        target: shiftEditsUp(edits.target)
      });

      setLocalTranscripts({ source: newSource, target: newTarget });
    };

    const handleRegenerate = async () => {
      if (!activeTask) return;
      setIsRegenerating(true);
      try {
        // Construct segments payload
        // We need to send list of {start, end, text(source), target_text, speaker}
        const segmentsPayload = localTranscripts.source.map((srcSeg, i) => {
          // Extract speaker from ORIGINAL source text (before edits) to preserve speaker info
          // Use the extractSpeaker helper that gets the LAST speaker tag
          const extractSpeaker = (text) => {
            if (!text) return null;
            const matches = text.match(/\[Speaker (\d+)\]/g);
            if (!matches || matches.length === 0) return null;
            const lastMatch = matches[matches.length - 1];
            const speakerNum = lastMatch.match(/\[Speaker (\d+)\]/);
            return speakerNum ? parseInt(speakerNum[1]) : null;
          };

          const speakerNum = extractSpeaker(srcSeg.text);
          const speaker = speakerNum ? `Speaker ${speakerNum}` : 'Speaker 1';

          const sourceText = edits.source[i] !== undefined ? edits.source[i] : srcSeg.text;

          return {
            start: srcSeg.start,
            end: srcSeg.end,
            text: sourceText,
            target_text: edits.target[i] !== undefined ? edits.target[i] : (localTranscripts.target[i]?.text || ""),
            deleted: !!deletedIndices[i], // Include deleted flag
            speaker: speaker // Include speaker for voice override lookup
          };
        });

        await axios.post(`${API_BASE}/regenerate`, {
          task_id: activeTask.id,
          segments: segmentsPayload,
          speaker_overrides: speakerOverrides
        });

        // Start polling again
        setTaskId(activeTask.id);
        setState('processing');
        setView('new'); // Switch to processing view

        // Poll until complete, then reload result
        // Use ref for cleanup
        pollIntervalRef.current = setInterval(async () => {
          try {
            const statusRes = await axios.get(`${API_BASE}/status/${activeTask.id}`);
            if (statusRes.data.status === 'completed') {
              clearInterval(pollIntervalRef.current);
              pollIntervalRef.current = null;
              // Reload the result to get updated video
              const resultRes = await axios.get(`${API_BASE}/result/${activeTask.id}`);
              setActiveTask(resultRes.data);
              setResult(resultRes.data); // Update result state to show new video
              setLocalTranscripts({
                source: resultRes.data.source_segments || [],
                target: resultRes.data.target_segments || []
              });
              // Persist speaker overrides from result if available
              if (resultRes.data.speaker_overrides) {
                setSpeakerOverrides(resultRes.data.speaker_overrides);
              }
              setView('player'); // Changed from 'result' to 'player' to stay on the player view
              setState('idle');
              setIsRegenerating(false);
            } else if (statusRes.data.status === 'failed') {
              clearInterval(pollIntervalRef.current);
              pollIntervalRef.current = null;
              showToast('Regeneration failed: ' + statusRes.data.message, 'error');
              setIsRegenerating(false);
              setState('idle');
            }
          } catch (err) {
            console.error('Polling error:', err);
          }
        }, 2000);

      } catch (err) {
        console.error("Regeneration failed", err);
        showToast("Failed to start regeneration", "error");
        setIsRegenerating(false);
      }
    };

    return (
      <div className="max-w-[95vw] mx-auto mt-6 px-6 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-10 h-[calc(100vh-100px)] flex flex-col">
        {/* Toolbar */}
        <div className="flex items-center justify-between mb-6 bg-white/80 backdrop-blur-md p-4 rounded-2xl shadow-sm border border-white/50">
          <div className="flex items-center gap-4">
            <button
              onClick={() => {
                setView('home');
                setResult(null);
                setActiveTask(null);
              }}
              className="p-2 hover:bg-slate-100 rounded-lg transition-colors text-slate-500"
            >
              <RefreshCw size={20} className="rotate-90" /> {/* Using as Back icon */}
            </button>
            <div>
              <h2 className="text-xl font-bold text-slate-900">{activeTask?.filename || "Dubbed Video"}</h2>
              <div className="flex items-center gap-2 text-xs font-bold text-slate-500">
                <span className="bg-orange-100 text-orange-700 px-2 py-0.5 rounded-md">{activeTask?.input_lang || settings.inputLang}</span>
                <span>to</span>
                <span className="bg-amber-100 text-amber-700 px-2 py-0.5 rounded-md">{activeTask?.output_lang || settings.outputLang}</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleRegenerate}
              disabled={isRegenerating}
              className="px-4 py-2 rounded-xl font-bold text-white bg-slate-900 hover:bg-slate-800 transition-colors shadow-lg shadow-slate-900/20 flex items-center gap-2 text-sm"
            >
              {isRegenerating ? <Loader2 size={18} className="animate-spin" /> : <RefreshCw size={18} />}
              Update Dub
            </button>

            <a
              href={`${SERVER_URL}${result.download_url}`}
              download
              className="px-4 py-2 rounded-xl font-bold text-white bg-orange-600 hover:bg-orange-700 transition-colors shadow-lg shadow-orange-500/20 flex items-center gap-2 text-sm"
            >
              <Download size={18} /> Download
            </a>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 flex-1 min-h-0">
          {/* Video Player (40%) */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            <div className="bg-black rounded-3xl overflow-hidden shadow-2xl aspect-video relative ring-4 ring-white/50 flex-shrink-0">
              <video
                ref={videoRef}
                src={`${SERVER_URL}${showOriginal ? result.original_video_url || result.download_url : result.download_url}?t=${videoRefreshKey}`}
                controls
                className="w-full h-full object-contain"
                onLoadStart={() => console.log("Video loading...")}
                onCanPlay={() => console.log("Video ready to play")}
                onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
                onLoadedMetadata={(e) => {
                  if (savedPlaybackTime.current > 0) {
                    e.target.currentTime = savedPlaybackTime.current;
                  }
                }}
                onError={(e) => {
                  console.error("Video Error:", e);
                  showToast("Error loading video.", "error");
                }}
              />
            </div>

            {/* Audio Toggle */}
            <div className="flex items-center justify-center gap-3 p-3 bg-white/90 backdrop-blur-sm rounded-2xl shadow-sm">
              <Volume2 size={18} className="text-slate-500" />
              <button
                onClick={() => {
                  if (videoRef.current) savedPlaybackTime.current = videoRef.current.currentTime;
                  setShowOriginal(false);
                }}
                className={`px-4 py-2 rounded-lg font-semibold transition-all ${!showOriginal
                  ? 'bg-orange-600 text-white shadow-md'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
              >
                Dubbed
              </button>
              <button
                onClick={() => {
                  if (videoRef.current) savedPlaybackTime.current = videoRef.current.currentTime;
                  setShowOriginal(true);
                }}
                className={`px-4 py-2 rounded-lg font-semibold transition-all ${showOriginal
                  ? 'bg-orange-600 text-white shadow-md'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
              >
                Original
              </button>
            </div>

            {/* Info Card */}
            <div className="bg-white/60 backdrop-blur-md rounded-2xl p-6 border border-white/50 flex-1">
              <h3 className="font-bold text-slate-900 mb-4 flex items-center gap-2">
                <Activity size={18} className="text-orange-500" /> Playback Info
              </h3>
              <div className="space-y-4 text-sm text-slate-600">
                <div className="flex justify-between">
                  <span>Current Time</span>
                  <span className="font-mono font-bold">{currentTime.toFixed(2)}s</span>
                </div>
                <div className="flex justify-between">
                  <span>Duration</span>
                  <span className="font-mono font-bold">{videoRef.current?.duration?.toFixed(2) || "0.00"}s</span>
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Transcript (60%) */}
          <div className="lg:col-span-3 bg-white/80 backdrop-blur-md rounded-3xl shadow-xl shadow-slate-200/50 border border-white/50 overflow-hidden flex flex-col relative">
            <div className="absolute top-0 left-0 right-0 h-12 bg-gradient-to-b from-white/90 to-transparent z-10 pointer-events-none" />

            <TranscriptViewer
              sourceSegments={localTranscripts.source}
              targetSegments={localTranscripts.target}
              currentTime={currentTime}
              onUpdate={handleTranscriptUpdate}
              onSegmentClick={handleSegmentClick}
              onMerge={handleMerge}
              onSplit={handleSplit}
              onAdd={handleAddSegment}
              onDelete={handleDelete}
              deletedIndices={deletedIndices}
              availableVoices={availableVoices}
              speakerOverrides={speakerOverrides}
              onVoiceOverride={(speaker, voice) => setSpeakerOverrides(prev => ({ ...prev, [speaker]: voice }))}
              inputLang={activeTask?.input_lang || settings.inputLang}
              outputLang={activeTask?.output_lang || settings.outputLang}
              edits={edits}
              onEdit={handleEdit}
            />

            <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-white/90 to-transparent z-10 pointer-events-none" />
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 selection:bg-orange-100 selection:text-orange-700 relative overflow-x-hidden">
      {/* Background Image */}
      <div
        className="fixed inset-0 z-0 opacity-100 pointer-events-none blur-sm"
        style={{
          backgroundImage: `url('/bg.png')`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      />

      {/* Content */}
      <div className="relative z-10">
        <Header />
        <main className="pb-20">
          {view === 'home' && <HomeState />}
          {view === 'new' && state === 'config' && <ConfigState />}
          {view === 'new' && state === 'processing' && <ProcessingState activeTask={activeTask} pipelineStatus={pipelineStatus} taskId={taskId} />}
          {view === 'player' && <PlayerState />}
        </main>

        {/* Resume Modal Overlay */}
        {resumeJobId && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl space-y-6 text-center animate-in zoom-in-95 duration-200 border border-white/50">
              <div className="w-20 h-20 bg-orange-50 text-orange-600 rounded-full flex items-center justify-center mx-auto shadow-sm">
                <RefreshCw size={40} />
              </div>
              <div>
                <h3 className="text-2xl font-black text-slate-900 tracking-tight">Resume Found</h3>
                <p className="text-slate-500 mt-2 font-medium">
                  We found an interrupted job for this file. Would you like to resume where it left off?
                </p>
              </div>
              <div className="flex flex-col gap-3">
                <button
                  onClick={handleResume}
                  className="w-full py-4 bg-orange-600 hover:bg-orange-700 text-white font-bold rounded-2xl transition-all shadow-xl shadow-orange-500/20 hover:scale-[1.02]"
                >
                  Resume Previous Job
                </button>
                <button
                  onClick={() => startDubbing(uploadedFilename)}
                  className="w-full py-4 bg-slate-100 hover:bg-slate-200 text-slate-500 font-bold rounded-2xl transition-colors hover:text-slate-700"
                >
                  Start Fresh
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Toast Notification */}
        {toast && (
          <div className="fixed bottom-6 right-6 z-[110] animate-in slide-in-from-bottom-5 fade-in duration-300">
            <div className={`px-6 py-4 rounded-xl shadow-2xl border flex items-center gap-3 ${toast.type === 'success' ? 'bg-green-600 text-white border-green-500' :
              toast.type === 'error' ? 'bg-red-600 text-white border-red-500' :
                'bg-slate-800 text-white border-slate-700'
              }`}>
              {toast.type === 'success' && <CheckCircle2 size={20} />}
              {toast.type === 'error' && <AlertCircle size={20} />}
              {toast.type === 'info' && <Activity size={20} />}
              <span className="font-bold">{toast.message}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const ProcessingState = ({ activeTask, pipelineStatus, taskId }) => {
  const [logs, setLogs] = useState([]);
  const [showLogs, setShowLogs] = useState(false);
  const logEndRef = useRef(null);

  useEffect(() => {
    const targetId = taskId || activeTask?.id;
    if (!targetId) return;

    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/logs/${targetId}`);
        setLogs(res.data.logs);
      } catch (e) {
        // console.error("Log fetch error", e);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [activeTask, taskId]);

  const shouldAutoScroll = useRef(true);

  useEffect(() => {
    if (showLogs && shouldAutoScroll.current) {
      logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, showLogs]);

  const handleLogScroll = (e) => {
    const { scrollTop, scrollHeight, clientHeight } = e.target;
    // If user is near bottom (within 50px), enable auto-scroll. Otherwise disable.
    const isAtBottom = scrollHeight - scrollTop - clientHeight < CONSTANTS.SCROLL_THRESHOLD;
    shouldAutoScroll.current = isAtBottom;
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 text-center space-y-8 animate-in fade-in duration-700 relative">
      {/* Logs Toggle Button - Moved to Top Right inside container */}
      <button
        onClick={() => setShowLogs(!showLogs)}
        className="absolute right-0 top-0 p-3 bg-slate-800 text-white rounded-xl shadow-lg hover:scale-105 transition-transform border border-slate-700 hover:bg-slate-700 group"
        title="Toggle Logs"
      >
        <Terminal size={20} className="group-hover:text-orange-400 transition-colors" />
      </button>

      <div className="relative w-32 h-32 mx-auto">
        <div className="absolute inset-0 bg-orange-500/20 rounded-full animate-ping" />
        <div className="absolute inset-0 bg-amber-500/20 rounded-full animate-pulse delay-75" />
        <div className="relative bg-white rounded-full w-full h-full flex items-center justify-center shadow-2xl shadow-orange-500/20 border-4 border-white">
          <Loader2 size={48} className="text-orange-600 animate-spin" strokeWidth={2.5} />
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-4xl font-black text-slate-900 tracking-tight">
          {pipelineStatus.step}
        </h2>
        <p className="text-lg text-slate-500 font-medium max-w-lg mx-auto leading-relaxed">
          {pipelineStatus.message}
        </p>

        <div className="max-w-md mx-auto h-2 bg-slate-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-orange-500 to-amber-500 transition-all duration-700 ease-out"
            style={{ width: `${pipelineStatus.progress}%` }}
          />
        </div>

      </div>

      {/* Step Indicators */}
      <div className="grid grid-cols-5 gap-4 text-xs font-bold bg-white/50 backdrop-blur-md p-4 rounded-2xl border border-white/40 shadow-sm mx-auto max-w-2xl">
        {['Upload', 'Transcribe', 'Translate', 'Synthesize', 'Muxing'].map((step, i) => {
          const currentStepIdx = ['Preprocessing', 'Transcribing', 'Translating & Dubbing', 'Finalizing', 'Muxing', 'Completed'].indexOf(pipelineStatus.step);
          const isActive = i <= currentStepIdx;

          return (
            <div key={step} className={`flex flex-col items-center gap-2 transition-colors duration-300 ${isActive ? 'text-orange-600 scale-105' : 'text-slate-300'}`}>
              <div className={`w-3 h-3 rounded-full border-2 ${isActive ? 'bg-orange-500 border-orange-500 shadow-md shadow-orange-500/30' : 'bg-transparent border-slate-300'}`} />
              {step}
            </div>
          )
        })}
      </div>

      {/* Logs Overlay */}
      {showLogs && (
        <div className="absolute top-0 left-0 w-full h-full min-h-[500px] z-50 bg-slate-900/95 backdrop-blur-xl rounded-3xl p-6 text-left border border-slate-700 shadow-2xl animate-in zoom-in-95 duration-200 flex flex-col">
          <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-4">
            <div className="flex items-center gap-3">
              <Terminal size={18} className="text-orange-500" />
              <span className="font-mono text-sm text-slate-400 font-bold">Execution Logs</span>
            </div>
            <button onClick={() => setShowLogs(false)} className="text-slate-500 hover:text-white transition-colors">
              <X size={20} />
            </button>
          </div>

          <div
            className="flex-1 overflow-y-auto font-mono text-xs space-y-2 pr-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent"
            onScroll={handleLogScroll}
          >
            {logs.length === 0 && <div className="text-slate-600 italic p-4 text-center">Initializing pipeline logs...</div>}

            {logs.map((log, i) => (
              <div key={i} className="break-all hover:bg-white/5 p-1 rounded transition-colors group">
                <span className="text-slate-600 mr-3 select-none text-[10px]">{log.substring(1, 9)}</span>
                <span className={`${log.includes("ERROR") ? "text-red-400 font-bold" : log.includes("Step") ? "text-orange-400 font-bold" : "text-green-400/80 group-hover:text-green-300"}`}>
                  {log.substring(11)}
                </span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
