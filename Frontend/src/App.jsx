
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Upload, FileVideo, Languages, Wand2, CheckCircle2, AlertCircle, Loader2, Play, Pause, Volume2, Activity, ArrowLeft, Mic, Clock, FileAudio, RefreshCw, Download
} from 'lucide-react';

const SERVER_URL = "http://127.0.0.1:8002";
const API_BASE = `${SERVER_URL}/api`;

const LANGUAGES = [
  "English", "Hindi", "Assamese", "Punjabi", "Telugu",
  "Tamil", "Marathi", "Gujarati", "Kannada", "Malayalam", "Odia", "Bengali"
];

function App() {
  const [view, setView] = useState('home'); // home, new, player
  const [history, setHistory] = useState([]);
  const [activeTask, setActiveTask] = useState(null);

  const [state, setState] = useState('config'); // config, processing, success
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [settings, setSettings] = useState({
    inputLang: 'English',
    outputLang: 'Hindi',
    voice: 'Female',
    speed: 1.0,
    durationLimit: null
  });
  const [taskId, setTaskId] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState({
    step: 'Initializing',
    progress: 0,
    message: 'Ready'
  });
  const [result, setResult] = useState(null);
  const [transcripts, setTranscripts] = useState({ source: [], target: [] });

  // --- Effects ---
  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API_BASE}/history`);
      setHistory(res.data);
    } catch (err) {
      console.error("Failed to fetch history", err);
    }
  };

  // --- Handlers ---

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) setFile(droppedFile);
  };

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

      const filename = uploadRes.data.filename;

      // 2. Start Dubbing
      const dubRes = await axios.post(`${API_BASE}/dub`, {
        filename: filename,
        input_lang: settings.inputLang,
        output_lang: settings.outputLang,
        target_voice: settings.voice,
        speed: settings.speed,
        duration_limit: settings.durationLimit
      });

      setTaskId(dubRes.data.task_id);
      setState('processing');

    } catch (err) {
      console.error(err);
      alert("Error starting process. Check console.");
    }
  };

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
            clearInterval(interval);
            fetchResult(taskId);
          } else if (data.status === 'failed') {
            clearInterval(interval);
            alert(`Task Failed: ${data.error}`);
            setState('config');
          }
        } catch (err) {
          console.error("Polling error", err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [state, taskId]);

  const fetchResult = async (id) => {
    try {
      const res = await axios.get(`${API_BASE}/result/${id}`);
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

      setState('success');
      setView('player');
    } catch (err) {
      console.error(err);
      alert("Failed to fetch results.");
    }
  };

  // --- Components ---

  const Header = () => (
    <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-orange-100 shadow-sm">
      <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center shadow-lg shadow-orange-500/20 backdrop-blur-sm overflow-hidden">
            <img src="/logo.png" alt="Logo" className="w-full h-full object-cover" />
          </div>
          <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-orange-600 to-amber-600">
            DubFlow
          </span>
        </div>
        <div className="text-sm font-medium text-slate-500">
          Indic Video Dubbing
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
              onClick={() => loadTask(task.id)}
              className="bg-white/80 backdrop-blur-md p-5 rounded-2xl shadow-sm border border-orange-100 hover:shadow-xl hover:shadow-orange-500/10 hover:border-orange-200 transition-all cursor-pointer group"
            >
              <div className="aspect-video bg-slate-900 rounded-xl mb-4 relative overflow-hidden">
                <div className="absolute inset-0 flex items-center justify-center group-hover:scale-110 transition-transform duration-500">
                  <Play size={32} className="text-white/80" />
                </div>
                <div className="absolute bottom-2 right-2 px-2 py-1 bg-black/60 backdrop-blur-md rounded-lg text-xs font-bold text-white">
                  {task.output_lang}
                </div>
              </div>
              <h3 className="font-bold text-slate-900 truncate mb-1">{task.filename}</h3>
              <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
                <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-orange-500" /> {task.input_lang}</span>
                <span>→</span>
                <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-amber-500" /> {task.output_lang}</span>
              </div>
              <div className="mt-4 text-xs text-slate-400 font-medium">
                {new Date(task.timestamp).toLocaleDateString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const loadTask = async (id) => {
    try {
      const res = await axios.get(`${API_BASE}/result/${id}`);
      setResult(res.data);
      setTranscripts({
        source: res.data.source_segments,
        target: res.data.target_segments
      });
      setActiveTask(history.find(t => t.id === id));
      setView('player');
    } catch (err) {
      console.error(err);
      alert("Failed to load task.");
    }
  };

  const ConfigState = () => (
    <div className="max-w-3xl mx-auto mt-12 space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <button
        onClick={() => setView('home')}
        className="flex items-center gap-2 text-slate-500 hover:text-orange-600 transition-colors font-bold mb-4"
      >
        <ArrowLeft size={20} /> Back to Dashboard
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
              onChange={(e) => setSettings({ ...settings, inputLang: e.target.value })}
            >
              {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Languages size={16} className="text-orange-500" /> Target Language
            </label>
            <select
              className="w-full p-3 rounded-xl bg-slate-50 border-2 border-transparent focus:border-orange-500 focus:ring-0 text-slate-900 font-semibold transition-all"
              value={settings.outputLang}
              onChange={(e) => setSettings({ ...settings, outputLang: e.target.value })}
            >
              {LANGUAGES.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Mic size={16} className="text-orange-500" /> Voice Model
            </label>
            <select
              className="w-full p-3 rounded-xl bg-slate-50 border-2 border-transparent focus:border-orange-500 focus:ring-0 text-slate-900 font-semibold transition-all"
              value={settings.voice}
              onChange={(e) => setSettings({ ...settings, voice: e.target.value })}
            >
              <option value="Female">Female (Neural)</option>
              <option value="Male">Male (Neural)</option>
              <optgroup label="Chirp 3 (HD) - Female">
                <option value="Female (Chirp 3 - Aoede)">Aoede</option>
                <option value="Female (Chirp 3 - Kore)">Kore</option>
                <option value="Female (Chirp 3 - Leda)">Leda</option>
                <option value="Female (Chirp 3 - Zephyr)">Zephyr</option>
                <option value="Female (Chirp 3 - Erinome)">Erinome</option>
              </optgroup>
              <optgroup label="Chirp 3 (HD) - Male">
                <option value="Male (Chirp 3 - Puck)">Puck</option>
                <option value="Male (Chirp 3 - Charon)">Charon</option>
                <option value="Male (Chirp 3 - Fenrir)">Fenrir</option>
                <option value="Male (Chirp 3 - Orus)">Orus</option>
                <option value="Male (Chirp 3 - Achird)">Achird</option>
                <option value="Male (Chirp 3 - Alnilam)">Alnilam</option>
              </optgroup>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-bold text-slate-700 flex items-center gap-2">
              <Activity size={16} className="text-orange-500" /> Speaking Rate ({settings.speed}x)
            </label>
            <input
              type="range"
              min="0.85" max="1.15" step="0.05"
              className="w-full accent-orange-600 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer"
              value={settings.speed}
              onChange={(e) => setSettings({ ...settings, speed: parseFloat(e.target.value) })}
            />
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
              onChange={(e) => setSettings({ ...settings, durationLimit: e.target.value ? parseInt(e.target.value) : null })}
            />
          </div>
        </div>

        <button
          onClick={handleUpload}
          disabled={!file}
          className={`
            w-full mt-8 py-4 rounded-2xl font-bold text-lg flex items-center justify-center gap-3
            transition-all duration-300 transform
            ${file
              ? 'bg-gradient-to-r from-orange-600 to-amber-500 text-white shadow-xl shadow-orange-500/30 hover:shadow-orange-500/50 hover:scale-[1.02] active:scale-[0.98]'
              : 'bg-slate-100 text-slate-400 cursor-not-allowed'}
          `}
        >
          <Wand2 size={24} />
          {file ? 'Dub Video Now' : 'Select a file to start'}
        </button>
      </div>
    </div>
  );

  const ProcessingState = () => (
    <div className="max-w-2xl mx-auto mt-20 text-center space-y-12 animate-in fade-in duration-700">
      <div className="relative w-48 h-48 mx-auto">
        <div className="absolute inset-0 bg-orange-500/20 rounded-full animate-ping" />
        <div className="absolute inset-0 bg-amber-500/20 rounded-full animate-pulse delay-75" />
        <div className="relative bg-white rounded-full w-full h-full flex items-center justify-center shadow-2xl shadow-orange-500/20 border-4 border-white">
          <Activity size={64} className="text-orange-600 animate-bounce" />
        </div>
      </div>

      <div className="space-y-4 bg-white/80 backdrop-blur-md p-8 rounded-3xl shadow-xl border border-white/50">
        <h2 className="text-3xl font-black text-slate-900">{pipelineStatus.step}</h2>
        <p className="text-slate-600 text-lg font-medium">{pipelineStatus.message}</p>

        <div className="w-full bg-slate-100 rounded-full h-4 overflow-hidden shadow-inner">
          <div
            className="bg-gradient-to-r from-orange-500 to-amber-500 h-full transition-all duration-500 ease-out shadow-lg shadow-orange-500/30"
            style={{ width: `${pipelineStatus.progress}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-5 gap-4 text-xs font-bold text-slate-400">
        {['Upload', 'Transcribe', 'Translate', 'Synthesize', 'Muxing'].map((step, i) => {
          const currentStepIdx = ['Preprocessing', 'Transcribing', 'Translating & Dubbing', 'Finalizing', 'Muxing', 'Completed'].indexOf(pipelineStatus.step);
          const isActive = i <= currentStepIdx;

          return (
            <div key={step} className={`flex flex-col items-center gap-2 transition-colors duration-300 ${isActive ? 'text-orange-600' : ''}`}>
              <div className={`w-4 h-4 rounded-full border-2 ${isActive ? 'bg-orange-600 border-orange-600' : 'bg-slate-100 border-slate-300'}`} />
              {step}
            </div>
          )
        })}
      </div>
    </div>
  );

  const TranscriptViewer = ({ sourceSegments, targetSegments, currentTime, onUpdate }) => {
    const containerRef = useRef(null);
    const activeRef = useRef(null);
    const [edits, setEdits] = useState({ source: {}, target: {} });

    useEffect(() => {
      if (activeRef.current && containerRef.current) {
        activeRef.current.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        });
      }
    }, [currentTime]);

    const handleEdit = (type, index, value) => {
      setEdits(prev => ({
        ...prev,
        [type]: { ...prev[type], [index]: value }
      }));
      onUpdate(type, index, value);
    };

    return (
      <div ref={containerRef} className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
        {sourceSegments.map((seg, i) => {
          const start = parseTime(seg.start);
          const end = parseTime(seg.end);
          const isActive = currentTime >= start && currentTime <= end;

          const sourceText = edits.source[i] !== undefined ? edits.source[i] : seg.text;
          const targetText = edits.target[i] !== undefined ? edits.target[i] : (targetSegments[i]?.text || "");

          return (
            <div
              key={i}
              ref={isActive ? activeRef : null}
              className={`
                grid grid-cols-2 gap-8 p-6 rounded-2xl transition-all duration-500
                ${isActive
                  ? 'bg-orange-50 scale-105 shadow-lg shadow-orange-500/10 border-orange-200'
                  : 'hover:bg-slate-50 opacity-60 hover:opacity-100'}
              `}
            >
              <div className="space-y-2">
                <textarea
                  className={`w-full bg-transparent border-none resize-none focus:ring-0 p-0 text-lg font-medium leading-relaxed ${isActive ? 'text-slate-900' : 'text-slate-500'}`}
                  value={sourceText}
                  onChange={(e) => handleEdit('source', i, e.target.value)}
                  rows={Math.max(2, Math.ceil(sourceText.length / 40))}
                />
              </div>
              <div className="space-y-2">
                <textarea
                  className={`w-full bg-transparent border-none resize-none focus:ring-0 p-0 text-lg font-medium leading-relaxed ${isActive ? 'text-orange-700' : 'text-slate-400'}`}
                  value={targetText}
                  onChange={(e) => handleEdit('target', i, e.target.value)}
                  rows={Math.max(2, Math.ceil(targetText.length / 40))}
                  dir="auto"
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const parseTime = (timeStr) => {
    if (typeof timeStr === 'number') return timeStr / 1000;
    // Format: 00:00:00,000
    if (!timeStr) return 0;
    const [h, m, s] = timeStr.split(':');
    const [sec, ms] = s.split(',');
    return parseInt(h) * 3600 + parseInt(m) * 60 + parseInt(sec) + parseInt(ms) / 1000;
  };

  const PlayerState = () => {
    const [currentTime, setCurrentTime] = useState(0);
    const videoRef = useRef(null);
    const [localTranscripts, setLocalTranscripts] = useState(transcripts);
    const [isRegenerating, setIsRegenerating] = useState(false);

    // Sync local transcripts when global transcripts change (e.g. after regeneration)
    useEffect(() => {
      setLocalTranscripts(transcripts);
    }, [transcripts]);

    const handleTranscriptUpdate = (type, index, value) => {
      const newTranscripts = { ...localTranscripts };
      if (type === 'source') {
        newTranscripts.source[index] = { ...newTranscripts.source[index], text: value };
      } else {
        newTranscripts.target[index] = { ...newTranscripts.target[index], text: value };
      }
      setLocalTranscripts(newTranscripts);
    };

    const handleRegenerate = async () => {
      if (!activeTask) return;
      setIsRegenerating(true);
      try {
        // Construct segments payload
        // We need to send list of {start, end, text (source), target_text}
        const segmentsPayload = localTranscripts.source.map((srcSeg, i) => ({
          start: srcSeg.start,
          end: srcSeg.end,
          text: srcSeg.text,
          target_text: localTranscripts.target[i]?.text || ""
        }));

        await axios.post(`${API_BASE}/regenerate`, {
          task_id: activeTask.id,
          segments: segmentsPayload
        });

        // Start polling again
        setTaskId(activeTask.id);
        setState('processing');
        setView('new'); // Switch to processing view

      } catch (err) {
        console.error("Regeneration failed", err);
        alert("Failed to start regeneration");
        setIsRegenerating(false);
      }
    };

    return (
      <div className="max-w-7xl mx-auto mt-6 px-6 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-10 h-[calc(100vh-100px)] flex flex-col">
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
              href={`${API_BASE}${result.download_url}`}
              download
              className="px-4 py-2 rounded-xl font-bold text-white bg-orange-600 hover:bg-orange-700 transition-colors shadow-lg shadow-orange-500/20 flex items-center gap-2 text-sm"
            >
              <Download size={18} /> Download
            </a>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
          {/* Video Player */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            <div className="bg-black rounded-3xl overflow-hidden shadow-2xl aspect-video relative ring-4 ring-white/50 flex-shrink-0">
              <video
                ref={videoRef}
                src={`${SERVER_URL}${result.download_url}`}
                controls
                className="w-full h-full object-contain"
                onTimeUpdate={(e) => setCurrentTime(e.target.currentTime)}
                onError={(e) => {
                  console.error("Video Error:", e);
                  alert("Error loading video.");
                }}
              />
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

          {/* Interactive Transcript */}
          <div className="lg:col-span-1 bg-white/80 backdrop-blur-md rounded-3xl shadow-xl shadow-slate-200/50 border border-white/50 overflow-hidden flex flex-col relative">
            <div className="absolute top-0 left-0 right-0 h-12 bg-gradient-to-b from-white/90 to-transparent z-10 pointer-events-none" />

            <TranscriptViewer
              sourceSegments={localTranscripts.source}
              targetSegments={localTranscripts.target}
              currentTime={currentTime}
              onUpdate={handleTranscriptUpdate}
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
      {/* Background Image */}
      <div
        className="fixed inset-0 z-0 opacity-100 pointer-events-none blur-xl" // <--- Add blur-xl here
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
          {view === 'new' && state === 'processing' && <ProcessingState />}
          {view === 'player' && <PlayerState />}
        </main>
      </div>
    </div>
  );
}

export default App;
