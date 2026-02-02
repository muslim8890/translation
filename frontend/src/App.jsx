import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Upload, FileText, Settings, Activity, CheckCircle, AlertCircle, Play, Loader2, Globe, Cpu, RotateCw, History, Menu, X, ChevronDown, Zap, Eye, Download, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  // --- STATE ---
  const [file, setFile] = useState(null);
  const [targetLang, setTargetLang] = useState('Arabic');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState('النظام جاهز');
  const [errorText, setErrorText] = useState(null);
  const [clientId, setClientId] = useState(null);
  const [completedPages, setCompletedPages] = useState([]);
  const [finalPdf, setFinalPdf] = useState(null);
  const wsRef = useRef(null);
  const logsEndRef = useRef(null);

  // Settings
  const [translationMode, setTranslationMode] = useState('word');

  // --- CONFIG ---
  // If running in Browser (Web App), use relative path (same server).
  // If running in Mobile App (Capacitor), use your SERVER IP/DOMAIN here.
  // Example: const API_BASE = "https://my-translator-app.onrender.com";
  // For local testing on Emulator: "http://10.0.2.2:8000"
  // --- CONFIG ---
  const isMobile = window.Capacitor !== undefined;

  // Dynamic API Base
  const getApiBase = () => {
    if (isMobile) return "http://10.0.2.2:8000"; // Android Emulator
    if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") return "http://localhost:8000"; // Local Dev
    return ""; // Production (Relative path)
  };

  const API_BASE = getApiBase();

  // --- CONNECT SYSTEM ---
  useEffect(() => {
    const connectWS = () => {
      const newId = clientId || `temp-${Date.now()}`;

      // Construct WS URL dynamically
      let wsBase;
      if (API_BASE.startsWith('http')) {
        wsBase = API_BASE.replace('http', 'ws');
      } else {
        // Production: explicit wss:// if https, ws:// if http
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsBase = `${protocol}//${window.location.host}`;
      }

      const wsUrl = `${wsBase}/ws/${newId}`;
      const socket = new WebSocket(wsUrl);
      socket.onopen = () => console.log('WS Connected');
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.client_id) setClientId(data.client_id);
        if (data.type === 'progress') {
          // AUTO-RECOVERY: If we get progress, we are definitely running.
          // Clear any initial HTTP errors and ensure loading state.
          setErrorText(null);
          setLoading(true);
          setProgress((data.current / data.total) * 100);
        }
        if (data.type === 'log') {
          setLogs(prev => [...prev.slice(-20), { original: data.original, translated: data.translated }]); // Keep last 20
        }
        if (data.type === 'complete') {
          setLoading(false); setSuccess(true); setStatusText('اكتملت الترجمة');
          if (data.download_url) window.location.href = data.download_url;
        }
        if (data.type === 'error') {
          setLoading(false); setErrorText(data.message); setStatusText('حدث خطأ');
        }
        if (data.type === 'page_ready') {
          setCompletedPages(prev => [...prev, data]);
        }
        if (data.type === 'final_ready') {
          setFinalPdf(data);
          setLoading(false); setStatusText('تم الدمج');
        }
      };
      socket.onclose = () => { setTimeout(connectWS, 3000); };
      wsRef.current = socket;
    };
    connectWS();
    return () => wsRef.current?.close();
  }, [clientId]); // Reconnect if clientId changes (e.g., after initial connection)

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);

  const handleUpload = async () => {
    if (!file || !clientId) return;
    setLoading(true); setSuccess(false); setErrorText(null); setLogs([]); setProgress(0); setStatusText('جاري البدء...');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_lang', targetLang);
    formData.append('translation_mode', translationMode);

    // Default configs invisible in UI for simplicity as per image
    formData.append('text_color', '#FF0000');
    formData.append('placement', 'below');
    formData.append('api_key', 'OFFLINE');

    try {
      // Must call translate endpoint to start the background task
      const res = await axios.post(`${API_BASE}/translate/${clientId}`, formData);
    } catch (e) {
      // ONLY show error if we haven't started processing yet.
      // If progress > 0, it's likely just a client-side timeout while the server is still working via WS.
      if (progress === 0 && !success) {
        setLoading(false);
        setErrorText('Network Error');
      }
    }
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white font-['Cairo'] overflow-hidden selection:bg-emerald-500/30" dir="rtl">

      {/* HEADER */}
      <header className="h-20 flex items-center justify-between px-10 border-b border-white/5 bg-[#050505]">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
            <Globe className="w-6 h-6 text-emerald-500" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-wide">المترجم الذكي برو</h1>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${clientId ? 'bg-emerald-500' : 'bg-red-500 animate-pulse'}`}></span>
              <span className="text-[10px] text-gray-400">{clientId ? 'متصل: جاهز للعمل' : 'جاري الاتصال...'}</span>
            </div>
          </div>
        </div>

        <div className="px-4 py-2 rounded-full bg-[#111] border border-white/10 flex items-center gap-2 text-xs text-gray-400">
          <Cpu className="w-3 h-3 text-purple-400" />
          النظام الهجين (Google + NLLB) مفعل
        </div>
      </header>

      {/* MAIN LAYOUT */}
      <main className="p-8 h-[calc(100vh-80px)] overflow-y-auto">
        <div className="max-w-7xl mx-auto grid grid-cols-12 gap-8 h-full">

          {/* MAIN COLUMN (RIGHT - 8 COLS) */}
          <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">

            {/* UPLOAD CARD */}
            <motion.div
              whileHover={{ scale: 1.002 }}
              className={`flex-1 min-h-[400px] border-2 border-dashed rounded-[3rem] relative flex flex-col items-center justify-center bg-[#0A0A0A] transition-colors group ${file ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-[#222] hover:border-[#333]'}`}
            >
              <input type="file" className="absolute inset-0 z-50 opacity-0 cursor-pointer" onChange={(e) => setFile(e.target.files[0])} accept=".pdf" disabled={loading} />

              <div className={`w-24 h-24 rounded-full flex items-center justify-center mb-6 transition-all ${file ? 'bg-emerald-500/20 text-emerald-500' : 'bg-[#151515] text-gray-600 group-hover:bg-[#202020]'}`}>
                {file ? <FileText className="w-10 h-10" /> : <Upload className="w-10 h-10" />}
              </div>

              {file ? (
                <div className="text-center">
                  <h3 className="text-2xl font-bold text-white mb-2" dir="ltr">{file.name}</h3>
                  <p className="text-emerald-500 font-mono text-sm">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              ) : (
                <div className="text-center">
                  <h3 className="text-2xl font-bold text-gray-200 mb-2">اسحب ملف PDF هنا</h3>
                  <p className="text-gray-500">أو انقر للاستعراض من جهازك</p>
                </div>
              )}
            </motion.div>

            {/* ACTION BUTTON */}
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className={`w-full py-6 rounded-[2rem] font-bold text-xl flex items-center justify-center gap-3 transition-all shadow-lg ${!file || loading ? 'bg-[#151515] text-gray-500 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-500/20 hover:shadow-emerald-500/40 hover:-translate-y-1'}`}
            >
              <Zap className={`w-6 h-6 ${loading ? 'animate-spin' : 'fill-current'}`} />
              {loading ? 'جاري المعالجة...' : 'ابدأ الترجمة الفورية'}
            </button>

            {/* DOWNLOAD RESULT (Conditional) */}
            <AnimatePresence>
              {finalPdf && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="p-6 rounded-[2rem] bg-[#111] border border-emerald-500/20 flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-500">
                      <CheckCircle className="w-6 h-6" />
                    </div>
                    <div>
                      <h3 className="font-bold text-white">الملف جاهز!</h3>
                      <p className="text-xs text-gray-400">تم دمج جميع الصفحات بنجاح</p>
                    </div>
                  </div>
                  <a href={finalPdf.url} target="_blank" className="px-6 py-3 rounded-xl bg-emerald-600 text-white font-bold flex items-center gap-2 hover:bg-emerald-500 transition-colors">
                    <Download className="w-5 h-5" />
                    تحميل PDF
                  </a>
                </motion.div>
              )}
            </AnimatePresence>

          </div>

          {/* SIDE COLUMN (LEFT - 4 COLS) */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-6 h-full">

            {/* 1. STATUS CARD */}
            <div className="bg-[#0A0A0A] rounded-[2.5rem] p-8 border border-[#1A1A1A] relative overflow-hidden h-64 flex flex-col justify-between">
              {errorText ? (
                <>
                  <div className="flex justify-between items-start">
                    <span className="px-3 py-1 rounded-full bg-red-500/10 text-red-500 text-xs font-bold border border-red-500/20">ERROR</span>
                    <Activity className="text-gray-600" />
                  </div>
                  <div className="text-center">
                    <h2 className="text-5xl font-black text-white mb-2">0%</h2>
                    <p className="text-red-400 text-sm font-bold" dir="ltr">{errorText}</p>
                  </div>
                  <div className="w-full bg-[#151515] h-2 rounded-full overflow-hidden">
                    <div className="h-full bg-red-500 w-full animate-pulse"></div>
                  </div>
                </>
              ) : (
                <>
                  <div className="flex justify-between items-start">
                    <span className={`px-3 py-1 rounded-full text-xs font-bold border ${loading ? 'bg-blue-500/10 text-blue-500 border-blue-500/20' : 'bg-gray-800 text-gray-500 border-gray-700'}`}>{loading ? 'PROCESSING' : 'IDLE'}</span>
                    <Activity className={`${loading ? 'text-blue-500 animate-pulse' : 'text-gray-600'}`} />
                  </div>
                  <div className="text-center relative z-10">
                    <h2 className={`text-6xl font-black mb-2 transition-colors ${loading ? 'text-blue-500' : 'text-white'}`}>{Math.round(progress)}%</h2>
                    <p className="text-gray-500 text-sm">{statusText}</p>
                  </div>
                  <div className="w-full bg-[#151515] h-3 rounded-full overflow-hidden relative">
                    <motion.div className="h-full bg-emerald-500 shadow-[0_0_15px_#10b981]" animate={{ width: `${progress}%` }} transition={{ type: "spring" }} />
                  </div>
                </>
              )}
            </div>

            {/* 2. SETTINGS CARD */}
            <div className="bg-[#0A0A0A] rounded-[2.5rem] p-8 border border-[#1A1A1A]">
              <div className="flex items-center gap-3 mb-6 text-gray-400">
                <Settings className="w-5 h-5 animate-[spin_10s_linear_infinite]" />
                <h3 className="font-bold">إعدادات دقيقة</h3>
              </div>

              <div className="space-y-5">
                <div className="flex items-center justify-between">
                  <label className="text-sm text-gray-500 font-medium">نوع الترجمة</label>
                  <div className="flex bg-[#111] p-1 rounded-lg border border-[#222]">
                    <button onClick={() => setTranslationMode('word')} className={`px-3 py-1.5 rounded-md text-xs transition-all ${translationMode === 'word' ? 'bg-[#222] text-gray-300' : 'text-gray-600'}`}>كلمات</button>
                    <button onClick={() => setTranslationMode('line')} className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all ${translationMode === 'line' ? 'bg-emerald-600 text-white shadow-lg' : 'text-gray-600'}`}>جمل (الموصى به)</button>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <label className="text-sm text-gray-500 font-medium">اللغة الهدف</label>
                  <div className="relative">
                    <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className="appearance-none bg-[#111] border border-[#222] text-gray-300 text-xs font-bold rounded-lg px-4 py-2 pr-8 focus:outline-none focus:border-emerald-500">
                      <option value="Arabic">العربية</option>
                      <option value="English">English</option>
                    </select>
                    <ChevronDown className="w-3 h-3 text-gray-500 absolute left-2 top-3 pointer-events-none" />
                  </div>
                </div>
              </div>
            </div>

            {/* 3. LOGS CARD */}
            <div className="bg-[#0A0A0A] rounded-[2.5rem] p-8 border border-[#1A1A1A] flex-1 overflow-hidden flex flex-col min-h-[150px]">
              <div className="flex items-center gap-3 mb-4 text-gray-400">
                <History className="w-4 h-4" />
                <h3 className="font-bold text-sm">سجل النظام</h3>
              </div>
              <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-hide text-xs font-mono" dir="ltr">
                {logs.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-[#222]">No Activity</div>
                ) : (
                  logs.map((log, i) => (
                    <div key={i} className="pb-2 border-b border-[#151515] last:border-0">
                      <span className="text-gray-600 block mb-1 opacity-50">{new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
                      <div className="text-emerald-500/80 truncate">Done: {log.translated.substring(0, 30)}...</div>
                    </div>
                  ))
                )}
                <div ref={logsEndRef} />
              </div>
            </div>

          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
