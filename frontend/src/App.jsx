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

      // FIX: Connect to plain /ws. Backend handles ID generation.
      const wsUrl = `${wsBase}/ws`;
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
  }, []); // FIX: Empty dependency array prevents infinite loop on clientId change

  useEffect(() => { logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);

  // --- DRAWER STATE ---
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

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
    <div className="h-screen bg-[#050505] text-white font-['Cairo'] overflow-hidden selection:bg-emerald-500/30 flex flex-col" dir="rtl">

      {/* HEADER */}
      <header className="h-16 flex-none flex items-center justify-between px-6 border-b border-white/5 bg-[#050505] z-30">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsDrawerOpen(true)}
            className="lg:hidden p-2 rounded-xl hover:bg-white/5 transition-colors"
          >
            <Menu className="w-6 h-6 text-white" />
          </button>

          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
            <Globe className="w-5 h-5 text-emerald-500" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-wide">المترجم الذكي برو</h1>
            <div className="flex items-center gap-2">
              <span className={`w-1.5 h-1.5 rounded-full ${clientId ? 'bg-emerald-500' : 'bg-red-500 animate-pulse'}`}></span>
              <span className="text-[10px] text-gray-400">{clientId ? 'متصل' : 'جاري الاتصال...'}</span>
            </div>
          </div>
        </div>

        <div className="hidden md:flex px-3 py-1.5 rounded-full bg-[#111] border border-white/10 items-center gap-2 text-[10px] text-gray-400">
          <Cpu className="w-3 h-3 text-purple-400" />
          النظام الهجين (Google + NLLB)
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <div className="flex-1 flex overflow-hidden relative">

        {/* --- MOBILE OVEERLAY --- */}
        <AnimatePresence>
          {isDrawerOpen && (
            <>
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                onClick={() => setIsDrawerOpen(false)}
                className="absolute inset-0 bg-black/80 backdrop-blur-sm z-40 lg:hidden"
              />
              <motion.aside
                initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }} transition={{ type: "spring", damping: 25, stiffness: 200 }}
                className="absolute right-0 top-0 bottom-0 w-[85%] max-w-[320px] bg-[#0A0A0A] border-l border-white/10 z-50 p-6 flex flex-col gap-6 shadow-2xl"
              >
                <div className="flex items-center justify-between mb-2">
                  <h2 className="font-bold text-lg">لوحة التحكم</h2>
                  <button onClick={() => setIsDrawerOpen(false)}><X className="w-6 h-6 text-gray-400" /></button>
                </div>
                <SidebarContent
                  progress={progress} statusText={statusText} errorText={errorText} loading={loading}
                  translationMode={translationMode} setTranslationMode={setTranslationMode}
                  targetLang={targetLang} setTargetLang={setTargetLang}
                  logs={logs} logsEndRef={logsEndRef}
                  isMobile={true}
                />
              </motion.aside>
            </>
          )}
        </AnimatePresence>


        {/* --- LEFT SIDEBAR (DESKTOP) --- */}
        <aside className="hidden lg:flex w-[350px] bg-[#080808] border-l border-white/5 flex-col p-6 gap-6 z-20 shadow-xl">
          <SidebarContent
            progress={progress} statusText={statusText} errorText={errorText} loading={loading}
            translationMode={translationMode} setTranslationMode={setTranslationMode}
            targetLang={targetLang} setTargetLang={setTargetLang}
            logs={logs} logsEndRef={logsEndRef}
            isMobile={false}
          />
        </aside>

        {/* --- MAIN CONTENT (CENTER) --- */}
        <main className="flex-1 bg-[#050505] relative flex flex-col items-center justify-center p-4 md:p-8">
          <div className="w-full max-w-2xl flex flex-col gap-6 h-full justify-center">

            {/* UPLOAD AREA (Compact) */}
            <motion.div
              whileHover={!loading && !file ? { scale: 1.01 } : {}}
              className={`flex-1 max-h-[400px] border-2 border-dashed rounded-[2rem] relative flex flex-col items-center justify-center transition-all group ${file ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-[#222] bg-[#0A0A0A] hover:border-[#333]'}`}
            >
              <input
                type="file"
                className="absolute inset-0 z-50 opacity-0 cursor-pointer"
                onChange={(e) => setFile(e.target.files[0])}
                accept=".pdf"
                disabled={loading}
              />

              <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-4 transition-all ${file ? 'bg-emerald-500/20 text-emerald-500' : 'bg-[#151515] text-gray-600 group-hover:bg-[#202020]'}`}>
                {file ? <FileText className="w-8 h-8" /> : <Upload className="w-8 h-8" />}
              </div>

              {file ? (
                <div className="text-center px-6">
                  <h3 className="text-lg font-bold text-white mb-1 dir-ltr truncate max-w-[250px]">{file.name}</h3>
                  <p className="text-emerald-500 font-mono text-xs">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              ) : (
                <div className="text-center">
                  <h3 className="text-lg font-bold text-gray-200 mb-1">اسحب ملف PDF هنا</h3>
                  <p className="text-gray-500 text-xs">أو انقر للاستعراض من جهازك</p>
                </div>
              )}
            </motion.div>

            {/* ACTION BUTTON */}
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className={`w-full py-4 rounded-2xl font-bold text-lg flex items-center justify-center gap-3 transition-all shadow-lg ${!file || loading ? 'bg-[#151515] text-gray-600 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-500/20 hover:-translate-y-0.5'}`}
            >
              <Zap className={`w-5 h-5 ${loading ? 'animate-spin' : 'fill-current'}`} />
              {loading ? 'جاري المعالجة...' : 'ابدأ الترجمة الفورية'}
            </button>

            {/* DOWNLOAD RESULT (Compact) */}
            <AnimatePresence>
              {finalPdf && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="p-4 rounded-2xl bg-[#111] border border-emerald-500/20 flex items-center justify-between shadow-lg shadow-emerald-900/10">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-500">
                      <CheckCircle className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="font-bold text-white text-sm">تم الانتهاء!</h3>
                      <p className="text-[10px] text-gray-400">جاهز للتحميل</p>
                    </div>
                  </div>
                  <a href={finalPdf.url} target="_blank" className="px-5 py-2 rounded-xl bg-emerald-600 text-white text-sm font-bold flex items-center gap-2 hover:bg-emerald-500 transition-colors">
                    <Download className="w-4 h-4" />
                    تحميل
                  </a>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
}

// --- SUB-COMPONENT: SIDEBAR CONTENT (Reused for Desktop & Mobile) ---
function SidebarContent({ progress, statusText, errorText, loading, translationMode, setTranslationMode, targetLang, setTargetLang, logs, logsEndRef, isMobile }) {
  return (
    <>
      {/* 1. STATUS CARD */}
      <div className={`bg-[#0A0A0A] rounded-[2rem] p-6 border border-[#1A1A1A] relative overflow-hidden flex flex-col justify-between shrink-0 ${isMobile ? 'h-48' : 'h-64'}`}>
        {errorText ? (
          <>
            <div className="flex justify-between items-start">
              <span className="px-2 py-0.5 rounded-full bg-red-500/10 text-red-500 text-[10px] font-bold border border-red-500/20">ERROR</span>
              <AlertCircle className="text-red-500 w-5 h-5" />
            </div>
            <div className="text-center mt-2">
              <p className="text-red-400 text-xs font-bold break-words" dir="ltr">{errorText}</p>
            </div>
            <div className="w-full bg-[#151515] h-1.5 rounded-full overflow-hidden mt-auto">
              <div className="h-full bg-red-500 w-full animate-pulse"></div>
            </div>
          </>
        ) : (
          <>
            <div className="flex justify-between items-start">
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${loading ? 'bg-blue-500/10 text-blue-500 border-blue-500/20' : 'bg-gray-800 text-gray-500 border-gray-700'}`}>{loading ? 'RUNNING' : 'IDLE'}</span>
              <Activity className={`w-5 h-5 ${loading ? 'text-blue-500 animate-pulse' : 'text-gray-600'}`} />
            </div>
            <div className="text-center relative z-10 my-auto">
              <h2 className={`text-5xl font-black mb-1 transition-colors ${loading ? 'text-blue-500' : 'text-white'}`}>{Math.round(progress)}%</h2>
              <p className="text-gray-500 text-xs">{statusText}</p>
            </div>
            <div className="w-full bg-[#151515] h-2 rounded-full overflow-hidden relative mt-auto">
              <motion.div className="h-full bg-emerald-500 shadow-[0_0_10px_#10b981]" animate={{ width: `${progress}%` }} transition={{ type: "spring" }} />
            </div>
          </>
        )}
      </div>

      {/* 2. SETTINGS CARD */}
      <div className="bg-[#0A0A0A] rounded-[2rem] p-6 border border-[#1A1A1A] shrink-0">
        <div className="flex items-center gap-2 mb-4 text-gray-400">
          <Settings className="w-4 h-4 animate-[spin_10s_linear_infinite]" />
          <h3 className="font-bold text-sm">إعدادات سريعة</h3>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-xs text-gray-500 font-medium">النمط</label>
            <div className="flex bg-[#111] p-1 rounded-lg border border-[#222]">
              <button onClick={() => setTranslationMode('word')} className={`px-2 py-1 rounded-md text-[10px] transition-all ${translationMode === 'word' ? 'bg-[#222] text-gray-300' : 'text-gray-600'}`}>كلمات</button>
              <button onClick={() => setTranslationMode('line')} className={`px-2 py-1 rounded-md text-[10px] font-bold transition-all ${translationMode === 'line' ? 'bg-emerald-600 text-white shadow-lg' : 'text-gray-600'}`}>جمل (Auto)</button>
            </div>
          </div>
        </div>
      </div>

      {/* 3. LOGS (Fill remaining space) */}
      <div className="bg-[#0A0A0A] rounded-[2rem] p-6 border border-[#1A1A1A] flex-1 overflow-hidden flex flex-col min-h-[120px]">
        <div className="flex items-center gap-2 mb-3 text-gray-400">
          <History className="w-4 h-4" />
          <h3 className="font-bold text-sm">السجل الحي</h3>
        </div>
        <div className="flex-1 overflow-y-auto space-y-2 pr-1 scrollbar-hide text-[10px] font-mono leading-relaxed" dir="ltr">
          {logs.length === 0 ? (
            <div className="h-full flex items-center justify-center text-[#222]">No Activity</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="pb-1.5 border-b border-[#151515] last:border-0">
                <span className="text-gray-600 inline-block mr-2 opacity-50">{new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                <span className="text-emerald-500/90">{log.translated.substring(0, 40)}{log.translated.length > 40 ? '...' : ''}</span>
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
    </>
  );
}

export default App;
