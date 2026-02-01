import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Settings, Download, ZoomIn, ZoomOut, ArrowLeft, ArrowRight, Save, Layout, Type, AlignLeft, AlignCenter, AlignRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Editor = ({ file, clientId, apiKey, targetLang, onClose, onExport }) => {
    const [loading, setLoading] = useState(true);
    const [pageData, setPageData] = useState(null);
    const [pageNum, setPageNum] = useState(0);
    const [totalPages, setTotalPages] = useState(1);

    // Controls
    const [fsScale, setFsScale] = useState(1.0);
    const [yOffset, setYOffset] = useState(0);
    const [alignment, setAlignment] = useState('auto');
    const [viewZoom, setViewZoom] = useState(0.5); // Default zoom level (0.5 because image is 2x)
    const [isSidebarOpen, setSidebarOpen] = useState(true);

    useEffect(() => {
        uploadForPreview();
    }, []);

    useEffect(() => {
        if (!loading) fetchPage(pageNum);
    }, [pageNum]);

    const uploadForPreview = async () => {
        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        try {
            const res = await axios.post(`http://localhost:8080/upload_temp/${clientId}`, formData);
            setTotalPages(res.data.pages);
            fetchPage(0);
        } catch (err) {
            alert("Upload failed: " + err.message);
            onClose();
        }
    };

    const fetchPage = async (idx) => {
        setLoading(true);
        const formData = new FormData();
        formData.append('api_key', apiKey);
        formData.append('target_lang', targetLang);
        try {
            const res = await axios.post(`http://localhost:8080/preview_page/${clientId}/${idx}`, formData);
            setPageData(res.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleExport = () => {
        onExport({ fs_scale: fsScale, y_offset: yOffset, alignment: alignment });
    };

    return (
        <div className="fixed inset-0 z-50 bg-[#020617] flex flex-col">
            {/* Header */}
            <div className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-[#0f172a]">
                <div className="flex items-center gap-4">
                    <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-lg text-slate-400">
                        <ArrowLeft />
                    </button>
                    <button onClick={() => setSidebarOpen(!isSidebarOpen)} className={`p-2 rounded-lg transition-colors ${isSidebarOpen ? 'bg-blue-600 text-white' : 'text-slate-400 hover:bg-white/5'}`}>
                        <Settings size={20} />
                    </button>
                    <h2 className="font-bold text-white uppercase tracking-widest hidden md:block">Live Editor <span className="text-blue-500 text-xs">Beta</span></h2>
                </div>

                <div className="flex items-center gap-4">
                    {/* Zoom Controls */}
                    <div className="flex items-center gap-2 bg-black/20 p-1 rounded-lg mr-4">
                        <button onClick={() => setViewZoom(z => Math.max(0.2, z - 0.1))} className="p-2 hover:text-white text-slate-400"><ZoomOut size={16} /></button>
                        <span className="text-xs font-mono w-12 text-center text-slate-400">{Math.round(viewZoom * 100)}%</span>
                        <button onClick={() => setViewZoom(z => Math.min(2.0, z + 0.1))} className="p-2 hover:text-white text-slate-400"><ZoomIn size={16} /></button>
                    </div>

                    <div className="flex items-center gap-1 bg-black/20 p-1 rounded-lg mr-4">
                        <button onClick={() => setAlignment('left')} className={`p-2 rounded-lg ${alignment === 'left' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}><AlignLeft size={16} /></button>
                        <button onClick={() => setAlignment('center')} className={`p-2 rounded-lg ${alignment === 'center' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}><AlignCenter size={16} /></button>
                        <button onClick={() => setAlignment('right')} className={`p-2 rounded-lg ${alignment === 'right' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}><AlignRight size={16} /></button>
                        <button onClick={() => setAlignment('auto')} className={`px-2 text-[10px] font-bold uppercase rounded-lg ${alignment === 'auto' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}>Auto</button>
                    </div>

                    <div className="flex items-center gap-2 bg-black/20 p-1 rounded-lg">
                        <button disabled={pageNum <= 0} onClick={() => setPageNum(p => p - 1)} className="p-2 disabled:opacity-30 hover:text-white text-slate-400"><ArrowLeft size={16} /></button>
                        <span className="text-xs font-mono w-16 text-center text-slate-400">Page {pageNum + 1}/{totalPages}</span>
                        <button disabled={pageNum >= totalPages - 1} onClick={() => setPageNum(p => p + 1)} className="p-2 disabled:opacity-30 hover:text-white text-slate-400"><ArrowRight size={16} /></button>
                    </div>

                    <button onClick={handleExport} className="flex items-center gap-2 px-6 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg font-bold text-sm uppercase tracking-widest transition-all">
                        <Save size={16} /> Export
                    </button>
                </div>
            </div>

            <div className="flex-1 flex overflow-hidden">
                {/* Sidebar Controls */}
                <AnimatePresence>
                    {isSidebarOpen && (
                        <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 320, opacity: 1 }} exit={{ width: 0, opacity: 0 }} className="border-r border-white/10 bg-[#0f172a]/50 flex flex-col overflow-hidden whitespace-nowrap">
                            <div className="p-6 flex flex-col gap-8 w-80">
                                <div className="space-y-4">
                                    <div className="flex items-center gap-2 text-blue-400 mb-2">
                                        <Type size={16} />
                                        <h3 className="text-xs font-bold uppercase tracking-widest">Typography</h3>
                                    </div>

                                    <div className="space-y-2">
                                        <div className="flex justify-between text-[10px] text-slate-500 font-bold uppercase">
                                            <span>Font Scale</span>
                                            <span>{Math.round(fsScale * 100)}%</span>
                                        </div>
                                        <input
                                            type="range" min="0.5" max="2.0" step="0.1"
                                            value={fsScale} onChange={(e) => setFsScale(parseFloat(e.target.value))}
                                            className="w-full accent-blue-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                                        />
                                    </div>
                                </div>

                                <div className="space-y-4">
                                    <div className="flex items-center gap-2 text-purple-400 mb-2">
                                        <Layout size={16} />
                                        <h3 className="text-xs font-bold uppercase tracking-widest">Layout</h3>
                                    </div>

                                    <div className="space-y-2">
                                        <div className="flex justify-between text-[10px] text-slate-500 font-bold uppercase">
                                            <span>Vertical Offset</span>
                                            <span>{yOffset} px</span>
                                        </div>
                                        <input
                                            type="range" min="-20" max="50" step="1"
                                            value={yOffset} onChange={(e) => setYOffset(parseFloat(e.target.value))}
                                            className="w-full accent-purple-500 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                                        />
                                    </div>
                                </div>

                                <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl text-xs text-blue-300 whitespace-normal">
                                    <p>💡 Tip: You can hide this sidebar to view the full document.</p>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Canvas Area */}
                <div className="flex-1 bg-black/50 p-8 overflow-auto flex justify-center">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center text-slate-500 animate-pulse">
                            <Layout className="w-12 h-12 mb-4 opacity-50" />
                            <p className="uppercase tracking-widest text-xs font-bold">Rendering Preview...</p>
                        </div>
                    ) : pageData ? (
                        <div
                            className="relative shadow-2xl ring-1 ring-white/10 bg-white transition-transform duration-200 ease-out origin-top"
                            style={{
                                width: 'fit-content',
                                height: 'fit-content',
                                transform: `scale(${viewZoom})`
                            }}
                        >
                            <img src={pageData.image} alt="Page Preview" className="max-w-none" />

                            {/* Overlay Layer */}
                            <div className="absolute inset-0 pointer-events-none">
                                {pageData.items.map((item, idx) => {
                                    // Calculate simplified relative internal coordinates ??
                                    // Note: fitz Matrix(2,2) means image is 2x stats.
                                    // Item bbox is 1x stats. So we must multiply bbox by 2.

                                    const scale = 2;
                                    const x = item.bbox[0] * scale;
                                    const y = item.bbox[1] * scale;
                                    const w = (item.bbox[2] - item.bbox[0]) * scale;
                                    // const h = (item.bbox[3] - item.bbox[1]) * scale;

                                    // Font size also needs 2x scale + user scale
                                    const fontSize = item.fs * scale * fsScale;

                                    // Offset needs 2x scale
                                    const top = (item.bbox[3] * scale) + (fontSize) + (yOffset * scale);
                                    // WAIT: Main logic uses `bbox.y1 + trans_fs + 1 + y_offset`
                                    // Here `fontSize` is `trans_fs` (approx).
                                    // Let's approximate visual position.

                                    return (
                                        <div key={idx}
                                            style={{
                                                position: 'absolute',
                                                left: x,
                                                top: (item.bbox[3] * scale) + (yOffset * scale) + (fontSize * 0.2), // Heuristic
                                                width: w,
                                                color: 'red', // Fixed for preview, or use props
                                                fontSize: `${fontSize}px`,
                                                lineHeight: 1,
                                                fontFamily: 'serif',
                                                textAlign: alignment === 'auto' ? (targetLang === 'Arabic' ? 'right' : 'left') : alignment,
                                                direction: targetLang === 'Arabic' ? 'rtl' : 'ltr',
                                                // backgroundColor: 'rgba(255, 255, 0, 0.2)' // Debug
                                            }}>
                                            {item.translation}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ) : null}
                </div>
            </div>
        </div>
    );
};

export default Editor;
