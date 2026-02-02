import os
import io
import fitz  # PyMuPDF
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import logging
import re
import json
import asyncio
import concurrent.futures
import uuid
import base64
import traceback
import requests
from deep_translator import GoogleTranslator
from translation_engine import TranslationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from nllb_engine import NLLBEngine
    NLLB_AVAILABLE = True
except ImportError:
    NLLB_AVAILABLE = False
    logger.warning("NLLB Engine Import Failed. Running in Online-Only mode.")
    class NLLBEngine:
         def __init__(self, *args, **kwargs): raise ImportError("NLLB Unavailable")

DEEPSEEK_KEYS = [
    "sk-ced5476b3f1146f989f2cba1610032dd",
    "sk-c90dcb5f489d4db69179923b96cd111e",
    "sk-6d2c18f43f294362a8c4d62eb42576f2",
    "sk-9806b4410c0f4480ac30ada2ecd62c55",
    "sk-26a87bd10da04a2c878d9342c0e2c01b",
    "sk-2068a93a7637462680f86dd8586fde42",
    "sk-ff384e5b963545338065e60d0f6419bb",
    "sk-c6a7e4bd5b16430cb69e5534e5dfb301",
    "sk-0efd91df6e4a4cc2b87f38bddfaef91c",
    "sk-3dea1957a9184dfc9d38f8cae9d4089a"
]

# Ensure temp directory exists
os.makedirs("temp_outputs", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# ... imports ...

app = FastAPI(title="AI PDF Translator Ultra - v18.0 Overlay Engine")

# Mount temp_outputs for live preview
app.mount("/outputs", StaticFiles(directory="temp_outputs"), name="outputs")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    with open("d:\\app_file\\translat\\backend\\global_error.txt", "w") as f:
        f.write(f"Global Error: {type(exc).__name__}: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"message": f"Global Error: {str(exc)}"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.active_connections[client_id] = websocket
        return client_id

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            try:
                del self.active_connections[client_id]
            except:
                pass

    async def send_msg(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(data)
            except:
                logger.warning(f"Failed to send message to {client_id}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    try:
        await websocket.send_json({"client_id": client_id})
        
        # User requested 1s Connection Check (Heartbeat)
        async def heartbeat_loop():
            try:
                while True:
                    await asyncio.sleep(1)
                    # Check if connection is still valid in manager before sending
                    if client_id in manager.active_connections:
                        from datetime import datetime
                        now_str = datetime.now().strftime("%H:%M:%S")
                        await manager.send_msg(client_id, {"type": "ping", "status": "active", "last_update": now_str})
                    else:
                        break
            except: pass

        # Start Heartbeat
        asyncio.create_task(heartbeat_loop())

        while True:
            # We must listen to keep connection open and handle incoming Pongs if client sends them
            data = await websocket.receive_text()
            if data == "ping":
                await manager.send_msg(client_id, {"type": "pong"}) 
    except WebSocketDisconnect:
        manager.disconnect(client_id)

def hex_to_rgb(hex_color: str):
    try:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    except:
        return (1, 0, 0)


def apply_manual_overrides(lines: list, translations: list):
     try:
         loop_len = min(len(lines), len(translations))
         for i in range(loop_len - 1):
             w1 = str(lines[i]).lower().strip()
             w2 = str(lines[i+1]).lower().strip()
             if w1.startswith("ass") and w2.startswith("lec"):
                 translations[i] = "مدرس"
                 translations[i+1] = "مساعد"
             elif w1.startswith("ass") and w2.startswith("prof"):
                 translations[i] = "أستاذ"
                 translations[i+1] = "مساعد"
         for i in range(loop_len):
             w = str(lines[i]).lower().strip()
             curr = translations[i]
             if "ass" in w and "lec" in w:
                 translations[i] = "مدرس مساعد"
             elif w.startswith("lec") and curr != "مساعد" and "ass" not in w:
                 translations[i] = "محاضر"
             if "bmr" in w:
                 translations[i] = "معدل الأيض الأساسي"
     except: pass
     return translations

async def translate_batch_robust(lines: list, api_key: str, target_lang: str, context: str = "", mode: str = "line"):
    # ... (Glossary Pre-processing omitted for brevity, logic remains) ...
    if not lines: return []
    
    # --- LAYER 1: STRICT GLOSSARY (Pre-Processing) ---
    glossary_pre = {
        r"\bAss\.?\s*Lec\.?": "Assistant Lecturer",
        r"\bAss\.?\s*Prof\.?": "Assistant Professor",
        r"\bLec\.?": "Lecturer"
    }
    pre_processed_lines = []
    for line in lines:
        if line:
            clean = str(line)
            try:
                for pattern, replacement in glossary_pre.items():
                    clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
            except: pass
            pre_processed_lines.append(clean)
        else:
            pre_processed_lines.append("")


def call_deepseek_api(prompt, api_key):
    try:
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a specialized academic translator. Output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "temperature": 0.1
        }
        # 60s timeout
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"DeepSeek API Error {resp.status_code}: {resp.text}")
            raise Exception(f"DeepSeek {resp.status_code}")
    except Exception as e:
         raise e



        
async def translate_batch_robust(lines: list, api_key: str, target_lang: str, context: str = "", mode: str = "line"):
    if not lines: return []
    
    # --- LAYER 1: STRICT GLOSSARY (Pre-Processing) ---
    glossary_pre = {
        r"\\bAss\\.?\\s*Lec\\.?": "Assistant Lecturer",
        r"\\bAss\\.?\\s*Prof\\.?": "Assistant Professor",
        r"\\bLec\\.?": "Lecturer"
    }
    
    pre_processed_lines = []
    for line in lines:
        if line:
            clean = str(line)
            try:
                for pattern, replacement in glossary_pre.items():
                    clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
            except: pass
            pre_processed_lines.append(clean)
        else:
            pre_processed_lines.append("")

    loop = asyncio.get_event_loop()
    response_text = None
    
    # DEBUG: Check what we are receiving
    if len(pre_processed_lines) > 0:
        logger.info(f"TRANSLATE INPUT [Mode: {mode}]: {pre_processed_lines[:5]}...")


    # --- LAYER 0: NLLB (Offline/Faster) ---
    if True: # Force NLLB Check first
         if hasattr(PipelineManager, "shared_nllb") and PipelineManager.shared_nllb and PipelineManager.shared_nllb.translator:
             try:
                 # Check Mode
                 if mode == "word":
                     # WORD-BY-WORD (Literal)
                     # No joining, just translate each item directly
                     text_inputs = [l if l else "" for l in pre_processed_lines]
                     res = await loop.run_in_executor(None, PipelineManager.shared_nllb.translate_batch, text_inputs, "english", target_lang)
                     return res
                     
                 else:
                     # SENTENCE/LINE (Contextual) - DEFAULT
                     # Smart Context Strategy: Join -> Translate -> Distribute
                     valid_indices = [i for i, x in enumerate(pre_processed_lines) if x and x.strip()]
                     valid_texts = [pre_processed_lines[i] for i in valid_indices]
                     
                     final_res = [""] * len(pre_processed_lines)
                     
                     if valid_texts:
                         joined_text = " ".join(valid_texts)
                         
                         # Translate the full context
                         res_joined = await loop.run_in_executor(None, PipelineManager.shared_nllb.translate_batch, [joined_text], "english", target_lang)
                         
                         if res_joined and res_joined[0]:
                             translated_sentence = res_joined[0]
                             arb_words = translated_sentence.split()
                             
                             n_in = len(valid_texts)
                             n_out = len(arb_words)
                             
                             if n_out > 0:
                                 for k, original_idx in enumerate(valid_indices):
                                     start = (k * n_out) // n_in
                                     end = ((k + 1) * n_out) // n_in
                                     
                                     if k == n_in - 1: end = n_out
                                     
                                     chunk = arb_words[start:end]
                                     final_res[original_idx] = " ".join(chunk)
                                     
                     return final_res



             except Exception as nllb_err:
                 logger.error(f"NLLB Robust Failed: {nllb_err}")

    # PRIMARY ENGINE: GEMINI 1.5 FLASH (Using Pool)
    try:
         # Configure Gemini
         genai.configure(api_key=api_key)
         model = genai.GenerativeModel('gemini-1.5-flash')
         
         if mode == "word":
             gemini_prompt = f"""
             SYSTEM: You are a professional translator engine.
             TASK: Translate the following array of English words/phrases into ARABIC.
             
             METHODOLOGY (CRITICAL):
             1. First, read the entire array to understand the full sentence.
             2. Translate the *meaning* of the full sentence to Arabic.
             3. Map the Arabic words back to the input array structure one-by-one.
             4. HANDLE ABBREVIATIONS: "Ass. Lec." -> "مدرس مساعد", "Prof." -> "أستاذ".
             
             RULES:
             1. Output MUST be a JSON Array of strings.
             2. Length MUST match exactly (One-to-One mapping).
             3. ABSOLUTELY NO ENGLISH IN OUTPUT.
             4. If the arabic translation has more words than English, combine them or distribute meaningfully.
             
             EXAMPLES:
             Input: ["Energy", "is", "defined", "as"]
             Output: ["الطاقة", "هي", "تعرف", "بأنها"]
             
             Input: ["Ass.", "Lec.", "Maher"]
             Output: ["مدرس", "مساعد", "ماهر"]
             
             INPUT ARRAY:
             {json.dumps(pre_processed_lines, ensure_ascii=False)}
             
             OUTPUT JSON ARRAY:
             """
         else:
             gemini_prompt = f"""
             ROLE: Expert Academic Translator.
             TARGET LANGUAGE: ARABIC.
             
             STRICT RULES:
             1. Output MUST be a JSON ARRAY of strings in ARABIC.
             2. Translate EVERYTHING. Do not leave English text.
             3. Context: "{context[:200]}..."
             
             INPUT ARRAY:
             {json.dumps(pre_processed_lines, ensure_ascii=False)}
             
             OUTPUT FORMAT:
             Return a JSON ARRAY of strings matching the input length exactly.
             """
             
         response = await loop.run_in_executor(None, lambda: model.generate_content(gemini_prompt))
         response_text = response.text
         
    except Exception as gemini_err:
         logger.warning(f"Gemini Primary Failed ({{gemini_err}})")
         response_text = None

    final_results = pre_processed_lines 
    
    if response_text:
        cleaned = re.sub(r'```json\n?|\n?```', '', response_text).strip()
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match: cleaned = match.group(0)
        
        try:
            results = json.loads(cleaned)
            if isinstance(results, list) and len(results) == len(pre_processed_lines):
                  if target_lang.lower() in ["arabic", "ar"]:
                      validated = []
                      for original, translated in zip(pre_processed_lines, results):
                          if translated: validated.append(translated)
                          else: validated.append(None)
                      final_results = [t if t is not None else "" for t in validated]
                  else:
                       final_results = [t if t is not None else "" for t in results]
        except: pass
    
    # Fallback to Google Translate
    if final_results == pre_processed_lines:
        try:
            lang_code = "ar" if target_lang.lower() == "arabic" else "en"
            translator = GoogleTranslator(source='auto', target=lang_code)
            res = await loop.run_in_executor(None, translator.translate_batch, pre_processed_lines)
            if res and isinstance(res, list) and len(res) == len(pre_processed_lines):
                 res = [r if r is not None else "" for r in res]
                 return res
        except Exception as e: pass
        
    return final_results

@app.post("/debug_translate")
async def debug_translate(text: str = Form(...), api_key: str = Form(...), target_lang: str = Form("Arabic")):
    try:
        # Use first DeepSeek key if default/invalid provided
        key_to_use = api_key
        if not key_to_use or "AIza" in key_to_use: key_to_use = DEEPSEEK_KEYS[0]
        
        resp = await asyncio.to_thread(call_deepseek_api, f"Translate to {target_lang}: {text}", key_to_use)
        deepseek_res = resp
    except Exception as e: deepseek_res = f"Error: {e}"
    
    try:
        lang_code = "ar" if target_lang.lower() == "arabic" else "en"
        translator = GoogleTranslator(source='auto', target=lang_code)
        fallback_res = await asyncio.to_thread(translator.translate, text)
    except Exception as e: fallback_res = f"Error: {e}"
    
    return JSONResponse({"gemini": gemini_res, "fallback": fallback_res, "final": gemini_res if "Error" not in gemini_res else fallback_res, "source": "Gemini" if "Error" not in gemini_res else "Fallback"})

# Helper: Check for Math
def is_math_line(text: str) -> bool:
    math_indicators = ["=", "+", "−", "×", "÷", "∫", "∑", "∂", "√", "∈", "∞", "≠", "≈", "≤", "≥"]
    if any(m in text for m in math_indicators): return True
    alpha_count = sum(c.isalpha() for c in text)
    if len(text) > 0 and (alpha_count / len(text)) < 0.4: return True
    return False

# Helper: Merge Rectangles
def merge_rects(rects, threshold=5):
    if not rects: return []
    sorted_rects = sorted(rects, key=lambda r: r.y0)
    merged = []
    current = sorted_rects[0]
    for i in range(1, len(sorted_rects)):
        next_r = sorted_rects[i]
        expanded = fitz.Rect(current.x0 - threshold, current.y0 - threshold, 
                             current.x1 + threshold, current.y1 + threshold)
        if expanded.intersects(next_r):
            current = current | next_r 
        else:
            merged.append(current)
            current = next_r
    merged.append(current)
    return merged

# --- v17.0: Accordion Expander Helper ---
def get_vertical_slices(page, threshold=10):
    """
    Scans the page for vertical gaps to identify safe "slicing" points.
    Returns a list of dicts: {'start': y, 'end': y, 'type': 'content'|'gap'}
    """
    # 1. Collect all non-empty vertical intervals
    intervals = []
    
    # Text
    blocks = page.get_text("blocks")
    for b in blocks:
        intervals.append((b[1], b[3])) # y0, y1

    # Images / Drawings
    # (Be conservative: merge all vector/image rects)
    image_list = page.get_images(full=True)
    drawings = page.get_drawings()
    
    for img in image_list:
        try:
             rects = page.get_image_rects(img[0])
             for r in rects: intervals.append((r.y0, r.y1))
        except: pass
        
    for d in drawings:
        r = d["rect"]
        if r.width > 2 and r.height > 2: # Ignore tiny dots
            intervals.append((r.y0, r.y1))

    if not intervals:
        return [{'start': 0, 'end': page.rect.height, 'type': 'content'}]

    # 2. Sort & Merge Intervals
    intervals.sort(key=lambda x: x[0])
    merged = []
    if intervals:
        current_y0, current_y1 = intervals[0]
        for next_y0, next_y1 in intervals[1:]:
            if next_y0 < current_y1 + threshold: # Overlap or close
                current_y1 = max(current_y1, next_y1)
            else:
                merged.append((current_y0, current_y1))
                current_y0, current_y1 = next_y0, next_y1
        merged.append((current_y0, current_y1))

    # 3. Create Slices (Content vs Gaps)
    slices = []
    last_y = 0
    
    for y0, y1 in merged:
        # Gap before content?
        if y0 > last_y + 5: # Minimal gap size
            slices.append({'start': last_y, 'end': y0, 'type': 'gap'})
        
        # Content
        slices.append({'start': y0, 'end': y1, 'type': 'content'})
        last_y = y1
    
    # Final Gap
    if last_y < page.rect.height:
         slices.append({'start': last_y, 'end': page.rect.height, 'type': 'gap'})
         
    return slices


# --- WORKER: Process Single Page Word-by-Word (v18.0) ---
# --- v19.0: PARALLEL PIPELINE ARCHITECTURE ---

class PipelineManager:
    def __init__(self, pdf_bytes, settings, client_id, font_path):
        self.pdf_bytes = pdf_bytes
        self.settings = settings
        self.client_id = client_id
        self.font_path = font_path
        
        # Queues
        self.translate_queue = asyncio.Queue()
        self.paste_queue = asyncio.Queue()
        
        # State
        self.total_pages = 0
        self.processed_pages = 0
        self.errors = []
        
    async def start(self):
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        self.total_pages = len(doc)
        doc.close()
        
        # 1. Start Extractor (Producer)
        extract_task = asyncio.create_task(self.extractor_producer())
        
        # 2. Start Translators (Consumers of extract_queue)
        # We use 10 workers to saturate the Gemini Pool
        translators = [asyncio.create_task(self.translator_worker(i)) for i in range(10)]
        
        # 3. Start Pasters (Consumers of paste_queue)
        # 4 Workers for CPU-bound rendering
        pasters = [asyncio.create_task(self.paster_worker(i)) for i in range(4)]
        
        # Wait for Extraction to finish (it pushes to translate_queue)
        await extract_task
        
        # Wait for queues to empty
        await self.translate_queue.join()
        await self.paste_queue.join()
        
        # Cancel workers
        for t in translators: t.cancel()
        for p in pasters: p.cancel()
        
        # --- MERGE PDFS ---
        try:
            final_doc = fitz.open()
            for i in range(self.total_pages):
                try:
                    # Try both line and word filenames just in case
                    p_path = f"temp_outputs/v19_line_p{i}.pdf"
                    if not os.path.exists(p_path):
                        p_path = f"temp_outputs/v19_pipeline_p{i}.pdf" # Legacy/Word fallback
                        
                    if os.path.exists(p_path):
                        p_doc = fitz.open(p_path)
                        final_doc.insert_pdf(p_doc)
                        p_doc.close()
                except Exception as e:
                    logger.error(f"Merge Error p{i}: {e}")
            
            final_filename = f"final_{self.client_id}.pdf"
            final_path = f"temp_outputs/{final_filename}"
            final_doc.save(final_path)
            final_doc.close()
            
            # Notify Frontend
            final_url = f"http://localhost:8000/outputs/{final_filename}"
            await manager.send_msg(self.client_id, {
                "type": "final_ready",
                "url": final_url,
                "filename": final_filename
            })
            logger.info(f"Final PDF Merged: {final_path}")
            
        except Exception as e:
            logger.error(f"Merge Failed: {e}")
            
        return self.errors

    async def extractor_producer(self):
        """Stage 1: Read pages and extract text (Runs in ThreadPool)"""
        loop = asyncio.get_event_loop()
        # We can run extraction in parallel using RunInExecutor if CPU bound, 
        # but fitz is fast. Let's iterate and dispatch.
        # Actually, to be truly parallel, we should map page numbers to threads.
        
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for page_num in range(self.total_pages):
                tasks.append(loop.run_in_executor(executor, self._extract_page_sync, page_num))
            
            await asyncio.gather(*tasks)
            
    def _extract_page_sync(self, page_num):
        try:
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            page = doc[page_num]
            words = page.get_text("words")
            doc.close()
            
            # Push to Translate Queue
            # We need to use run_coroutine_threadsafe because we are in a thread
            loop = asyncio.new_event_loop() 
            # WAIT: We are in a thread, we can't easily push to an async queue from here without the main loop.
            # BETTER APPROACH: The extractor_producer is ASYNC. It should call simple sync functions.
            return (page_num, words)
        except Exception as e:
            logger.error(f"Extract Error p{page_num}: {e}")
            return (page_num, None)

    async def extractor_producer(self):
        """Revised Stage 1: Async Dispatcher"""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for page_num in range(self.total_pages):
                # Submit task to thread pool
                futures.append(loop.run_in_executor(executor, self._extract_page_sync, page_num))
            
            for future in asyncio.as_completed(futures):
                try:
                    page_num, words = await future
                    if words:
                        await self.translate_queue.put((page_num, words))
                        await manager.send_msg(self.client_id, {"type": "log", "original": f"Extracted Page {page_num+1}", "translated": "..."})
                except Exception as e:
                    logger.error(f"Extraction failed: {e}")

    def _extract_page_sync(self, page_num):
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        page = doc[page_num]
        words = page.get_text("words")
        doc.close()
        return page_num, words

    async def translator_worker(self, worker_id):
        """Stage 2: Translation (NLLB + Online Fallback)"""
        
        # 1. Determine Mode
        api_keys_str = self.settings.get("api_key", "")
        # Check for specific "OFFLINE" flag
        is_offline_req = "OFFLINE" in api_keys_str.upper()
        
        # Check NLLB Readiness
        nllb_ready = hasattr(PipelineManager, "shared_nllb") and PipelineManager.shared_nllb and PipelineManager.shared_nllb.translator
        
        # Default keys for Online Fallback
        DEFAULT_KEYS = [
            "AIzaSyCV4RzAV0IJkRz_ZKux4kzHm1VvPLguiJI",
            "AIzaSyB14R8e4HKbPnt3c_mAqF77tv3tA4RXJnw",
            "AIzaSyCXOJ0t0vvg-a1_eirE8YowqDLhW-RqP4A",
            "AIzaSyCzF3Bp4Qx8uHmMNEOjT7q0BFAWtcMsN7Y"
        ]
        
        engine = None
        use_nllb = False
        
        if True: # Force NLLB Attempt
             try:
                 if not nllb_ready:
                     if not hasattr(PipelineManager, "shared_nllb") or not PipelineManager.shared_nllb:
                          logger.info("NLLB Forced: Initializing Engine on Worker...")
                          PipelineManager.shared_nllb = NLLBEngine(device='cpu')
                          PipelineManager.shared_nllb.load_model()
                     engine = PipelineManager.shared_nllb
                     use_nllb = True
                 else:
                     use_nllb = True
                     engine = PipelineManager.shared_nllb
             except Exception as nllb_init_err:
                 logger.error(f"Failed to initialize NLLB: {nllb_init_err}. Falling back to Online Engine.")
                 use_nllb = False


        while True:
            try:
                page_num, words = await self.translate_queue.get()
                texts = [w[4] for w in words]
                target_lang = self.settings.get("target_lang", "Arabic")
                if not words: continue 
                
                # --- GLOSSARY PRE-PROCESSING (STRICT) ---
                # Expand academic abbreviations to ensure correct NLLB translation
                glossary_pre = {
                    r"\bAss\.?\s*Lec\.?": "Assistant Lecturer",
                    r"\bAss\.?\s*Prof\.?": "Assistant Professor",
                    r"\bLec\.?": "Lecturer",
                    r"\bDr\.?": "Doctor",
                    r"\bProf\.?": "Professor"
                }
                
                processed_texts = []
                for t in texts:
                    if t:
                        clean = str(t)
                        for pattern, replacement in glossary_pre.items():
                             clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
                        processed_texts.append(clean)
                    else:
                         processed_texts.append("")
                
                texts = processed_texts
                # ------------------------------------------

                final_translations = []

                # --- HYBRID TRANSLATION LOGIC ---
                try:
                    # 1. Primary: NLLB
                    nllb_result = []
                    if use_nllb:
                        try:
                            # ... (Existing NLLB Logic) ...
                            mode = self.settings.get("translation_mode", "line")
                            loop = asyncio.get_event_loop()
                            
                            if True: # Force Context-Aware Mode (Visual Segmentation) for ALL requests
                                # LINE MODE (Refined with Visual Segmentation)
                                # Instead of joining the whole page, we group words by visual lines/blocks.
                                
                                nllb_result = [""] * len(texts)
                                
                                # 1. Segment the page
                                segments = []
                                current_segment = []
                                last_w = None
                                
                                for i, w_data in enumerate(words):
                                    text = texts[i] # Use glossary-processed text
                                    if not text or not text.strip(): continue
                                    
                                    is_split = False
                                    if last_w:
                                        # Split on new Line or Block
                                        if w_data[6] != last_w[6] or w_data[5] != last_w[5]:
                                            is_split = True
                                        # Split on wide gap (Columns) - threshold 15px
                                        elif w_data[0] > last_w[2] + 15: 
                                            is_split = True
                                    
                                    if is_split and current_segment:
                                        segments.append(current_segment)
                                        current_segment = []
                                    
                                    current_segment.append((i, text))
                                    last_w = w_data
                                
                                if current_segment: segments.append(current_segment)
                                
                                # 2. Process Segments
                                for seg in segments:
                                    seg_indices = [s[0] for s in seg]
                                    seg_texts = [s[1] for s in seg]
                                    joined_text = " ".join(seg_texts)
                                    
                                    # --- GLOSSARY APPLICATION (CONTEXT AWARE) ---
                                    # Apply replacements on the full phrase to catch split tokens like "Ass. Lec."
                                    glossary_map = {
                                        r"\bAss\.?\s*Lec\.?": "Assistant Lecturer",
                                        r"\bAss\.?\s*Prof\.?": "Assistant Professor",
                                        r"\bLec\.?": "Lecturer",
                                        r"\bDr\.?": "Doctor",
                                        r"\bProf\.?": "Professor",
                                        r"\bEng\.?": "Engineer"
                                    }
                                    for pattern, replacement in glossary_map.items():
                                        joined_text = re.sub(pattern, replacement, joined_text, flags=re.IGNORECASE)
                                    
                                    # --- HYBRID TRANSLATION FOR SEGMENT ---
                                    translated_sentence = None
                                    
                                    # Try Google First
                                    try:
                                         def google_sync(txt):
                                             from deep_translator import GoogleTranslator
                                             return GoogleTranslator(source='auto', target='ar').translate(txt)
                                         
                                         translated_sentence = await loop.run_in_executor(None, google_sync, joined_text)
                                    except Exception as g_err:
                                         # logger.warning(f"Google Seg Failed: {g_err}") 
                                         pass
                                    
                                    # Fallback to NLLB
                                    if not translated_sentence:
                                         res_joined = await loop.run_in_executor(None, engine.translate_batch, [joined_text], "english", target_lang)
                                         if res_joined and res_joined[0]:
                                             translated_sentence = res_joined[0]
                                    
                                    # Distribute Result
                                    if translated_sentence:
                                         arb_words = translated_sentence.split()
                                         n_in = len(seg_texts)
                                         n_out = len(arb_words)
                                         
                                         if n_out > 0:
                                             for k, original_idx in enumerate(seg_indices):
                                                 start = (k * n_out) // n_in
                                                 end = ((k + 1) * n_out) // n_in
                                                 if k == n_in - 1: end = n_out
                                                 chunk = arb_words[start:end]
                                                 nllb_result[original_idx] = " ".join(chunk)

                        except Exception as e:
                            logger.error(f"NLLB/Hybrid Segmentation Failed p{page_num}: {e}")
                            nllb_result = []

                    # 2. Validation / Fallback
                    final_translations = []
                    
                    if not nllb_result or all(not x for x in nllb_result):
                         final_translations = texts 
                    else:
                         final_translations = nllb_result

                    if not final_translations: final_translations = texts

                except Exception as e:
                    logger.error(f"Critical Translation Logic Error p{page_num}: {e}")
                    final_translations = texts # Ultimate Fallback

                # Validation & Push
                try:
                    if len(final_translations) < len(words):
                        final_translations.extend([""] * (len(words) - len(final_translations)))
                    elif len(final_translations) > len(words):
                        final_translations = final_translations[:len(words)]

                    await self.paste_queue.put((page_num, words, final_translations))
                except Exception as e:
                     logger.error(f"Failed to push to paste queue p{page_num}: {e}")
                
                self.translate_queue.task_done()
                
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Translator {worker_id} Fatal Error: {e}")
                self.translate_queue.task_done()

    async def paster_worker(self, worker_id):
        """Stage 3: Rendering (ThreadPool)"""
        loop = asyncio.get_event_loop()
        while True:
            try:
                page_num, words, translations = await self.paste_queue.get()
                
                # Offload to Thread for rendering to avoid blocking event loop
                await loop.run_in_executor(None, self._paste_page_sync, page_num, words, translations)
                
                self.processed_pages += 1
                await manager.send_msg(self.client_id, {"type": "progress", "current": self.processed_pages, "total": self.total_pages})
                
                # Notify Frontend of Page Completion (Live Preview)
                page_filename = f"v19_line_p{page_num}.pdf"
                preview_url = f"http://localhost:8000/outputs/{page_filename}"
                await manager.send_msg(self.client_id, {
                    "type": "page_ready", 
                    "page": page_num + 1, 
                    "url": preview_url
                })
                
                self.paste_queue.task_done()
                
            except asyncio.CancelledError: break
            except Exception as e:
                logger.error(f"Paster {worker_id} Error: {e}")
                self.paste_queue.task_done()


    def _paste_page_sync(self, page_num, words, translations):
        try:
            doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
            page = doc[page_num]
            
            # Setup Fonts/Colors
            text_color = hex_to_rgb(self.settings.get("text_color", "#FF0000"))
            font_path = self.font_path
            has_custom_font = os.path.exists(font_path)
            font_name = "amiri" if has_custom_font else "helv"
            if has_custom_font:
                page.insert_font(fontname=font_name, fontfile=font_path)
            
            # Check Layout Mode
            mode = self.settings.get("translation_mode", "line")
            
            if mode == "word":
                # --- WORD MODE: Distribute words in exact places ---
                for i, w_data in enumerate(words):
                    if i >= len(translations) or not translations[i]: continue
                    
                    x0, y0, x1, y1, original_text, block, line, word = w_data
                    trans_text = translations[i]
                    
                    fs = 5 
                    trans_y = y1 + fs
                    
                    try:
                        if self.settings.get("target_lang", "Arabic").lower() == "arabic":
                             reshaped = reshape(trans_text)
                             final_trans = get_display(reshaped)
                        else: final_trans = trans_text
                        
                        # Calculate Text Width for Centering
                        try:
                             tw = fitz.get_text_length(final_trans, fontname=font_name, fontsize=fs)
                        except: tw = len(final_trans) * fs * 0.5

                        # Center Alignment: Align center of translation with center of original word
                        # x_center = x0 + (x1 - x0) / 2
                        # trans_x = x_center - tw / 2
                        word_width = x1 - x0
                        trans_x = x0 + (word_width - tw) / 2
                        
                        page.insert_text((trans_x, trans_y), final_trans, fontsize=fs, fontname=font_name, color=text_color)
                    except: pass
                    
            else:
                # --- LINE MODE: Group by line and place full sentence ---
                
                # 1. Group by (Block, Line)
                lines_map = {} # Key: (block_idx, line_idx) -> List of (word_data, translation_fragment)
                
                for i, w_data in enumerate(words):
                    if i >= len(translations): break
                    # w_data keys: 0:x0, 1:y0, 2:x1, 3:y1, 4:text, 5:block, 6:line, 7:word
                    block_idx = w_data[5]
                    line_idx = w_data[6]
                    key = (block_idx, line_idx)
                    
                    if key not in lines_map: lines_map[key] = []
                    lines_map[key].append((w_data, translations[i]))
                
                # 2. Process each line
                for key, items in lines_map.items():
                    if not items: continue
                    
                    # Sort items by X-coordinate (Reading Order)
                    items.sort(key=lambda x: x[0][0])
                    
                    # Reconstruct Full Arabic Sentence
                    trans_fragments = [item[1] for item in items if item[1] and item[1].strip()]
                    if not trans_fragments: continue
                    
                    full_arabic_line = " ".join(trans_fragments)
                    
                    # Calculate Line Bounding Box (Union of all words)
                    x0 = min(item[0][0] for item in items)
                    y0 = min(item[0][1] for item in items)
                    x1 = max(item[0][2] for item in items)
                    y1 = max(item[0][3] for item in items)
                    
                    # Render Config
                    fs = 6 
                    line_height = y1 - y0
                    render_x = x0
                    render_y = y1 + fs + 1
                    
                    try:
                        is_arabic = self.settings.get("target_lang", "Arabic").lower() == "arabic"
                        if is_arabic:
                             reshaped = reshape(full_arabic_line)
                             final_trans = get_display(reshaped)
                        else: final_trans = full_arabic_line
                        
                        page.insert_text((render_x, render_y), final_trans, fontsize=fs, fontname=font_name, color=text_color)
                    except: pass
                



            filename = f"v19_line_p{page_num}.pdf"
            output_path = f"temp_outputs/{filename}"
            
            # Save single page
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_doc.save(output_path)
            new_doc.close()
            doc.close()
            
        except Exception as e:
            logger.error(f"Paste Sync Error p{page_num}: {e}")


# --- WORKER: Process Single Page (v18.0 Overlay Approach - Preserves Original) ---
async def process_single_page(page_num: int, pdf_bytes: bytes, settings: dict, client_id: str, font_path: str, sem: asyncio.Semaphore):
    # Mode Dispatch
    mode = settings.get("translation_mode", "line")
    if mode == "word":
        return await process_single_page_word_by_word(page_num, pdf_bytes, settings, client_id, font_path, sem)

    async with sem:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[page_num]
            page_rect = page.rect
            
            api_key = settings.get("api_key")
            target_lang = settings.get("target_lang", "Arabic")
            text_color = hex_to_rgb(settings.get("text_color", "#FF0000"))
            font_name = "amiri"
            has_custom_font = os.path.exists(font_path)
            
            # Register custom font if available
            if has_custom_font:
                page.insert_font(fontname=font_name, fontfile=font_path)

            # 1. Extract ALL text blocks with full detail (preserving whitespace)
            try:
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)["blocks"]
            except:
                blocks = []
            
            # 2. Collect all text items with their positions
            texts_to_translate = []
            text_items = []
            
            # Also collect all Y positions to detect available space
            all_y_positions = []
            
            for b in blocks:
                if b["type"] == 0:  # Text block
                    for l in b["lines"]:
                        line_text = "".join([s["text"] for s in l["spans"]]).strip()
                        if line_text:
                            bbox = fitz.Rect(l["bbox"])
                            all_y_positions.append(bbox.y0)
                            all_y_positions.append(bbox.y1)
                            
                            # Get font info from first span
                            first_span = l["spans"][0]
                            fs = first_span["size"]
                            font = first_span.get("font", "")
                            span_color = first_span.get("color", 0)  # Original text color
                            
                            is_serif = "Times" in font or "Serif" in font
                            is_math = is_math_line(line_text)
                            
                            item = {
                                "bbox": bbox,
                                "text": line_text,
                                "fs": fs,
                                "is_serif": is_serif,
                                "is_math": is_math,
                                "orig_color": span_color,
                                "index": -1
                            }
                            
                            if not is_math:
                                item["index"] = len(texts_to_translate)
                                texts_to_translate.append(line_text)
                            
                            text_items.append(item)
            
            # Also collect Image RECTS to avoid overlapping images
            images = page.get_images()
            for img in images:
                try:
                    rects = page.get_image_rects(img[0])
                    for r in rects:
                        all_y_positions.append(r.y0)
                        # all_y_positions.append(r.y1) # Only top matters for "next element"? 
                        # actually we want to know if there is something BELOW.
                        all_y_positions.append(r.y1)
                except: pass

            # Sort Y positions for spacing calculation
            all_y_positions.sort()
            
            # 3. Translate all texts in batch
            if texts_to_translate:
                ctx = " ".join(texts_to_translate)[:1000]
                translations = await translate_batch_robust(texts_to_translate, api_key, target_lang, context=ctx, mode="line")
            else:
                translations = []
            
            # 4. Calculate available space and insert translations
            for i, item in enumerate(text_items):
                if item["is_math"] or item["index"] < 0:
                    continue  # Skip math expressions
                
                idx = item["index"]
                if idx >= len(translations) or not translations[idx]:
                    continue
                
                bbox = item["bbox"]
                fs = item["fs"]
                trans_text = translations[idx]
                
                # Calculate available vertical space
                # Find next element below current bbox
                next_y = page_rect.height  # Default to page bottom
                for y in all_y_positions:
                    if y > bbox.y1 + 1:  # Very tight buffer
                        next_y = y
                        break
                
                available_space = next_y - bbox.y1
                
                # Calculate translation font size
                # Adaptive: Fit within available space strictly
                # Space needed approx: fontsize * 1.2
                # max_fs * 1.2 <= available_space  => max_fs <= available_space / 1.2
                max_fs = (available_space - 1) / 1.1 # Even tighter calculation
                
                fs_scale = float(settings.get("fs_scale", 1.0))
                y_offset = float(settings.get("y_offset", 0.0))
                
                # Logic: Original - 2, but Cap at 16 to prevent massive title overlap
                trans_fs = min(max(8, fs - 6), 16) * fs_scale
                
                # Strict floor: If result is too small (<5pt), it's unreadable.
                # User said "never overlap". Better to skip than overlap or be microscopic?
                # User also said "shrink the word".
                # We will limit to 5pt. If max_fs < 5, we skip this line.
                if trans_fs < 4:
                    trans_fs = 4 # Clamp to minimum legible size instead of skipping 

                # Calculate Y position: Align top of translation to bbox.y1 + 1 (tight)
                trans_y = bbox.y1 + trans_fs + 1 + y_offset
                
                # Ensure we don't go past page boundary
                if trans_y > page_rect.height - 5:
                    continue  # Skip if no space
                
                # Prepare Arabic text with proper RTL handling
                if target_lang.lower() == "arabic":
                    try:
                        reshaped = reshape(trans_text)
                        final_trans = get_display(reshaped)
                        used_font = font_name if has_custom_font else "helv"
                    except:
                        final_trans = trans_text
                        used_font = "helv"
                else:
                    final_trans = trans_text
                    used_font = "helv"
                
                # Calculate text width for background
                try:
                    text_width = fitz.get_text_length(final_trans, fontname=used_font, fontsize=trans_fs)
                except:
                    text_width = len(final_trans) * trans_fs * 0.5
                
                # Align based on settings
                alignment = settings.get("alignment", "auto")
                if alignment == "auto":
                     if target_lang.lower() == "arabic": alignment = "right"
                     else: alignment = "left"
                
                if alignment == "right":
                    trans_x = max(bbox.x0, bbox.x1 - text_width)
                elif alignment == "center":
                    trans_x = bbox.x0 + (bbox.width - text_width) / 2
                else: # left
                     trans_x = bbox.x0
                
                # Insert translated text
                try:
                    # Metrics Calculation
                    word_count = len(final_trans.split())
                    
                    # Send Granular Update (Throttle to avoid freezing UI)
                    if i % 2 == 0: 
                        await manager.send_msg(client_id, {
                            "type": "log",
                            "original": item["text"][:50] + "..." if len(item["text"]) > 50 else item["text"],
                            "translated": final_trans[:50] + "..." if len(final_trans) > 50 else final_trans
                        })
                        await manager.send_msg(client_id, {
                            "status": f"Page {page_num+1}: Item {i+1}/{len(text_items)} | Words: {word_count} added"
                        })

                    page.insert_text(
                        (trans_x, trans_y),
                        final_trans,
                        fontsize=trans_fs,
                        fontname=used_font,
                        color=text_color
                    )
                except Exception as e:
                    logger.warning(f"Failed to insert translation: {e}")
            
            # Save Page to temp file
            temp_name = f"temp_page_{client_id}_{page_num}.pdf"
            temp_path = os.path.join("temp_outputs", temp_name)
            
            # Create a new document with just this page
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_doc.save(temp_path)
            new_doc.close()
            doc.close()
            
            return (page_num, temp_path)
            
        except Exception as e:
            logger.error(f"Page {page_num} Error: {traceback.format_exc()}")
            return (page_num, None)

@app.post("/upload_temp/{client_id}")
async def upload_temp(client_id: str, file: UploadFile = File(...)):
    try:
        path = f"temp_uploads/{client_id}.pdf"
        with open(path, "wb") as f:
            f.write(await file.read())
        return {"status": "ok", "pages": fitz.open(path).page_count}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/preview_page/{client_id}/{page_num}")
async def preview_page(client_id: str, page_num: int, api_key: str = Form(...), target_lang: str = Form("Arabic")):
    try:
        path = f"temp_uploads/{client_id}.pdf"
        if not os.path.exists(path): raise HTTPException(404, "File not found")
        
        doc = fitz.open(path)
        if page_num >= len(doc): raise HTTPException(404, "Page not found")
        page = doc[page_num]
        
        # 1. Render Image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for clarity
        img_data = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        
        # 2. Extract Text & Translate
        # Use simple block extraction for preview
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        texts_to_trans = []
        
        for b in blocks:
            if b["type"] == 0:
                for l in b["lines"]:
                     line_text = "".join([s["text"] for s in l["spans"]]).strip()
                     if line_text:
                         texts_to_trans.append(line_text)
                         # Store ref
                         text_blocks.append({
                             "bbox": list(l["bbox"]),
                             "original": line_text,
                             "fs": l["spans"][0]["size"],
                             "color": l["spans"][0].get("color", 0)
                         })
                         
        # Translate
        ctx = " ".join(texts_to_trans)[:1000]
        # Use existing robust translator
        translations = await translate_batch_robust(texts_to_trans, api_key, target_lang, context=ctx)
        
        # Merge
        final_items = []
        for i, item in enumerate(text_blocks):
            if i < len(translations):
                item["translation"] = translations[i]
                final_items.append(item)
                
        return {"image": f"data:image/png;base64,{img_data}", "items": final_items}
        
    except Exception as e:
        logger.error(f"Preview Error: {e}")
        raise HTTPException(500, str(e))

async def process_pdf_background_task(pdf_bytes: bytes, settings: dict, client_id: str, original_filename: str):
    logger.info(f"Starting V19 Parallel Pipeline for {client_id}")
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(base_dir, "fonts", "Amiri-Regular.ttf")
        
        # Initialize Pipeline
        pipeline = PipelineManager(pdf_bytes, settings, client_id, font_path)
        
        await manager.send_msg(client_id, {"progress": 1, "status": "Initializing Parallel Pipeline..."})
        
        # Start Pipeline (3-Stage Parallel)
        errors = await pipeline.start()
        
        if errors:
            logger.error(f"Pipeline Errors: {errors}")
            
        # Merge Results
        await manager.send_msg(client_id, {"progress": 90, "status": "Merging Final PDF..."})
        
        final_doc = fitz.open()
        
        valid_pages = 0
        for i in range(pipeline.total_pages):
            p_path = f"temp_outputs/v19_line_p{i}.pdf"
            if os.path.exists(p_path):
                try:
                    temp_doc = fitz.open(p_path)
                    final_doc.insert_pdf(temp_doc)
                    temp_doc.close()
                    os.remove(p_path)
                    valid_pages += 1
                except: pass
        
        if valid_pages == 0:
             raise Exception("Processing failed for all pages. Check logs.")

        filename = f"v19_pro_{original_filename}"
        output_path = f"temp_outputs/{filename}"
        final_doc.save(output_path, garbage=4, deflate=True)
        final_doc.close()

        download_url = f"http://localhost:8000/download/{filename}"
        await manager.send_msg(client_id, {"type": "complete", "download_url": download_url, "filename": filename})
        logger.info(f"V19 Pipeline Complete for {client_id}")

    except Exception as e:
        logger.error(f"FATAL: {traceback.format_exc()}")
        await manager.send_msg(client_id, {"type": "error", "message": f"Global Error: {str(e)}"})

# --- v19.1: CONTEXT-AWARE TRANSLATOR (Solves Literal Translation) ---
async def translate_batch_context_aware(lines: list, api_key: str, target_lang: str, context: str = "", mode: str = "line"):
    """
    New function with strictly enforced Context-First Prompting.
    Replaces translate_batch_robust for Pipeline use.
    """
    if not lines: return []
    
    # 1. Pre-Process Glossary
    glossary_pre = {
        r"\\bAss\\.?\\s*Lec\\.?": "Assistant Lecturer",
        r"\\bAss\\.?\\s*Prof\\.?": "Assistant Professor",
        r"\\bLec\\.?": "Lecturer"
    }
    pre_processed_lines = []
    for line in lines:
        if line:
            clean = str(line)
            try:
                for pattern, replacement in glossary_pre.items():
                    clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
            except: pass
            pre_processed_lines.append(clean)
        else:
            pre_processed_lines.append("")

    loop = asyncio.get_event_loop()
    response_text = None

    try:
         genai.configure(api_key=api_key)
         model = genai.GenerativeModel('gemini-1.5-flash')
         
         if mode == "word":
             gemini_prompt = f"""
             SYSTEM: You are an Expert Contextual Translator (Arabic Native).
             TASK: Translate the tokens to ARABIC, but YOU MUST UNDERSTAND THE CONTEXT FIRST.
             
             PROCESS (STRICT):
             1.  **RECONSTRUCT**: Read the tokens to form the full English sentence.
             2.  **UNDERSTAND**: Identify the meaning (e.g., "Ass. Lec." -> "مدرس مساعد").
             3.  **TRANSLATE**: Translate the *meaning* of the full sentence.
             4.  **MAP**: Project back to the array slots.
             
             RULES:
             - Output MUST be a JSON Array of strings.
             - Length MUST match input exactly.
             - ABSOLUTELY NO ENGLISH IN OUTPUT.
             
             EXAMPLES:
             Input: ["Energy", "is", "defined", "as"]
             Output: ["الطاقة", "هي", "تعرف", "بأنها"]
             
             Input: ["Ass.", "Lec.", "Maher"]
             Output: ["مدرس", "مساعد", "ماهر"]
             
             INPUT ARRAY:
             {json.dumps(pre_processed_lines, ensure_ascii=False)}
             
             OUTPUT JSON ARRAY:
             """
         else:
             gemini_prompt = f"""
             SYSTEM: You are a Token-Level Arabic Translator value mapper.
             TASK: Map each English token index to its Arabic meaning.
             
             INSTRUCTIONS:
             1. Receive an Input Array of English Tokens (indexed 0 to N).
             2. Return a JSON Array of OBJECTS.
             3. Structure: {{"i": Index, "t": "Arabic Meaning"}}
             4. **CRITICAL**: The translation "t" MUST be the meaning of the word at index "i", NOT the word that comes first in Arabic grammar.
             
             EXAMPLE:
             Input: ["Medical", "Physics"]
             (0: Medical, 1: Physics)
             
             Output: [
               {{"i": 0, "t": "الطبية"}},  <-- Meaning of "Medical"
               {{"i": 1, "t": "الفيزياء"}} <-- Meaning of "Physics"
             ]
             (Note: Even though "الفيزياء الطبية" is the correct phrase, you MUST map index 0 to "Medical"'s meaning "الطبية").
             
             CONTEXT: "{context[:500]}..."
             
             BOX INPUT ARRAY:
             {json.dumps(pre_processed_lines, ensure_ascii=False)}
             
             OUTPUT JSON:
             """
             
         response = await loop.run_in_executor(None, lambda: model.generate_content(gemini_prompt))
         response_text = response.text
         
    except Exception as e:
         logger.warning(f"Context Translator Failed: {e}")
         response_text = None

    final_results = [""] * len(pre_processed_lines)
    
    if response_text:
        cleaned = re.sub(r'```json\n?|\n?```', '', response_text).strip()
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match: cleaned = match.group(0)
        
        try:
            results_obj = json.loads(cleaned)
            if isinstance(results_obj, list):
                 for obj in results_obj:
                     idx = obj.get("i")
                     trans = obj.get("t", "")
                     if isinstance(idx, int) and 0 <= idx < len(final_results):
                         final_results[idx] = trans
            
            # fill missing with fallback? 
            # Or assume handled.
        except: pass

    # Robust Fallback (Google)
    if final_results == pre_processed_lines:
        try:
            translator = GoogleTranslator(source='auto', target='ar')
            res = await loop.run_in_executor(None, translator.translate_batch, pre_processed_lines)
            if res and len(res) == len(pre_processed_lines): return res
        except: pass
        
    return final_results

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"temp_outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/translate/{client_id}")
async def translate_pdf(
    client_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(""),
    target_lang: str = Form("Arabic"),
    text_color: str = Form("#FF0000"),
    placement: str = Form("below"),
    hide_original: bool = Form(False),
    translation_mode: str = Form("line"),
    fs_scale: float = Form(1.0),
    y_offset: float = Form(-9.0),
    alignment: str = Form("auto")
):
    try:
        print(f"DEBUG: Endpoint Hit for {client_id}")
        logger.info(f"DEBUG: Endpoint Hit for {client_id}")
        content = await file.read()
        settings = {
            "api_key": api_key, "target_lang": target_lang,
            "text_color": text_color, "placement": placement,
            "hide_original": hide_original, "translation_mode": translation_mode,
            "fs_scale": fs_scale, "y_offset": y_offset, "alignment": alignment
        }
        background_tasks.add_task(process_pdf_background_task, content, settings, client_id, file.filename)
        return JSONResponse({"message": "Processing started", "status": "accepted"})
    except Exception as e:
        logger.error(f"Endpoint Error: {traceback.format_exc()}")
        with open("d:\\app_file\\translat\\backend\\endpoint_error.txt", "w") as f:
            f.write(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Preload NLLB Engine on Startup"""
    logger.info("--- NLLB SYSTEM STARTUP ---")
    logger.info("Initializing NLLB Engine (Background Download)...")
    
    def preload():
        try:
            # Shared instance on PipelineManager
            # We attach it to the class for global access
            PipelineManager.shared_nllb = NLLBEngine(device='cpu') 
            PipelineManager.shared_nllb.load_model() # Triggers download if needed
            logger.info("NLLB Engine Ready!")
        except Exception as e:
            logger.error(f"NLLB Preload Failed: {e}")

    # --- SERVE REACT FRONTEND (PRODUCTION) ---
    # --- SERVE REACT FRONTEND (PRODUCTION) ---
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    # Mount the 'dist' folder (Result of npm run build)
    frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
    
    if os.path.exists(frontend_dist):
        logger.info(f"Frontend Dist Found at: {frontend_dist}")
        logger.info(f"Dist Contents: {os.listdir(frontend_dist)}")
        assets_path = os.path.join(frontend_dist, "assets")
        if os.path.exists(assets_path):
             logger.info(f"Assets Contents: {os.listdir(assets_path)}")
             # Mount assets specifically (highest priority for static files)
             app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
        else:
             logger.warning(f"ASSETS DIRECTORY MISSING: {assets_path}")
             # Create empty assets dir to prevent crash if logic depends on it later? 
             # No, just don't mount.

        
        # Catch-All for React (Return index.html for everything else)
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Check if file exists in dist (e.g. vite.svg, favicon.ico)
            file_path = os.path.join(frontend_dist, full_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                return FileResponse(file_path)
            
            # Otherwise return index.html (SPA Routing)
            return FileResponse(os.path.join(frontend_dist, "index.html"))

    import threading
    t = threading.Thread(target=preload, daemon=True)
    t.start()
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)