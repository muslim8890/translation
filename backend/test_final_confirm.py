import sys
import os

def append_mark(msg):
    with open("final_status.txt", "a") as f:
        f.write(msg + "\n")

try:
    from nllb_engine import NLLBEngine
    append_mark("IMPORT_OK")
    
    engine = NLLBEngine(device='cpu')
    engine.load_model() # Should be instant if downloaded
    append_mark("LOAD_OK")
    
    res = engine.translate_batch(["Hello world"], target_lang="Arabic")
    append_mark(f"TRANS_OK: {res}")
    
except Exception as e:
    append_mark(f"ERROR: {e}")
