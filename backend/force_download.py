import logging
import time
import sys
from nllb_engine import NLLBEngine

# Configure logging to see download progress
with open("download_status.log", "w") as f:
    f.write("STARTING DOWNLOAD...\n")

try:
    engine = NLLBEngine(device='cpu')
    engine.load_model()
    with open("download_status.log", "a") as f:
        f.write("DOWNLOAD COMPLETE: Success\n")
    
    # Test Translation
    res = engine.translate_batch(["Hello World"], target_lang="arabic")
    with open("download_status.log", "a") as f:
        f.write(f"TEST SUCCESS: {res}\n")
    
except Exception as e:
    with open("download_status.log", "a") as f:
        f.write(f"CRITICAL FAILURE: {e}\n")
    import traceback
    traceback.print_exc()
