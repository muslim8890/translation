import sys
import os
import logging

# Configure logging to file
logging.basicConfig(filename='debug_result_internal.txt', level=logging.INFO, filemode='w')

def log(msg):
    print(msg)
    logging.info(msg)

log(f"Python executable: {sys.executable}")
log(f"HF_TOKEN present: {'HF_TOKEN' in os.environ}")
log(f"HUGGING_FACE_HUB_TOKEN present: {'HUGGING_FACE_HUB_TOKEN' in os.environ}")

try:
    import ctranslate2
    log(f"ctranslate2 imported successfully {ctranslate2.__version__}")
except ImportError as e:
    log(f"ctranslate2 import failed: {e}")

try:
    import sentencepiece
    log(f"sentencepiece imported successfully {sentencepiece.__version__}")
except ImportError as e:
    log(f"sentencepiece import failed: {e}")

try:
    from nllb_engine import NLLBEngine
    log("NLLBEngine imported successfully")
    engine = NLLBEngine(device='cpu')
    log("Loading model...")
    engine.load_model()
    log("NLLBEngine loaded successfully")
    res = engine.translate_batch(["Hello world"], target_lang="Arabic")
    log(f"Translation: {res}")
except Exception as e:
    import traceback
    err = traceback.format_exc()
    log(f"NLLBEngine test failed: {e}\n{err}")

