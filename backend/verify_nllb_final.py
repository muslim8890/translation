import os
import sys

# Add backend to path
sys.path.append(os.getcwd())

from nllb_engine import NLLBEngine

def log(msg):
    print(msg, flush=True)
    with open("final_verification.log", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
        f.flush()
        os.fsync(f.fileno())

model_path = "d:/app_file/translat/backend/models/nllb-200-ct2/model.bin"

if os.path.exists(model_path):
    size = os.path.getsize(model_path)
    log(f"Model file found. Size: {size} bytes")
    if size < 500000000: # Less than 500MB
        log("WARNING: Model file seems too small! (Expected ~600MB)")
    else:
        log("Model file size looks correct.")
else:
    log("ERROR: Model file NOT found!")
    sys.exit(1)

log("Initializing NLLB Engine...")
try:
    engine = NLLBEngine(device='cpu')
    engine.load_model()
    log("Engine loaded.")
    
    # TEST 1: Full Sentence (Checking Line Mode)
    test_sentence = "Energy is defined as the ability to do work"
    log(f"Translating Sentence: '{test_sentence}'")
    res_sentence = engine.translate_batch([test_sentence], target_lang="Arabic")
    log(f"Result Sentence: {res_sentence}")

    # TEST 2: Word List (Checking Word Mode Failure)
    test_words = ["Energy", "is", "defined", "as", "the", "ability", "to", "do", "work"]
    log(f"Translating Word List: {test_words}")
    res_words = engine.translate_batch(test_words, target_lang="Arabic")
    log(f"Result Word List: {res_words}")
    
    # TEST 3: JOIN-TRANSLATE-DISTRIBUTE STRATEGY
    log("\n--- TEST 3: SMART CONTEXT STRATEGY ---")
    
    # 1. Join
    joined_text = " ".join(test_words)
    log(f"Joined Input: {joined_text}")
    
    # 2. Translate
    joined_res = engine.translate_batch([joined_text], target_lang="Arabic")[0]
    log(f"Translated Sentence: {joined_res}")
    
    # 3. Distribute
    # Split Arabic result
    arabic_words = joined_res.split()
    input_len = len(test_words)
    output_len = len(arabic_words)
    
    final_output = [""] * input_len
    
    # Heuristic distribution
    if output_len > 0:
        ratio = output_len / input_len
        current_arb_idx = 0
        
        for i in range(input_len):
            target_idx = min(int((i + 1) * ratio), output_len)
            chunk = arabic_words[current_arb_idx:target_idx]
            final_output[i] = " ".join(chunk)
            current_arb_idx = target_idx
            
    log(f"Distributed Output ({len(final_output)} items): {final_output}")

except Exception as e:
    log(f"ERROR: {e}")
