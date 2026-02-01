import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.getcwd())

from main import PipelineManager, NLLBEngine
import logging

def append_log(msg):
    with open("stack_test_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
    print(msg)

async def test_stack():
    append_log("--- TESTING NLLB ---")
    try:
        engine = NLLBEngine(device='cpu')
        engine.load_model()
        res = engine.translate_batch(["Hello world"], target_lang="Arabic")
        append_log(f"NLLB Result: {res}")
    except Exception as e:
        append_log(f"NLLB Failed: {e}")

    append_log("\n--- TESTING GEMINI FALLBACK (ALL KEYS) ---")
    import google.generativeai as genai
    
    KEYS = [
        "AIzaSyCV4RzAV0IJkRz_ZKux4kzHm1VvPLguiJI",
        "AIzaSyB14R8e4HKbPnt3c_mAqF77tv3tA4RXJnw",
        "AIzaSyCXOJ0t0vvg-a1_eirE8YowqDLhW-RqP4A",
        "AIzaSyCzF3Bp4Qx8uHmMNEOjT7q0BFAWtcMsN7Y"
    ]
    
    found_good = False
    for i, key in enumerate(KEYS):
        append_log(f"Testing Key {i+1}...")
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = await model.generate_content_async("Translate 'Hello World' to Arabic. Output JSON array of strings: [\"translation\"]")
            append_log(f"SUCCESS Key {i+1}: {response.text}")
            found_good = True
            break
        except Exception as e:
             append_log(f"FAILED Key {i+1}: {e}")

    if not found_good:
        append_log("ALL KEYS FAILED.")


if __name__ == "__main__":
    asyncio.run(test_stack())

