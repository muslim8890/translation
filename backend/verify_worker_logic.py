import asyncio

class MockEngine:
    def translate_batch(self, texts, target_lang="Arabic"):
        # Simple mock
        if len(texts) == 1 and "Energy is defined" in texts[0]:
             return ["الطاقة تعريفها باعتبارها القدرة على القيام بعمل"]
        
        # Word mode mock
        mapping = {"Energy": "طاقة", "is": "يكون", "defined": "محدد"}
        return [mapping.get(t, t) for t in texts]

async def test_worker_logic(mode):
    print(f"\n--- TESTING MODE: {mode} ---")
    settings = {"translation_mode": mode}
    engine = MockEngine()
    
    texts = ["Energy", "is", "defined"]
    target_lang = "Arabic"
    final_translations = []

    # LOGIC FROM MAIN.PY
    if mode == "word":
         print("LOGIC: executing WORD path")
         translations = engine.translate_batch(texts, target_lang=target_lang)
         final_translations = translations
    else:
         print("LOGIC: executing LINE path")
         valid_indices = [i for i, x in enumerate(texts) if x and x.strip()]
         valid_texts = [texts[i] for i in valid_indices]
         
         final_translations = [""] * len(texts)
         
         if valid_texts:
             joined_text = " ".join(valid_texts)
             print(f"DEBUG: Joined Text: {joined_text}")
             res_joined = engine.translate_batch([joined_text], target_lang=target_lang)
             
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
                         final_translations[original_idx] = " ".join(chunk)

    print(f"RESULT: {final_translations}")

async def run():
    await test_worker_logic("word")
    await test_worker_logic("line")

if __name__ == "__main__":
    asyncio.run(run())
