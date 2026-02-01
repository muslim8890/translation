import asyncio
import os
import sys

# Mock NLLB Engine
class MockNLLB:
    def translate_batch(self, texts, src, tgt):
        results = []
        for t in texts:
            if t.strip().startswith("Energy is defined"):
               results.append("الطاقة تعريفها باعتبارها القدرة على القيام بعمل")
            else:
               # Simple word mock
               mock_map = {
                   "Energy": "طاقة", "is": "يكون", "defined": "محدد", 
                   "as": "مثل", "the": "ال", "ability": "قدرة", 
                   "to": "ل", "do": "عمل", "work": "شغل"
               }
               results.append(mock_map.get(t, "AR:" + t))
        return results

# Re-implement the logic function to test it in isolation
async def test_logic(lines, mode):
    pre_processed_lines = lines # simplify for test
    
    mock_nllb = MockNLLB()
    loop = asyncio.get_event_loop()
    
    # LOGIC UNDER TEST
    if mode == "word":
         print("MODE: WORD DETECTED")
         text_inputs = [l if l else "" for l in pre_processed_lines]
         # Use run_in_executor mock
         res = mock_nllb.translate_batch(text_inputs, "english", "arabic")
         return res
         
    else:
         print("MODE: SENTENCE DETECTED")
         valid_indices = [i for i, x in enumerate(pre_processed_lines) if x and x.strip()]
         valid_texts = [pre_processed_lines[i] for i in valid_indices]
         
         final_res = [""] * len(pre_processed_lines)
         
         if valid_texts:
             joined_text = " ".join(valid_texts)
             print(f"DEBUG JOINED: {joined_text}")
             
             res_joined = mock_nllb.translate_batch([joined_text], "english", "arabic")
             
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

async def run_test():
    test_words = ["Energy", "is", "defined", "as", "the", "ability", "to", "do", "work"]
    
    print("\n--- TEST 1: SENTENCE MODE ---")
    res1 = await test_logic(test_words, "line")
    print(f"Result: {res1}")
    
    print("\n--- TEST 2: WORD MODE ---")
    res2 = await test_logic(test_words, "word")
    print(f"Result: {res2}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_test())
