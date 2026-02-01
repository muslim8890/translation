import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.getcwd())

# Mock classes to simulate environment without full server
class MockNLLB:
    def translate_batch(self, texts, src, tgt):
        # Simulate NLLB behavior
        # Simple mock: just reverse string or add "AR:" prefix
        results = []
        for t in texts:
            if t == "Energy is defined as the ability to do work":
                results.append("الطاقة تعريفها باعتبارها القدرة على القيام بعمل")
            elif t == "Energy": results.append("طاقة")
            elif t == "is": results.append("يكون")
            elif t == "defined": results.append("محدد")
            elif t == "as": results.append("مثل")
            elif t == "the": results.append("ال")
            elif t == "ability": results.append("قدرة")
            elif t == "to": results.append("ل")
            elif t == "do": results.append("عمل")
            elif t == "work": results.append("شغل")
            else: results.append("AR:" + t)
        return results

class MockPipeline:
    shared_nllb = None

import main
main.PipelineManager = MockPipeline
main.PipelineManager.shared_nllb = MockNLLB()
main.PipelineManager.shared_nllb.translator = True # just to pass check

async def test():
    test_words = ["Energy", "is", "defined", "as", "the", "ability", "to", "do", "work"]
    
    print("\n--- TEST 1: SENTENCE MODE (Default) ---")
    res_sentence = await main.translate_batch_robust(test_words, "key", "arabic", mode="line")
    print(f"Result: {res_sentence}")
    
    print("\n--- TEST 2: WORD MODE ---")
    res_word = await main.translate_batch_robust(test_words, "key", "arabic", mode="word")
    print(f"Result: {res_word}")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test())
