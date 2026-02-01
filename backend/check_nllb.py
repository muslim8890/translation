import sys
print(f"Python: {sys.version}")

try:
    import ctranslate2
    print("CTranslate2: OK")
except ImportError as e:
    print(f"CTranslate2: MISSING ({e})")

try:
    import sentencepiece as spm
    print("SentencePiece: OK")
except ImportError as e:
    print(f"SentencePiece: MISSING ({e})")

# Check if we can instantiate Translator (requires valid path, but check class existence)
try:
    t = ctranslate2.Translator
    print("CTranslate2.Translator class: FOUND")
except Exception as e:
    print(f"CTranslate2 Class Check: FAIL ({e})")
