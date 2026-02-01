import os
import sentencepiece as spm

model_path = "d:/app_file/translat/backend/models/nllb-200-ct2"
sp_model = os.path.join(model_path, "sentencepiece.bpe.model")
vocab_file = os.path.join(model_path, "shared_vocabulary.txt")

def log(msg):
    print(msg)
    with open("debug_vocab.log", "a") as f:
        f.write(str(msg) + "\n")

if os.path.exists(vocab_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = [l.strip().split()[0] for l in f.readlines()]
    
    log(f"Vocab size: {len(vocab)}")
    if "eng_Latn" in vocab:
        log("SUCCESS: 'eng_Latn' found in shared_vocabulary.txt")
    else:
        log("FAILURE: 'eng_Latn' NOT found in shared_vocabulary.txt")
        
    if "arb_Arab" in vocab:
        log("SUCCESS: 'arb_Arab' found in shared_vocabulary.txt")
    else:
        log("FAILURE: 'arb_Arab' NOT found in shared_vocabulary.txt")

    sp = spm.SentencePieceProcessor()
    sp.load(sp_model)
    
    tokenized_code = sp.encode("eng_Latn", out_type=str)
    log(f"Tokenized 'eng_Latn': {tokenized_code}")
    
    if len(tokenized_code) == 1 and tokenized_code[0] == "eng_Latn":
        log("SPM treats language code as single token.")
    else:
        log("SPM splits language code! This is the problem.")
        
else:
    log("vocab file not found")
