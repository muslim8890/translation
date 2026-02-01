import os
import sentencepiece as spm

try:
    path = "d:/app_file/translat/backend/models/nllb-200-ct2/sentencepiece.bpe.model"
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    
    tokens = sp.encode("eng_Latn", out_type=str)
    
    with open("sp_debug.txt", "w") as f:
        f.write(f"Tokens: {tokens}\n")
        
        if len(tokens) == 1 and tokens[0] == "eng_Latn":
             f.write("STATUS: SINGLE_TOKEN")
        else:
             f.write("STATUS: SPLIT_TOKEN")

except Exception as e:
    with open("sp_debug.txt", "w") as f:
         f.write(f"ERROR: {e}")
