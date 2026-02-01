import os
import logging
import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

logger = logging.getLogger("uvicorn")

class NLLBEngine:
    def __init__(self, model_id="JustFrederik/nllb-200-distilled-600M-ct2-int8", device="cpu"):
        self.model_id = model_id
        self.device = device
        self.model_path = os.path.join(os.getcwd(), "models", "nllb-200-ct2")
        self.sp_model_path = os.path.join(self.model_path, "sentencepiece.bpe.model")
        self.translator = None
        self.sp = None
        
        # Mapping for common language names to NLLB codes
        self.lang_map = {
            "arabic": "arb_Arab", "ar": "arb_Arab",
            "english": "eng_Latn", "en": "eng_Latn",
            "french": "fra_Latn", "fr": "fra_Latn",
            "spanish": "spa_Latn", "es": "spa_Latn",
            "german": "deu_Latn", "de": "deu_Latn",
            "italian": "ita_Latn", "it": "ita_Latn",
            "portuguese": "por_Latn", "pt": "por_Latn",
            "russian": "rus_Cyrl", "ru": "rus_Cyrl",
            "chinese": "zho_Hans", "zh": "zho_Hans",
            "japanese": "jpn_Jpan", "ja": "jpn_Jpan",
            "korean": "kor_Hang", "ko": "kor_Hang",
            "turkish": "tur_Latn", "tr": "tur_Latn",
            "hindi": "hin_Deva", "hi": "hin_Deva",
            "dutch": "nld_Latn", "nl": "nld_Latn"
        }

    def load_model(self):
        """Downloads and loads the NLLB model."""
        if self.translator:
            return

        logger.info(f"Loading NLLB Engine ({self.model_id})...")
        
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading model to {self.model_path}...")
            os.makedirs(self.model_path, exist_ok=True)
            try:
                snapshot_download(repo_id=self.model_id, local_dir=self.model_path, local_dir_use_symlinks=False, token=False)
                logger.info("Download Complete.")
            except Exception as e:
                logger.error(f"Failed to download NLLB model: {e}")
                raise e

        # Load Tokenizer (SentencePiece)
        # Note: Some CT2 conversions include spm model, some don't. 
        # The softcatala repo usually includes 'sentencepiece.bpe.model' or similar.
        # We need to find it.
        sp_candidates = [f for f in os.listdir(self.model_path) if "sentencepiece" in f or "spm" in f]
        if not sp_candidates:
             # Fallback: Download generic NLLB-200 SP model? Or fail.
             # Trying to find specifically 'sentencepiece.bpe.model'
             logger.error("SentencePiece model not found in download.")
             # For robustness, we might need a dedicated SP download if the repo is weird.
             # But softcatala usually has it.
             raise FileNotFoundError("SentencePiece model missing")
        
        self.sp_model_path = os.path.join(self.model_path, sp_candidates[0])
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.sp_model_path)
        
        # Load CTranslate2 Translator
        # Load CTranslate2 Translator
        # PERFORMANCE FIX: Set intra_threads=1 to prevent CPU starvation when running multiple workers.
        # This allows running 10 parallel workers without freezing the server.
        self.translator = ctranslate2.Translator(
            self.model_path, 
            device=self.device,
            device_index=[0], 
            compute_type="int8", 
            inter_threads=1, 
            intra_threads=1
        )
        logger.info("NLLB Engine Loaded Successfully.")

    def translate_batch(self, texts, source_lang="english", target_lang="arabic"):
        if not texts: return []
        if not self.translator: self.load_model()
        
        src_code = self.lang_map.get(str(source_lang).lower(), "eng_Latn")
        tgt_code = self.lang_map.get(str(target_lang).lower(), "ary_Arab")
        
        # 1. Tokenize
        # NLLB expects standard SP tokenization.
        # Ideally, we should add the language token, but CT2 handles target prefix.
        # For source, NLLB usually pre-pends source token? 
        # Actually, for NLLB-200 with CT2, we usually need to handle tokens manually unless using transformers.
        # BUT, check softcatala README. Usually:
        # source_tokens = [sp.encode(text, out_type=str) for text in texts]
        # target_prefix = [tgt_code]
        
        # 1. Tokenize
        # NLLB requires source language code at start and EOS at end
        source_tokens = []
        for text in texts:
            tokens = self.sp.encode(text, out_type=str)
            tokens.insert(0, src_code)
            tokens.append("</s>") # EOS token
            source_tokens.append(tokens)
            
        logger.info(f"DEBUG NLLB Tokens: {source_tokens[0][:5]}...") 

        # 2. Translate
        # We must provide target_prefix for NLLB to know the target language
        results = self.translator.translate_batch(
            source_tokens,
            target_prefix=[[tgt_code]] * len(texts),
            beam_size=1 # Greed search for speed
        )
        
        # 3. Detokenize
        translated_texts = []
        for res in results:
            # res.hypotheses is list of list of tokens. We take top 1.
            hyp_tokens = res.hypotheses[0]
            # Detokenize
            detok = self.sp.decode(hyp_tokens)
            # Remove the target language code if present (e.g. arb_Arab)
            detok = detok.replace(tgt_code, "").strip()
            translated_texts.append(detok)
            
        return translated_texts
