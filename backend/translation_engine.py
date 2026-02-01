import asyncio
import json
import logging
import re
import google.generativeai as genai
from database import get_cached_translation, save_to_cache

logger = logging.getLogger("uvicorn")

class TranslationEngine:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.key_count = len(api_keys)
        self.current_key_idx = 0
        
    def get_next_key(self):
        key = self.api_keys[self.current_key_idx]
        self.current_key_idx = (self.current_key_idx + 1) % self.key_count
        return key

    def group_words_into_sentences(self, words):
        """
        Groups raw word tuples from PyMuPDF into full semantic sentences.
        Word format: (x0, y0, x1, y1, text, block_no, line_no, word_no)
        """
        sentences = []
        current_sentence_words = []
        current_text_buffer = ""
        
        # Sort mainly by block, then line, then word index implied by order
        # Assuming words come in reading order roughly
        
        for w in words:
            text = w[4]
            current_sentence_words.append(w)
            current_text_buffer += text + " "
            
            # Simple heuristic: Split on period, question mark, exclamation
            # AND ensure the period isn't part of a known abbreviation like "Ass." or "Lec." or "Prof."
            if text.endswith(".") or text.endswith("?") or text.endswith("!"):
                # Check abbreviations
                lower_text = text.lower()
                if lower_text in ["ass.", "lec.", "prof.", "dr.", "mr.", "mrs.", "ms.", "eng.", "dept."]:
                    continue
                
                # Check if it looks like a list item "1." or "a."
                if re.match(r"^\d+\.$", text) or re.match(r"^[a-zA-Z]\.$", text):
                    continue

                # End of sentence
                sentences.append({
                    "text": current_text_buffer.strip(),
                    "words": current_sentence_words
                })
                current_sentence_words = []
                current_text_buffer = ""
        
        # Flush remaining
        if current_sentence_words:
            sentences.append({
                "text": current_text_buffer.strip(),
                "words": current_sentence_words
            })
            
        return sentences

    async def translate_sentences(self, sentences, target_lang="Arabic"):
        """
        Translates a list of sentence objects.
        Returns the same list with 'translation' and 'word_map' added.
        """
        if not sentences: return []
        
        loop = asyncio.get_event_loop()
        tasks = []
        
        for sent in sentences:
            tasks.append(self._process_single_sentence(sent, target_lang, loop))
            
        return await asyncio.gather(*tasks)

    async def _process_single_sentence(self, sent, target_lang, loop):
        original_text = sent["text"]
        words = sent["words"]
        tokens = [w[4] for w in words]
        
        api_key = self.get_next_key()
        
        try:
            # Construct Prompt
            prompt_text = f"""
            SYSTEM: You are an Academic Translator.
            TASK: Translate the following tokens to {target_lang} while preserving the One-to-One mapping.
            
            METHODOLOGY:
            1. Read the full token sequence to understand the context.
            2. Translate the *meaning* of the sentence.
            3. Map the translation back to the individual tokens.
            4. **ABBREVIATIONS**: 
               - "Ass." + "Lec." -> "مدرس" + "مساعد"
               - "Prof." -> "أستاذ"
            
            INPUT TOKENS: {json.dumps(tokens)}
            
            OUTPUT FORMAT:
            Return ONLY a JSON Array of strings. Length MUST match input.
            """

            # Direct REST API Call (Thread-Safe)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"response_mime_type": "application/json"}
            }
            
            def call_api():
                import requests
                return requests.post(url, json=payload, timeout=10)
            
            response = await loop.run_in_executor(None, call_api)
            
            if response.status_code != 200:
                logger.error(f"Gemini API Error {response.status_code}: {response.text}")
                sent["translations"] = tokens
                return sent
                
            response_json = response.json()
            try:
                response_text = response_json["candidates"][0]["content"]["parts"][0]["text"]
            except:
                logger.error(f"Gemini Bad Response Structure: {response_json}")
                sent["translations"] = tokens
                return sent
                
            logger.info(f"GEMINI RAW: {response_text[:100]}...") # Debug

            # Parse JSON
            cleaned = re.sub(r'```json\n?|\n?```', '', response_text).strip()
            match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            
            if match:
                translations = json.loads(match.group(0))
                if len(translations) == len(tokens):
                    sent["translations"] = translations
                else:
                    logger.warning(f"Len Mismatch inputs={len(tokens)} outputs={len(translations)}")
                    # Fix Mismatch: Pad or Truncate
                    final = translations[:len(tokens)]
                    if len(final) < len(tokens):
                        final += [""] * (len(tokens) - len(final))
                    sent["translations"] = final
            else:
                 logger.error(f"No JSON in text: {cleaned}")
                 sent["translations"] = tokens

        except Exception as e:
            logger.error(f"Translation Exception: {e}")
            sent["translations"] = tokens
            
        return sent
