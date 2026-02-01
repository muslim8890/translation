import sqlite3
import hashlib
import json
import os
import logging

logger = logging.getLogger("uvicorn")

DB_PATH = "translations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS translation_cache (
            source_hash TEXT PRIMARY KEY,
            source_text TEXT,
            arabic_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_cached_translation(source_text: str):
    try:
        source_hash = hashlib.md5(source_text.strip().encode('utf-8')).hexdigest()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT arabic_text FROM translation_cache WHERE source_hash = ?", (source_hash,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"DB Read Error: {e}")
        return None

def save_to_cache(source_text: str, arabic_text: str):
    try:
        block_list = ["Ass.", "Lec.", "Prof."] # Don't cache single abbreviations if they might be part of larger context? 
        # Actually proper sentences should be cached.
        
        source_hash = hashlib.md5(source_text.strip().encode('utf-8')).hexdigest()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO translation_cache (source_hash, source_text, arabic_text) VALUES (?, ?, ?)",
                  (source_hash, source_text, arabic_text))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Write Error: {e}")
