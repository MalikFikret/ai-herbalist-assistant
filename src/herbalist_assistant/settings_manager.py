import json
import threading
from pathlib import Path
from typing import Any

from herbalist_assistant import config

_SETTINGS_FILE = Path("settings.json")
_settings_lock = threading.Lock()
_cache = {}
_cache_valid = False

def _load_settings() -> dict[str, Any]:
    global _cache, _cache_valid
    with _settings_lock:
        if _cache_valid:
            return _cache
            
        if _SETTINGS_FILE.exists():
            try:
                with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    _cache = json.load(f)
                    _cache_valid = True
                    return _cache
            except Exception:
                pass
        
        _cache = {}
        _cache_valid = True
        return _cache

def get_setting(key: str) -> Any:
    settings = _load_settings()
    if key in settings:
        return settings[key]
    return getattr(config, key)

def save_settings(new_settings: dict[str, Any]) -> None:
    global _cache, _cache_valid
    with _settings_lock:
        current = {}
        if _SETTINGS_FILE.exists():
            try:
                with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    current = json.load(f)
            except Exception:
                pass
                
        current.update(new_settings)
        
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=4)
            
        _cache = current
        _cache_valid = True
