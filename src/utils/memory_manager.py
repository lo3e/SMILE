import os
import json
import pickle
import time
import datetime

from src.config import CONVERSATIONS_DIR, PROFILES_DIR, EMBEDDINGS_FILE

'''
EMB_FILE = "../data/embeddings.pkl"
CONV_DIR = "../data/conversations"
PROFILE_DIR = "../data/profiles"

os.makedirs(CONV_DIR, exist_ok=True)
'''
# ====== GESTIONE VOLTI ======

def save_new_face(name, embedding):
    """Salva un nuovo volto nel database embeddings.pkl."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            known = pickle.load(f)
    else:
        known = {}

    known[name] = embedding
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(known, f)
    print(f"💾 Nuovo volto salvato come '{name}' in embeddings.pkl")


# ====== GESTIONE CONVERSAZIONI ======

def log_full_conversation(name, user_text, bot_reply):
    """Salva ogni scambio in un file JSON per utente."""
    path = os.path.join(CONVERSATIONS_DIR, f"{name}.json")
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_text,
        "bot": bot_reply
    }

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_profile(name: str) -> dict:
    """Carica o crea un profilo per l'utente."""
    path = os.path.join(PROFILES_DIR, f"{name}.json")
    if not os.path.exists(path):
        # nuovo profilo base
        profile = {
            "name": name,
            "age": None,
            "occupation": None,
            "location": None,
            "interests": [],
            "personality": None,
            "goals": [],
            "notes_summary": "",
            "recent_conversations": []
        }
        save_profile(name, profile)
        return profile

    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # retrocompatibilità
    if "notes_summary" not in profile:
        profile["notes_summary"] = ""
    if "recent_conversations" not in profile:
        profile["recent_conversations"] = []

    return profile

def save_profile(name: str, profile: dict):
    """Salva il profilo aggiornato."""
    path = os.path.join(PROFILES_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

def append_conversation(name: str, user_text: str, reply: str):
    """Salva i nuovi scambi nella memoria breve."""
    profile = load_profile(name)
    profile["recent_conversations"].append({
        "input": user_text,
        "reply": reply
    })
    # Mantiene solo gli ultimi 7 turni
    profile["recent_conversations"] = profile["recent_conversations"][-7:]
    save_profile(name, profile)


def update_profile_summary(name: str, new_summary: str):
    """
    Aggiorna il riassunto (notes_summary) integrando un nuovo resoconto.
    Mantiene un testo sintetico e coerente.
    """
    profile = load_profile(name)
    if not profile.get("notes_summary"):
        profile["notes_summary"] = new_summary
    else:
        # combina vecchio e nuovo
        profile["notes_summary"] = f"{profile['notes_summary'].strip()} " \
                                   f"{new_summary.strip()}"

    profile["last_update"] = datetime.datetime.now().isoformat()
    save_profile(name, profile)

def update_episodes(name: str, new_episodes: list):
    """Aggiunge nuovi episodi al profilo, evitando duplicati."""
    profile = load_profile(name)
    eps = profile.get("episodes", [])

    for ep in new_episodes:
        if ep not in eps:
            eps.append(ep)

    profile["episodes"] = eps
    profile["last_update"] = datetime.datetime.now().isoformat()
    save_profile(name, profile)

    print(f"[MEMORY] Episodi aggiornati per {name}: {len(new_episodes)} nuovi episodi aggiunti.")
