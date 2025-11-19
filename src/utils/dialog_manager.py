# utils/dialog_manager.py
import requests
import json
from datetime import datetime

# 🔴 IMPORT GIUSTI
# prendi il profilo e la history SOLO da profile_manager
from src.utils.profile_manager import (
    load_profile, save_profile,
    load_recent_history,
    format_profile_for_prompt,
    format_history_for_prompt,
)
from src.utils.memory_manager import update_episodes

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def build_llm_prompt(
    user_name: str,
    user_text: str,
    is_first_turn: bool = False,
    state: str = "FREE_TALK",
) -> str:
    profile = load_profile(user_name)
    history = load_recent_history(user_name, window=7)

    profile_txt = format_profile_for_prompt(profile)
    history_txt = format_history_for_prompt(history)
    notes_summary = profile.get("notes_summary", "").strip() or "(nessuna memoria episodica disponibile)"

    if state == "GREETING":
        stage = "È l'inizio della conversazione. Puoi salutare brevemente e introdurti con naturalezza."
    elif state == "FAREWELL":
        stage = "La conversazione sta per terminare. Rispondi con un saluto finale caldo e coerente, non riaprire."
    else:
        stage = "La conversazione è già in corso. NON salutare di nuovo."

    prompt = f"""
Tu sei "Robot", un assistente robotico che parla in italiano, con tono caldo e naturale.
Parli con {user_name}, che conosci.

STATO ATTUALE: {state.upper()}
{stage}

⚠️ REGOLE ⚠️
- NON iniziare ogni risposta con "Ciao" o con il nome, tranne nel primissimo turno.
- Se lo stato è FREE_TALK, considera i "ciao" come parte della conversazione, non come inizio.
- Se lo stato è FAREWELL, fai solo un saluto e NON fare domande.
- Risposte brevi: 2-3 frasi.

📘 MEMORIA A LUNGO TERMINE:
{profile_txt}

🧠 MEMORIA EPISODICA (riassunti precedenti):
{notes_summary}

💬 MEMORIA A BREVE TERMINE (ultimi 7 turni):
{history_txt}

🗣️ INPUT UTENTE:
{user_text}

Rispondi come "Robot":
Robot:
""".strip()

    return prompt


def ask_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    data = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=data)
    resp.raise_for_status()
    return resp.json().get("response", "")


def ask_ollama_with_context(
    user_name: str,
    user_text: str,
    is_first_turn: bool = False,
    state: str = "FREE_TALK",
) -> str:
    prompt = build_llm_prompt(user_name, user_text, is_first_turn=is_first_turn, state=state)
    return ask_ollama(prompt)

def summarize_conversation(name, conversation):
    """
    Riassume la conversazione, deduce informazioni sull'utente e aggiorna il profilo.
    Ora integra sesso, età, interessi, tono, personalità e obiettivi.
    """
    try:
        profile = load_profile(name)

        # --- 1. Prepara testo conversazione ---
        dialogue_text = "\n".join(
            [f"Utente: {x['user']}\nAssistente: {x['bot']}" for x in conversation]
        )

        # --- 2. Prepara prompt ---
        prompt = f"""
Tu sei un sistema di memoria conversazionale. Riceverai:
1. Il profilo attuale dell'utente (potenzialmente incompleto)
2. La trascrizione dell'ultima conversazione

Il tuo compito è aggiornare il profilo in modo coerente e verosimile,
deducendo solo ciò che emerge chiaramente.

Profili incompleti vanno completati solo se ci sono indizi solidi.

In aggiunta, estrai eventuali EPISODI (memoria episodica vera):
- un episodio è un'esperienza, attività, evento o contesto specifico menzionato dall'utente
- deve essere qualcosa di concreto, singolo, situato in un contesto ("ieri", "ultimamente", "sto lavorando a...")

Ogni episodio deve contenere:
- title (breve titolo)
- timeframe (quando è avvenuto o in che periodo è rilevante)
- summary (1–2 frasi che descrivono l'episodio)
- tags (lista di parole chiave)

Se non ci sono episodi, restituisci episodes: [].

=== PROFILO ATTUALE ===
{json.dumps(profile, ensure_ascii=False, indent=2)}

=== CONVERSAZIONE ===
{dialogue_text}

Ora restituisci in formato JSON:
- summary: breve riassunto dell’interazione (3-4 frasi)
- gender: "maschio", "femmina" o null se non deducibile
- age: fascia d’età stimata (es. "20-30", "30-40") o null se non chiaro
- occupation: eventuale professione o ambito lavorativo se emerge
- interests: elenco sintetico di temi o hobby citati
- personality: tratti comportamentali (es. curioso, empatico, analitico)
- goals: obiettivi personali o professionali se emergono
- episodes: lista di episodi rilevanti (può essere vuota)
        """

        response = ask_ollama(prompt, model=MODEL_NAME)

        # --- 3. Parsa output LLM ---
        try:
            data = json.loads(response)
        except Exception:
            # fallback: tenta di isolare il blocco JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end]) if start != -1 and end != -1 else {}

        # --- 4. Merge intelligente ---
        profile["notes_summary"] = data.get("summary", profile.get("notes_summary", ""))

        # aggiorna solo se mancante o migliorabile
        for key in ["gender", "age", "occupation", "personality"]:
            val = data.get(key)
            if val and (profile.get(key) in [None, ""]):
                profile[key] = val

        # merge di liste senza duplicati
        def merge_list(a, b):
            return list(set((a or []) + (b or [])))

        profile["interests"] = merge_list(profile.get("interests", []), data.get("interests", []))
        profile["goals"] = merge_list(profile.get("goals", []), data.get("goals", []))

        # --- EPISODI ---
        episodes = data.get("episodes", [])
        if isinstance(episodes, list) and episodes:
            try:
                update_episodes(name, episodes)
            except Exception as e:
                print(f"[MEMORY] Errore aggiornando gli episodi: {e}")

        # Salva anche le ultime conversazioni recenti
        profile["recent_conversations"] = conversation[-5:]
        profile["last_update"] = datetime.now().isoformat()

        # --- 5. Salva ---
        save_profile(name, profile)
        print(f"[MEMORY] ✅ Profilo di {name} aggiornato correttamente con nuove informazioni.")
        return profile

    except Exception as e:
        print(f"[MEMORY] Errore durante il riassunto: {e}")
        return None


