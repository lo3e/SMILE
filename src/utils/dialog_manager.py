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

'''def build_llm_prompt(
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

def filter_memory_for_prompt(state, last_user_message, profile, recent_conv, episodes):
    """
    Memory gate robusto, compatibile con l'implementazione attuale.
    Gestisce:
    - GREETING → nessuna memoria
    - primi due turni → nessuna memoria
    - cambio argomento → memoria soppressa
    - temi proibiti → memoria soppressa
    - episodi rilevanti → SOLO se pertinenti
    - fallback sicuro
    """

    msg = last_user_message.lower().strip()

    # ============================
    #  0. FLAG DI OUTPUT
    # ============================
    suppress_all = False

    # ============================
    #  1. GREETING → memoria OFF
    # ============================
    if state == "GREETING":
        suppress_all = True
        return {
            "profile": None,
            "episodes": [],
            "recent_conv": [],
            "suppress_all": True
        }

    # ============================
    #  2. Primi due turni → OFF
    # ============================
    if len(recent_conv) < 2:
        suppress_all = True
        return {
            "profile": None,
            "episodes": [],
            "recent_conv": [],
            "suppress_all": True
        }

    # ============================
    #  3. Rileva cambi tema espliciti
    # ============================
    # Se l’utente *dichiara* di voler cambiare argomento → memoria OFF
    switches = [
        "non voglio parlare",
        "parliamo di",
        "cambiamo argomento",
        "preferisco parlare di",
        "parliamo d'altro",
        "non parlare di"
    ]
    if any(sw in msg for sw in switches):
        suppress_all = True
        return {
            "profile": None,
            "episodes": [],
            "recent_conv": [],
            "suppress_all": True
        }

    # ============================
    #  4. Tema nuovo / non correlato
    # ============================
    # Rilevazione robusta di cambio contesto
    context_keywords = {
        "sport": ["sport", "calcio", "palestra", "allenamento"],
        "arte": ["arte", "museo", "dipinto", "mostra"],
        "cibo": ["cibo", "mangiare", "ristorante", "colazione", "cena"],
        "meteo": ["meteo", "tempo", "freddo", "caldo", "piove"],
        "studio": ["studio", "dottorato", "tesi", "ricerca", "università"],
        "viaggi": ["parigi", "viaggio", "treno", "aereo", "weekend"]
    }

    detected_categories = set()

    for cat, kws in context_keywords.items():
        if any(kw in msg for kw in kws):
            detected_categories.add(cat)

    # Se il messaggio tocca UNA sola categoria → trattiamo come argomento corrente.
    # Se tocca ZERO categorie → nuovo tema → memoria OFF
    if len(detected_categories) == 0:
        suppress_all = True
        return {
            "profile": None,
            "episodes": [],
            "recent_conv": [],
            "suppress_all": True
        }

    # ============================
    #  5. Episodi pertinenti SOLO per parole chiave corrispondenti
    # ============================
    relevant_eps = []
    for ep in episodes:
        summary = ep.get("summary", "").lower()
        if any(token in summary for token in msg.split()):
            relevant_eps.append(ep)

    # ============================
    #  6. Limitare Working Memory
    # ============================
    limited_recent = recent_conv[-3:] if recent_conv else []

    return {
        "profile": profile,
        "episodes": relevant_eps,
        "recent_conv": limited_recent,
        "suppress_all": suppress_all
    }
'''

def build_llm_prompt(
    user_name: str,
    user_text: str,
    is_first_turn: bool = False,
    state: str = "FREE_TALK",
):
    profile = load_profile(user_name)
    history = load_recent_history(user_name, window=7)

    # ============================
    #   PROFILO FILTRATO
    # ============================
    age = profile.get("age", "sconosciuta")
    gender = profile.get("gender", "sconosciuto")
    occupation = profile.get("occupation", "sconosciuta")
    episodes = profile.get("episodes", [])

    interests = profile.get("interests", [])
    if isinstance(interests, list):
        interests = interests[:3]
    else:
        interests = [interests]

    goals = profile.get("goals", [])
    if isinstance(goals, list):
        goals = goals[:2]
    else:
        goals = [goals]

    profile_stable_txt = (
        f"- Età stimata: {age}\n"
        f"- Genere: {gender}\n"
        f"- Occupazione: {occupation}\n"
        f"- Interessi principali: {', '.join(interests) if interests else 'N/A'}\n"
        f"- Obiettivi personali: {', '.join(goals) if goals else 'N/A'}"
    )

    # ============================
    #   EPISODI RECENTI
    # ============================
    recent_eps = episodes[-3:] if episodes else []

    if recent_eps:
        episodic_summary = "\n".join(
            [
                f"- {ep.get('title','(senza titolo)')} ({ep.get('timeframe','?')})\n"
                f"  Riassunto: {ep.get('summary','')}"
                for ep in recent_eps
            ]
        )
    else:
        episodic_summary = "Nessun episodio registrato recentemente."

    # ============================
    #   HISTORY (working memory)
    # ============================
    history_txt = format_history_for_prompt(history)

    # ============================
    #   NOTES SUMMARY (FILTRATO)
    # ============================
    notes_summary = profile.get("notes_summary", "").strip()
    if not notes_summary:
        notes_summary = "Nessun riassunto disponibile."

    # ============================
    #   STAGE DESCRIPTION
    # ============================
    if state == "GREETING":
        stage = (
            "È l'inizio della conversazione. Puoi iniziare con un saluto naturale "
            "('Ciao' o il nome dell'utente VA BENE solo in questo stato)."
        )
    elif state == "FAREWELL":
        stage = (
            "La conversazione sta terminando. Dai un saluto caldo e breve, "
            "senza porre domande."
        )
    else:  # FREE_TALK o altri stati
        stage = (
            "La conversazione è in corso. NON iniziare la risposta con 'Ciao' o il nome. "
            "Se l'utente dice 'ciao', trattalo come parte naturale del discorso."
        )

    # ============================
    #   PROMPT FINALE
    # ============================
    prompt = f"""
Tu sei Robot, un assistente robotico sociale. 
Parli in italiano, con un tono caldo, semplice e naturale. 
Il tuo obiettivo è sostenere la conversazione in modo chiaro, rispettoso, non invadente.

Oggi è: {datetime.now().strftime('%A %d %B %Y, %H:%M')}.
STATO ATTUALE: {state.upper()}
{stage}

====================================================================
📌 CONTESTO E IDENTITÀ
====================================================================
Stai parlando con **{user_name}**, una persona che conosci da precedenti interazioni.  
Queste sono informazioni stabili che hai su di lui/lei:

{profile_stable_txt}

====================================================================
📘 MEMORIA PERSONALE (profilo)
====================================================================
È una memoria “passiva”:  
- usala SOLO se chiaramente pertinente a ciò che l’utente sta dicendo ora  
- non introdurre tu nuovi temi basandoti sul profilo  
- non costruire inferenze emotive o interpretative  
- non usare il profilo per cambiare argomento  

====================================================================
🧠 EPISODI della MEMORIA (episodic memory)
====================================================================
Questi sono i ricordi episodici registrati in passato:

{episodic_summary}

ISTRUZIONI PER L’USO:
- NON citarli spontaneamente.
- Se trovi una somiglianza chiara tra il messaggio attuale dell'utente e uno o più episodi:
    → puoi richiamarli brevemente, in modo naturale.
- Se non vi è alcuna somiglianza, ignorali completamente.

====================================================================
💬 MEMORIA RECENTE (working memory)
====================================================================
Questi sono gli ultimi turni della conversazione:

{history_txt}

ISTRUZIONI:
- Questa memoria ha priorità massima.
- Mantieni la coerenza con gli ultimi scambi.
- Se l’utente dice qualcosa simile a ciò che ha detto pochi turni fa,
  puoi riprendere quel riferimento.

====================================================================
⚠️ COMPORTAMENTO
====================================================================
- Rispondi con frasi brevi (1–3 frasi).
- Non attribuire emozioni, intenzioni o stati d’animo che l'utente non ha espresso.
- Se una parola sembra frutto di un errore di pronuncia/ASR → chiedi conferma (“Intendi X?”).
- Non inventare ricordi o collegamenti non presenti.
- NON iniziare la risposta con “Ciao” o il nome dell’utente, tranne nello stato GREETING.
- In FREE_TALK: i saluti come “ciao” sono parte naturale del discorso, NON trattarli come nuovo saluto.
- In FAREWELL: dai un saluto breve e non fare domande.
- Se l’utente cambia argomento, segui il nuovo argomento senza riprendere memorie precedenti.

====================================================================
🗣️ MESSAGGIO ATTUALE DELL’UTENTE
====================================================================
{user_text}

--------------------------------------------------------------------
🎯 RISPOSTA RICHIESTA
--------------------------------------------------------------------
Rispondi come Robot, in modo caldo, naturale e pertinente al messaggio dell’utente.
Evita interpretazioni non dette. Usa la memoria solo quando è applicabile.
"""

    return prompt.strip()


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
                # aggiorna il profilo su disco (via memory_manager)
                update_episodes(name, episodes)
            except Exception as e:
                print(f"[MEMORY] Errore aggiornando gli episodi: {e}")

            # 🔧 allinea anche il profilo locale, così non li perdiamo
            existing_eps = profile.get("episodes", []) or []
            for ep in episodes:
                if ep not in existing_eps:
                    existing_eps.append(ep)
            profile["episodes"] = existing_eps

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

