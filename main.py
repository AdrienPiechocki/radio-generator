import sys
import asyncio
import edge_tts
import re
import ollama
import os
import json
import random
import logging
from typing import Optional
from weather import OpenMeteoClient, WeatherResult
import feedparser
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from babel.dates import format_date

now = datetime.now()
date = format_date(now, format='EEEE d MMMM yyyy', locale='fr_FR').capitalize()
if now.minute >= 20:
    now = now + timedelta(hours=1)
hour = now.strftime("%Hh")

# ---------------------------
# 🪵 Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ---------------------------
# ⚙️ Config
# ---------------------------
MODEL = "gemma3n"
MAX_RETRIES = 3
VOICE = "fr-FR-HenriNeural"

# Seuil en km/h à partir duquel le vent est mentionné
WIND_THRESHOLD_KMH = 50
# Garder les 30 derniers articles pour couvrir environ 3 jours de news
HISTORY_SIZE = 30 
SIMILARITY_THRESHOLD = 0.6
HISTORY_FILE = os.path.join(os.path.dirname(__file__), ".news_history.json")

# ---------------------------
# 📋 History persistence
# ---------------------------
def load_history() -> tuple[list[str], list[str]]:
    """
    Retourne (titres_anciens, titres_du_jour).
    titres_anciens : hier ou avant → utilisés pour filtre de similarité
    titres_du_jour : aujourd'hui → utilisés uniquement pour filtre de doublon exact
    """
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return [], []
        old, today_titles = [], []
        for entry in data:
            if isinstance(entry, dict):
                if entry.get("date", "1970-01-01") < today:
                    old.append(entry["title"])
                else:
                    today_titles.append(entry["title"])
            else:
                old.append(entry)  # anciens formats → traités comme anciens
        return old, today_titles
    except (FileNotFoundError, json.JSONDecodeError):
        return [], []

def save_history(history: list[str]) -> None:
    """Persist the last HISTORY_SIZE topics to disk, avoiding duplicates."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Charger l'existant
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    # 2. Normaliser l'existant (gestion des anciens formats str -> dict)
    # Et extraire un set des titres déjà présents pour une recherche rapide
    normalized = []
    existing_titles_set = set()
    
    for entry in existing:
        if isinstance(entry, str):
            title = entry
            entry_dict = {"title": title, "date": "1970-01-01"}
        else:
            title = entry.get("title", "")
            entry_dict = entry
            
        normalized.append(entry_dict)
        existing_titles_set.add(title)

    # 3. Ajouter UNIQUEMENT les nouveaux titres qui ne sont pas dans le set
    added_count = 0
    for title in history:
        if title not in existing_titles_set:
            normalized.append({"title": title, "date": today})
            existing_titles_set.add(title) # Évite les doublons au sein du même batch
            added_count += 1

    if added_count > 0:
        log.info(f"Historique : {added_count} nouveaux titres ajoutés.")
        # 4. Sauvegarder les derniers HISTORY_SIZE
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(normalized[-HISTORY_SIZE:], f, ensure_ascii=False, indent=2)
    else:
        log.info("Historique : aucun nouveau titre à sauvegarder.")

# ---------------------------
# FETCH RSS
# ---------------------------
def clean_html(text):
    """Supprime les balises HTML pour un texte propre"""
    return re.sub('<.*?>', '', text)

import json

def is_similar(a: str, b: str) -> bool:
    """Calcule le ratio de similarité entre deux chaînes."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > SIMILARITY_THRESHOLD

def select_top_news(articles: list[dict], top_k: int = 6, history: Optional[list[str]] = None):
    history = history or []

    # 1. Filtrage similarité contre l'historique
    filtered_articles = []
    for art in articles:
        is_duplicate = False
        title_to_check = art['title']

        for past_title in history:
            if is_similar(title_to_check, past_title):
                is_duplicate = True
                log.info(f"Filtre historique : {title_to_check} -> {past_title}")
                break

        if not is_duplicate:
            for accepted in filtered_articles:
                if is_similar(title_to_check, accepted['title']):
                    is_duplicate = True
                    log.info(f"Filtre doublon batch : {title_to_check} -> {accepted['title']}")
                    break

        if not is_duplicate:
            filtered_articles.append(art)

    if not filtered_articles:
        log.warning("Tous les articles filtrés, on repart des originaux.")
        filtered_articles = articles[:]

    # 2. Sélection équilibrée par source
    sources = list({a["source_name"] for a in filtered_articles})
    per_source: dict[str, list[dict]] = {s: [] for s in sources}
    for art in filtered_articles:
        per_source[art["source_name"]].append(art)

    selected = []
    # Round-robin entre les sources pour garantir la diversité
    min_per_source = max(1, top_k // len(sources))
    for s in sources:
        selected.extend(per_source[s][:min_per_source])

    # Compléter jusqu'à top_k avec les articles restants
    selected_ids = {id(a) for a in selected}
    for art in filtered_articles:
        if len(selected) >= top_k:
            break
        if id(art) not in selected_ids:
            selected.append(art)
            selected_ids.add(id(art))

    log.info(f"Articles sélectionnés : {len(selected)}")
    return selected[:top_k]

def fetch_and_rank_news(rss_urls: list[str], target_news_number: int, max_articles: int = 20):
    all_articles = []

    for url in rss_urls:
        feed = feedparser.parse(url)

        if not feed.entries:
            log.warning(f"Flux vide ou invalide : {url}")
            continue
        
        feed_title = extract_source_name(url)

        for entry in feed.entries[:max_articles]:
            title = entry.get("title", "")

            if "content" in entry:
                summary = entry.content[0].value
            else:
                summary = entry.get("summary", "") or entry.get("description", "")

            summary = clean_html(summary)

            # Récupérer la date de publication
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                pub_dt = datetime(*published[:6])
            else:
                pub_dt = datetime.min  # articles sans date mis en dernier

            all_articles.append({
                "title": title,
                "summary": summary,
                "source": url,
                "source_name": feed_title,
                "published": pub_dt,
            })
    
    if not all_articles:
        raise ValueError("Tous les flux RSS sont vides ou invalides")

    # Dédoublonnage
    seen_titles = set()
    unique_articles = []
    for art in all_articles:
        title_key = re.sub(r'\s+', ' ', art['title'].strip().lower())
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(art)
    all_articles = unique_articles

    # Tri du plus récent au plus ancien (plus de shuffle)
    all_articles.sort(key=lambda a: a["published"], reverse=True)
    log.info(f"Article le plus récent : {all_articles[0]['published']} — {all_articles[0]['title']}")

    history, _ = load_history()
    top_articles = select_top_news(all_articles, top_k=target_news_number, history=history)

    return top_articles

# ---------------------------
# 🔧 LLM utilities
# ---------------------------
def call_llm(prompt: str, system_prompt: str, temperature: float = 1.0, max_tokens: int = 1024) -> Optional[str]:
    """Call Ollama with retry logic and truncation detection."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "repeat_penalty": 1.2,
                    "num_predict": max_tokens,
                }
            )
            content = response["message"]["content"].strip()
            if not content:
                raise ValueError("Empty response from model")
            return content
        except Exception as e:
            log.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")

    log.error("All attempts failed.")
    return None

def clean_text(text: str) -> str:
    # Supprimer les méta-introductions du LLM
    meta_patterns = [
        r"^Voici un flash info.*?(?=\n\n|\w{10})",
        r"^Voici les informations.*?(?=\n\n)",
        r"^Flash Info\s*[-–]\s*",
    ]
    for pattern in meta_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r'#+\s*', '', text)                              # ## titres
    text = re.sub(r'(?m)^\s*\d+[\).\s]+', '', text)               # 1. 2. 3.
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)            # **gras**
    text = re.sub(r'\[.*?\]', '', text)                            # [crochets]
    text = re.sub(r'\(.*?\)', '', text)                            # (parenthèses)
    text = re.sub(r'\xa0', ' ', text)                              # espaces insécables
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------------------
# 🧠 Main pipeline
# ---------------------------

def anounce_podcast(topic):
    system_prompt = "Tu es un animateur radio dynamique. \n Tu DOIS répondre aux prompts en une seule phrase simple et assez courte. \nFais comme si tu étais en direct à la radio. \n Pas d'emoji dans ta réponse ; tout doit être lisible par synthèse vocale."
    prompt = f"Tu dois annoncer le prochain podcast dont le sujet est : \"{topic}\""
    content = call_llm(prompt, system_prompt)
    if not content:
        log.warning("LLM returned nothing...")
        return 
    return content

def weather_code_to_text(code):
    mapping = {
        0: "ciel dégagé",
        1: "quasi dégagé",
        2: "partiellement nuageux",
        3: "couvert",
        45: "brouillard",
        48: "brouillard givrant",
        51: "bruine légère",
        53: "bruine",
        61: "pluie légère",
        63: "pluie modérée",
        65: "pluie forte",
        71: "neige légère",
        80: "averses",
        95: "orage"
    }
    return mapping.get(code, "conditions variables")

def anounce_weather(weather: WeatherResult):
    system_prompt = (
        "Tu es un présentateur météo radio.\n"
        "Tu fais un bulletin naturel, fluide et précis.\n"
        "N'INVENTE RIEN.\n"
        "Maximum 3 phrases.\n"
        "Pas d'emoji."
    )

    current = weather.current
    today = weather.forecast[0]

    conditions = weather_code_to_text(today.weathercode)

    wind_line = ""
    if current.windspeed >= WIND_THRESHOLD_KMH:
        wind_line = f"Vent fort : {round(current.windspeed)} km/h.\n"

    prompt = (
        f"Nous sommes le {date}. Il est {hour}\n"
        f"Météo à {weather.city} aujourd'hui.\n"
        f"Actuellement : {round(current.temperature)}°C.\n"
        f"{wind_line}"
        f"Prévisions : min {round(today.temp_min)}°C, max {round(today.temp_max)}°C.\n"
        f"Précipitations prévues : {today.precipitation_sum} mm.\n"
        f"Conditions : {conditions}."
    )

    return call_llm(prompt, system_prompt, temperature=0.2)

def anounce_weather_tomorrow(weather: WeatherResult):
    system_prompt = (
        "Tu es un présentateur météo radio.\n"
        "Tu fais un bulletin fluide et naturel.\n"
        "N'INVENTE RIEN.\n"
        "Maximum 3 phrases.\n"
        "Pas d'emoji."
    )

    tomorrow = weather.forecast[1]

    conditions = weather_code_to_text(tomorrow.weathercode)

    wind_line = ""
    if tomorrow.wind_speed_max >= WIND_THRESHOLD_KMH:
        wind_line = f"Vent fort attendu : jusqu'à {round(tomorrow.wind_speed_max)} km/h.\n"

    prompt = (
        f"Nous sommes le {date}. Il est {hour}\n"
        f"Météo pour demain à {weather.city}.\n"
        f"Températures : min {round(tomorrow.temp_min)}°C, max {round(tomorrow.temp_max)}°C.\n"
        f"Pluie prévue : {tomorrow.precipitation_sum} mm.\n"
        f"{wind_line}"
        f"Conditions : {conditions}."
    )

    return call_llm(prompt, system_prompt, temperature=0.2)


def announce_weather_national(data):
    system_prompt = (
        f"Tu es un présentateur météo radio. Nous sommes le {date}, il est {hour}.\n"
        "N'invente rien et base toi uniquement sur ce qui t'es donné en prompt.\n"
        "Synthétise les données pour ne pas trop te répéter."
    )
    
    context = ""
    for i, d in enumerate(data):
        dt = datetime.strptime(d['date'], "%Y-%m-%d")
        # Format : "Vendredi 24 avril"
        jour_nom = format_date(dt, format='EEEE d MMMM', locale='fr_FR').capitalize()
        reg_details = ", ".join([f"{r}: {v['t_max']}°C ({weather_code_to_text(v['weathercode'])}, {v['pluie']}mm de pluie{f", vent fort {v['vent']} km/h" if v["vent"] >= WIND_THRESHOLD_KMH else ""})" for r, v in d['regions'].items()])
        context += f"{jour_nom} : {reg_details}\n\n"
    
    prompt = f"DONNÉES MÉTÉO :\n{context}\n\nRédige le bulletin radio :"
    return call_llm(prompt, system_prompt, temperature=0.3)

def extract_source_name(url: str) -> str:
    """Extrait un nom de média lisible depuis une URL de flux RSS."""
    known_sources = {
        "lemonde.fr": "Le Monde",
        "lefigaro.fr": "Le Figaro",
        "liberation.fr": "Libération",
        "rfi.fr": "RFI",
        "franceinfo.fr": "France Info",
        "france24.com": "France 24",
        "bfmtv.com": "BFM TV",
        "nouvelobs.com": "Le Nouvel Obs",
        "mediapart.fr": "Mediapart",
        "leparisien.fr": "Le Parisien",
        "20minutes.fr": "20 Minutes",
        "rtl.fr": "RTL",
        "europe1.fr": "Europe 1",
        "lexpress.fr": "L'Express",
        "lepoint.fr": "Le Point",
    }

    for domain, name in known_sources.items():
        if domain in url:
            return name

    # Fallback : extraire le domaine brut
    match = re.search(r'https?://(?:www\.|feeds\.|rss\.)?([^/]+)', url)
    if match:
        return match.group(1)

    return "source inconnue"

def anounce_news(rss_url: str, target_news_number: int):
    system_prompt = (
        "Tu es un présentateur de journal radio expérimenté sur une grande antenne nationale.\n"
        "Ton objectif est de transformer des dépêches brutes en un flash info fluide, vivant et structuré.\n"
        "\n"
        "STRUCTURE OBLIGATOIRE :\n"
        "- Regroupe les articles par grandes thématiques : International, France, Justice, Culture, Économie, etc.\n"
        "- Commence chaque thématique par une phrase de transition naturelle (ex: 'Sur le plan international...', 'En France maintenant...', 'Du côté de la justice...', 'Dans l'actualité culturelle...', 'Côté économie...').\n"
        "- Entre chaque article d'une même thématique, utilise des connecteurs logiques (ex: 'Toujours à l'international...', 'Dans un autre registre...', 'Par ailleurs...', 'On enchaîne avec...').\n"
        "\n"
        "CITATIONS DE SOURCES :\n"
        "- Cite toutes les sources journalistiques à la fin.\n"
        "- Ne répète jamais la même source deux fois de suite.\n"
        "\n"
        "RÈGLES ABSOLUES :\n"
        "- N'INVENTE RIEN. Chaque fait, chiffre, nom propre doit provenir EXACTEMENT du texte fourni.\n"
        "- COPIE les noms propres tels quels depuis les dépêches. N'en déduis ou n'en corriges AUCUN.\n"
        "- INTERDIT : compléter ou corriger un nom propre depuis ta mémoire. Si le texte dit 'pape Léon XIV', tu écris 'pape Léon XIV', jamais 'pape François'.\n"
        "- INTERDIT : Ne mentionne jamais deux fois la même information, même si elle apparaît dans deux dépêches différentes.\n"
        "- INTERDIT : les formulations 'retour sur', 'rappel de', 'récapitulatif', 'bilan de la semaine'. Le flash info est toujours présenté comme actuel et en direct.\n"
        "- Chaque changement de zone géographique ou de thématique doit être explicitement annoncé par une phrase de transition.\n"
        "- Si tu n'es pas certain d'un nom propre, utilise une formulation générique ('le pape', 'le président', 'le ministre') plutôt que d'inventer.\n"
        "- Développe chaque article avec les détails du résumé : chiffres, lieux, noms, causes, conséquences.\n"
        "- Commence DIRECTEMENT par la phrase d'accueil radio, sans aucune introduction sur ce que tu vas faire.\n"
        "- Termine par une courte conclusion de clôture.\n"
        "\n"
        "CONTRAINTES TECHNIQUES :\n"
        "- Aucune indication de mise en scène entre crochets ou parenthèses.\n"
        "- Aucun markdown (pas de **, ##, tirets de liste, numérotation).\n"
        "- Tu n'as pas de correspondant ou d'intervenant, ne pas en parler.\n"
        "- Réponds uniquement avec le texte qui sera lu par la synthèse vocale.\n"
    )
    all_articles = []
    urls = rss_url.split()
    selected_articles = fetch_and_rank_news(urls, int(target_news_number))

    if not selected_articles:
        log.warning("Aucune nouvelle information après filtrage.")
        return None

    # Mise à jour de l'historique avec les titres sélectionnés
    new_titles = [a['title'] for a in selected_articles]
    save_history(new_titles)

    # Construire le contexte avec la source pour chaque article
    sources_uniques = sorted(list({a.get("source_name") for a in selected_articles}))
    sources_str = ", ".join(sources_uniques)

    # 2. Construire le contexte pour le LLM
    formatted_news = ""
    for art in selected_articles:
        formatted_news += (
            f"SOURCE DE CETTE DÉPÊCHE : {art['source_name']}\n" # Source spécifique à l'article
            f"Titre : {art['title']}\n"
            f"Détails : {art['summary']}\n\n"
        )

    prompt = (
        f"Nous sommes le {date}. Il est {hour}\n"
        f"Voici les dépêches à traiter :\n\n{formatted_news}\n"
        "CONSIGNES DE RÉDACTION :\n"
        "- Analyse bien le lieu de l'action : si un article parle de la Cour Suprême américaine, ne le place PAS en rubrique 'France'.\n"
        "- INTERDICTION ABSOLUE de répéter deux fois le même sujet (ex: si deux dépêches parlent des mêmes artistes, fusionne-les ou choisis la meilleure).\n"
        f"- Termine obligatoirement par cette phrase exacte : 'Ces informations nous ont été présentées par {sources_str}.'"
    )

    content = call_llm(prompt, system_prompt, temperature=0.0, max_tokens=4096)
    
    if not content:
        log.warning("LLM returned nothing...")
        return None

    return content
# ---------------------------
# 🔊 Generate TTS
# ---------------------------
async def generate_audio_and_subs(text, voice, audio_path, srt_path):
    communicate = edge_tts.Communicate(text, voice)
    submaker = edge_tts.SubMaker()

    with open(audio_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            # MODIFICATION ICI : On accepte les phrases si les mots sont absents
            elif chunk["type"] in ["WordBoundary", "SentenceBoundary"]:
                submaker.feed(chunk)

    # On vérifie si on a récupéré quelque chose
    subtitles = submaker.get_srt()

    with open(srt_path, "w", encoding="utf-8") as f:
        if srt_path.endswith(".vtt"):
            f.write("WEBVTT\n\n")
            vtt_content = re.sub(r'(\d),(\d)', r'\1.\2', subtitles)
            f.write(vtt_content)
        else:
            f.write(subtitles)

# ---------------------------
# 🚀 Entry point
# ---------------------------
if __name__ == "__main__":
    arg_1 = sys.argv[1]

    match arg_1:
        case "podcast":
            arg_2 = sys.argv[2]
            content = clean_text(anounce_podcast(arg_2))
            audio_path = "./announce.wav"
            srt_path = "./announce.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("Annonce de podcast générée avec succès.")

        case "news":
            rss_url = sys.argv[2]
            target_news_number = sys.argv[3]
            
            content = clean_text(anounce_news(rss_url, target_news_number))
            audio_path = "./news.wav"
            srt_path = "./news.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("Flash Info généré avec succès.")

        case "meteo":
            client = OpenMeteoClient()
            log.info("Analyse de la tendance d'aujourd'hui sur toute la France...")
            data = client.get_national_today_forecast()
            
            content = clean_text(announce_weather_national([data]))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("Bulletin national généré avec succès.")

        case "meteo_demain":
            client = OpenMeteoClient()
            log.info("Analyse de la tendance de demain sur toute la France...")
            data = client.get_national_tomorrow_forecast()
            
            content = clean_text(announce_weather_national([data]))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("Bulletin national généré avec succès.")

        case "meteo_semaine":
            client = OpenMeteoClient()
            log.info("Analyse de la tendance de la semaine sur toute la France...")
            data = client.get_national_weekly_forecast()
            
            content = clean_text(announce_weather_national(data))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("Bulletin national généré avec succès.")

        case "meteo_ville":
            arg_2 = sys.argv[2]
            client = OpenMeteoClient()
            weathers = client.get_weather_by_city(arg_2)

            content = clean_text(anounce_weather(weathers))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")
        
        case "meteo_ville_demain":
            arg_2 = sys.argv[2]
            client = OpenMeteoClient()
            weathers = client.get_weather_by_city(arg_2)

            content = clean_text(anounce_weather_tomorrow(weathers))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")

        case "meteo_ville_semaine":
            arg_2 = sys.argv[2]
            client = OpenMeteoClient()
            weathers = client.get_weather_by_city(arg_2)

            content = clean_text(announce_weather_week(weathers))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info(f"DONE :)")  
        case _:
            pass

