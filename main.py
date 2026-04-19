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


# ---------------------------
# FETCH RSS
# ---------------------------
def clean_html(text):
    """Supprime les balises HTML pour un texte propre"""
    return re.sub('<.*?>', '', text)

import json

def select_top_news_llm(articles: list[dict], top_k: int = 10) -> list[dict]:
    system_prompt = (
        "Tu es un système de sélection d'articles.\n"
        "Tu réponds UNIQUEMENT avec un tableau JSON brut, sans aucun texte avant ou après.\n"
        "AUCUN commentaire, AUCUN markdown, AUCUNE explication.\n"
        "Commence ta réponse directement par [ et termine par ].\n\n"
        "Format STRICT :\n"
        "[{\"index\": 0}, {\"index\": 3}, {\"index\": 5}]"
    )

    formatted = ""
    for i, art in enumerate(articles):
        formatted += (
            f"Article {i}: {art['title']}\n"
            f"Résumé: {art['summary'][:300]}\n\n"
        )

    prompt = (
        f"Sélectionne les {top_k} articles les plus importants parmi cette liste.\n"
        f"Critères : actualité nationale, internationale, politique, crise, économie.\n\n"
        f"{formatted}\n"
        f"Réponds UNIQUEMENT avec le JSON. Exemple : [{{\"index\": 0}}, {{\"index\": 2}}]"
    )

    response = call_llm_json(prompt, system_prompt, temperature=0.1, max_tokens=800)

    if not response:
        log.warning("LLM vide, fallback")
        return articles[:top_k]

    indexes = parse_indexes(response, len(articles))
    
    if not indexes:
        log.warning(f"Aucun index extrait | Réponse brute : {response[:300]}")
        return articles[:top_k]

    selected = [articles[i] for i in indexes if 0 <= i < len(articles)]

    if not selected:
        log.warning("Indexes hors limites, fallback")
        return articles[:top_k]

    # log.info(f"Sélection OK ({len(selected[:top_k])}) : {[a['title'] for a in selected[:top_k]]}")
    return selected[:top_k]

def fetch_and_rank_news(rss_urls: list[str], max_articles: int = 20):
    all_articles = []

    for url in rss_urls:
        feed = feedparser.parse(url)

        if not feed.entries:
            continue  # skip les flux vides plutôt que crash
        
        feed_title = feed.feed.get("title", "") or extract_source_name(url)

        for entry in feed.entries[:max_articles]:
            title = entry.get("title", "")

            if "content" in entry:
                summary = entry.content[0].value
            else:
                summary = entry.get("summary", "") or entry.get("description", "")

            summary = clean_html(summary)

            all_articles.append({
                "title": title,
                "summary": summary,
                "source": url,
                "source_name": feed_title   # <-- ajout
            })
    
    if not all_articles:
        raise ValueError("Tous les flux RSS sont vides ou invalides")

    seen_titles = set()
    unique_articles = []
    for art in all_articles:
        title_key = re.sub(r'\s+', ' ', art['title'].strip().lower())
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(art)
    all_articles = unique_articles

    # 🧠 sélection intelligente TOP 10 par LLM
    top_articles = select_top_news_llm(all_articles, top_k=10)

    return top_articles

def parse_indexes(response: str, nb_articles: int) -> list[int]:
    """Extrait des indexes valides depuis n'importe quelle structure JSON ou texte."""
    
    # Tentative 1 : JSON direct
    try:
        data = json.loads(response)
        
        # Format : [{"index": 0}, {"index": 1}]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return [item["index"] for item in data if "index" in item]
        
        # Format : [0, 1, 2]
        if isinstance(data, list) and data and isinstance(data[0], int):
            return data
        
        # Format : {"indexes": [0, 1, 2]} ou {"selected": [0, 1]}
        if isinstance(data, dict):
            for key in ("indexes", "selected", "indices", "articles"):
                if key in data and isinstance(data[key], list):
                    val = data[key]
                    if val and isinstance(val[0], int):
                        return val
                    if val and isinstance(val[0], dict):
                        return [item.get("index", item.get("id")) for item in val]
    
    except json.JSONDecodeError:
        pass

    # Tentative 2 : extraire un tableau JSON embarqué dans du texte
    match = re.search(r'\[[\s\S]*?\]', response)
    if match:
        try:
            data = json.loads(match.group())
            if data and isinstance(data[0], int):
                return data
            if data and isinstance(data[0], dict):
                return [item["index"] for item in data if "index" in item]
        except Exception:
            pass

    # Tentative 3 : extraire les numéros de liste ("1. Titre", "2. Titre")
    nums = re.findall(r'(?:^|\n)\s*(\d+)[\.\)]\s', response)
    if nums:
        return [int(n) - 1 for n in nums]  # "1." → index 0

    return []

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

def call_llm_json(prompt: str, system_prompt: str, temperature: float = 0.1, max_tokens: int = 800) -> Optional[str]:
    """Variante de call_llm qui force la sortie JSON via Ollama."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                format="json",   # <-- contraint le sampler
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            content = response["message"]["content"].strip()
            if not content:
                raise ValueError("Empty response")
            return content
        except Exception as e:
            log.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
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

    prompt = (
        f"Météo à {weather.city} aujourd'hui.\n"
        f"Actuellement : {current.temperature}°C, vent {current.windspeed} km/h.\n"
        f"Prévisions : min {today.temp_min}°C, max {today.temp_max}°C.\n"
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

    prompt = (
        f"Météo pour demain à {weather.city}.\n"
        f"Températures : min {tomorrow.temp_min}°C, max {tomorrow.temp_max}°C.\n"
        f"Pluie prévue : {tomorrow.precipitation_sum} mm.\n"
        f"Vent maximum : {tomorrow.wind_speed_max} km/h.\n"
        f"Conditions : {conditions}."
    )

    return call_llm(prompt, system_prompt, temperature=0.2)

def get_france_weather(client):
    cities = [
        "Paris", "Lille", "Strasbourg", "Brest",
        "Lyon", "Clermont-Ferrand",
        "Toulouse", "Bordeaux",
        "Marseille", "Nice", "Montpellier"
    ]

    results = []

    for city in cities:
        try:
            weather = client.get_weather_by_city(city, country="France")
            results.append(weather)
        except Exception as e:
            log.warning(f"Erreur météo pour {city}: {e}")

    return results

def anounce_weather_france(weathers: list[WeatherResult]):
    system_prompt = (
        "Tu es un présentateur météo radio national.\n"
        "Tu fais un bulletin météo pour toute la France.\n"
        "Style fluide, naturel et dynamique.\n"
        "Pas d'emoji ni de markdown.\n"
        "Précise la température en °C et les précipitations en ml."
    )

    formatted = ""
    for w in weathers:
        today = w.forecast[0]
        conditions = weather_code_to_text(today.weathercode)
        formatted += (
            f"{w.city} : min {today.temp_min}°C, max {today.temp_max}°C, "
            f"pluie {today.precipitation_sum} mm, conditions : {conditions}.\n"
        )

    prompt = (
        "Voici la météo du jour pour chaque ville :\n\n"
        f"{formatted}\n"
        "Fais un bulletin météo radio, naturel et fluide."
    )

    return call_llm(prompt, system_prompt, temperature=0.2)


def anounce_weather_france_tomorrow(weathers: list[WeatherResult]):
    system_prompt = (
        "Tu es un présentateur météo radio national.\n"
        "Tu fais les prévisions météo pour demain sur toute la France.\n"
        "Style fluide, naturel et dynamique.\n"
        "Pas d'emoji ni de markdown.\n"
        "Précise la température en °C et les précipitations en ml."
    )

    formatted = ""
    for w in weathers:
        tomorrow = w.forecast[1]
        conditions = weather_code_to_text(tomorrow.weathercode)
        formatted += (
            f"{w.city} : min {tomorrow.temp_min}°C, max {tomorrow.temp_max}°C, "
            f"pluie {tomorrow.precipitation_sum} mm, "
            f"vent max {tomorrow.wind_speed_max} km/h, conditions : {conditions}.\n"
        )

    prompt = (
        "Voici les prévisions météo pour demain, ville par ville :\n\n"
        f"{formatted}\n"
        "Fais un bulletin météo radio, naturel et fluide."
    )

    return call_llm(prompt, system_prompt, temperature=0.2)

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

def anounce_news(rss_url: str):
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
        "- Cite les sources journalistiques UNE SEULE FOIS, à la toute fin.\n"
        "- Ne répète jamais la même source deux fois de suite.\n"
        "\n"
        "RÈGLES ABSOLUES :\n"
        "- N'INVENTE RIEN. Chaque fait, chiffre, nom propre doit provenir EXACTEMENT du texte fourni.\n"
        "- COPIE les noms propres tels quels depuis les dépêches. N'en déduis ou n'en corriges AUCUN.\n"
        "- INTERDIT : compléter ou corriger un nom propre depuis ta mémoire. Si le texte dit 'pape Léon XIV', tu écris 'pape Léon XIV', jamais 'pape François'.\n"
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

    urls = rss_url.split()
    try:
        articles = fetch_and_rank_news(urls)
    except Exception as e:
        log.error(f"Erreur RSS : {e}")
        return None

    # Construire le contexte avec la source pour chaque article
    formatted_news = ""
    for art in articles:
        source_name = art.get("source_name") or extract_source_name(art['source'])
        formatted_news += (
            f"Source : {source_name}\n"
            f"Titre : {art['title']}\n"
            f"Détails : {art['summary']}\n\n"
        )

    prompt = (
        "Voici les informations phares du moment :\n\n"
        f"{formatted_news}"
        "\n"
        "RAPPELS STRICTS avant de rédiger :\n"
        "- Recopie les noms propres EXACTEMENT comme ils apparaissent dans les dépêches ci-dessus.\n"
        "- N'utilise JAMAIS ta mémoire pour compléter un nom. Ce qui n'est pas dans le texte n'existe pas.\n"
        "- Si le texte mentionne 'pape Léon XIV', écris 'pape Léon XIV'. Jamais autre chose.\n"
        "\n"
        "COMMENCE DIRECTEMENT par la phrase d'accueil radio.\n"
        "Regroupe les articles par thématiques avec des transitions naturelles.\n"
        "Pour chaque article : développe avec TOUS les détails présents dans le résumé.\n"
        "Prévoie environ 30 secondes de temps de parole par information.\n"
        "Cite les sources journalistiques à la fin.\n"
        "PAS de numérotation, PAS de titres, PAS de markdown.\n"
    )

    content = call_llm(prompt, system_prompt, temperature=0.2, max_tokens=4096)

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
    arg_2 = sys.argv[2]

    match arg_1:
        case "podcast":
            content = clean_text(anounce_podcast(arg_2))
            audio_path = "./announce.wav"
            srt_path = "./announce.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")
        
        case "meteo":
            client = OpenMeteoClient()
            weathers = get_france_weather(client)

            content = clean_text(anounce_weather_france(weathers))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")

        case "meteo_demain":
            client = OpenMeteoClient()
            weathers = get_france_weather(client)

            content = clean_text(anounce_weather_france_tomorrow(weathers))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")

        case "meteo_ville":
            client = OpenMeteoClient()
            weathers = client.get_weather_by_city(arg_2)

            content = clean_text(anounce_weather(weathers))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")
        
        case "meteo_ville_demain":
            client = OpenMeteoClient()
            weathers = client.get_weather_by_city(arg_2)

            content = clean_text(anounce_weather_tomorrow(weathers))
            audio_path = "./weather.wav"
            srt_path = "./weather.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")

        case "news":
            rss_url = arg_2
            content = clean_text(anounce_news(rss_url))
            audio_path = "./news.wav"
            srt_path = "./news.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")
        
        case _:
            pass
