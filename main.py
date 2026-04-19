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

def select_top_news_llm(articles: list[dict], top_k: int = 5) -> list[dict]:
    system_prompt = (
        "Tu es un rédacteur en chef dans une grande agence de presse.\n"
        "Tu dois sélectionner les articles les plus importants.\n"
        "Retourne UNIQUEMENT un JSON valide.\n\n"
        "Format attendu :\n"
        "[\n"
        "  {\"index\": 0},\n"
        "  {\"index\": 3},\n"
        "  {\"index\": 5}\n"
        "]\n\n"
        "Choisis les articles les plus importants pour un flash info radio.\n"
        "Critères : impact international, politique, crise, économie.\n"
        "Ne donne aucun texte en dehors du JSON."
    )

    formatted = ""
    for i, art in enumerate(articles):
        formatted += (
            f"Article {i}:\n"
            f"Titre: {art['title']}\n"
            f"Résumé: {art['summary'][:500]}\n\n"
        )

    prompt = (
        f"Voici une liste d'articles d'actualité :\n\n"
        f"{formatted}\n"
        f"Sélectionne les {top_k} plus importants pour un flash info radio."
    )

    response = call_llm(prompt, system_prompt, temperature=0.2, max_tokens=800)

    if not response:
        return articles[:top_k]

    try:
        selection = json.loads(response)

        selected_indexes = [item["index"] for item in selection]

        # sécurisation + fallback
        selected = []
        for i in selected_indexes:
            if 0 <= i < len(articles):
                selected.append(articles[i])

        # si LLM bug → fallback
        if not selected:
            return articles[:top_k]

        return selected[:top_k]

    except Exception:
        return articles[:top_k]


def fetch_and_rank_news(rss_url: str, max_articles: int = 10):
    feed = feedparser.parse(rss_url)

    if not feed.entries:
        raise ValueError("Flux RSS vide ou invalide")

    articles = []

    for entry in feed.entries[:max_articles]:
        title = entry.get("title", "")

        if "content" in entry:
            summary = entry.content[0].value
        else:
            summary = entry.get("summary", "") or entry.get("description", "")

        summary = clean_html(summary)

        articles.append({
            "title": title,
            "summary": summary
        })

    # 🧠 sélection intelligente TOP 5 par LLM
    top_articles = select_top_news_llm(articles, top_k=5)
    
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
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'^\d+[\).\s]+', '', text)
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)
    text = re.sub(r'\(.*?\)', '', text)
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

def anounce_news(rss_url: str):
    system_prompt = (
        "Tu es un présentateur de journal radio expérimenté sur une grande antenne nationale.\n"
        "Ton objectif est de transformer des dépêches brutes en un flash info fluide, vivant et structuré.\n"
        "\n"
        "RÈGLES D'ÉCRITURE :\n"
        "- N'INVENTE RIEN. Si une information te semble éronnée, ne la corrige pas.\n"
        "- Développe les informations : ne te contente pas du titre, utilise les détails du résumé pour faire des phrases complètes.\n"
        "- Commence par une phrase d'accueil professionnelle (ex: 'L'essentiel de l'actualité de ce midi, c'est avec...') et termine par une conclusion brève.\n"
        "- Le ton doit être sérieux mais dynamique, avec un rythme soutenu.\n"
        "\n"
        "CONTRAINTES TECHNIQUES (STRICTES) :\n"
        "- N'ajoute AUCUNE indication de mise en scène entre crochets ou parenthèses (ex: [Musique], [Pause]).\n"
        "- Réponds uniquement avec le texte qui sera lu par la synthèse vocale.\n"
        "- Tu n'as pas de correspondant ou d'intervennant ; Tu ne dois pas en parler.\n" 
    )

    try:
        articles = fetch_and_rank_news(rss_url)
    except Exception as e:
        log.error(f"Erreur RSS : {e}")
        return None

    formatted_news = ""
    for art in articles:
        formatted_news += (
            f"Titre : {art['title']}\n"
            f"Détails : {art['summary']}\n\n"
        )
    
    prompt = (
        "Voici les informations phares du moment (pas forcément liées entre elles) :\n\n"
        f"{formatted_news}"
        "Rédige un flash info complet d'environ 2 minutes de temps de parole.\n"
        "Organise les infos de la plus importante à la plus légère.\n"
        "Fais des phrases travaillées pour une lecture fluide à la radio."
    )

    content = call_llm(prompt, system_prompt, temperature=0.2)

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

        case "news":
            rss_url = arg_2
            content = clean_text(anounce_news(rss_url))
            audio_path = "./news.wav"
            srt_path = "./news.vtt"
            asyncio.run(generate_audio_and_subs(content, VOICE, audio_path, srt_path))
            log.info("DONE :)")
        
        case _:
            pass
