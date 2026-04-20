# Radio Genetrator

Generate radio audios (with subtitles) in French.

## Usage

```bash
./run.sh "news" "rss feed url" 5        # generate news (5 articles selected)
./run.sh "meteo_ville" "city"           # generate forecast for a city
./run.sh "meteo_ville_demain" "city"    # generate tomorrow's forecast for a city
./run.sh "meteo_ville_semaine" "city"   # generate next 5 days' forecast for a city
./run.sh "meteo"                        # generate forecast for France
./run.sh "meteo_demain"                 # generate tomorrow's forecast for France
./run.sh "meteo_semaine"                # generate next 5 days' forecast for France
./run.sh "podcast" "theme"              # generate announce for podcast
```
