# GridPilot AI

## Date meteo și configurația liniei

Rulați din directorul aplicației `gridpilot-ai`:

```bash
python -m pip install -r requirements.txt
python -m src.data_pipeline
python -m src.data_pipeline --forecast-hours 24
```

- `config/line_segments.json` este configurația comună pentru colectarea meteo și calculul ampacității. Cele trei puncte și azimuturile lor sunt ilustrative și nu reprezintă active PPC.
- `data/processed/weather.csv` conține date orare istorice pentru 2022–2025, din [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), model ERA5 (date distribuite sub CC BY 4.0). Fișierul poate fi folosit și offline. `--start` și `--end` schimbă intervalul.
- `data/processed/weather_forecast.csv` conține prognoza pentru demo, din [Open-Meteo Forecast API](https://open-meteo.com/en/docs), și se regenerează când este necesară. Pentru un demo offline, folosiți `weather.csv` ca reluare istorică.
- `data/processed/dataset.csv` este setul anterior, cu laguri și ținte meteo la +1 h. Fluxul de ampacitate citește `weather.csv`.

Contractul meteo: `timestamp_utc` (UTC), `segment_id`, `air_temp_c` (°C la 2 m), `wind_speed_ms` (m/s la 10 m), `wind_direction_deg` (direcția din care bate vântul, în grade de la nord, la 10 m), `solar_radiation` (GHI, W/m²), `source`, `data_quality`. Orele cu măsurători indisponibile sau valori invalide sunt păstrate cu celulele respective goale și marcajul `missing`. ERA5 este o reanaliză pe grilă, fără senzori pe conductoare.

După integrarea modulului ML, din același director:

```bash
python -m src.ml_model train --weather data/processed/weather.csv --segments config/line_segments.json
python -m src.ml_model predict --weather data/processed/weather_forecast.csv --out data/processed/predictions.csv
```
