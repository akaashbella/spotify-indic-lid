# Spotify South Asian (Hindi/Tamil) Playlist Pipeline

Filters your Spotify **Liked Songs** into **per-language playlists** using language detection (AI4Bharat IndicLID). Supports Hindi and Tamil in both native and Romanized script (Hinglish/Tanglish). Songs with multiple Indian languages appear in each corresponding playlist.

## Quick start

1. **Environment**
   - Copy `.env.example` to `.env` and set:
     - `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` (from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard))
     - `GENIUS_ACCESS_TOKEN` (from [Genius API](https://genius.com/api-clients))
   - Optional: `SPOTIFY_REDIRECT_URI` (default `http://127.0.0.1:9090`), `SPOTIFY_PLAYLIST_NAME` (default `Indian Collection`).

2. **IndicLID**
   - Clone the repo and download models (see **IndicLID setup** in `requirements.txt`).
   - Or set `INDICLID_MODEL_DIR` to a directory containing `indiclid-ftn/`, `indiclid-ftr/`, `indiclid-bert/`.

3. **Run**
   - Local: `pip install -r requirements.txt` then `python main.py`.

## Behavior

- LID runs **per line**; each track gets a confidence per language (e.g. Hindi and Tamil). A song can be assigned to **multiple** languages.
- **Confidence ≥ 0.8** for a language: track is added to that language’s playlist (e.g. **Indian Collection - Hindi**, **Indian Collection - Tamil**). If a track has both Hindi and Tamil above threshold, it is added to both playlists.
- **Confidence 0.4–0.7**: track is written to `needs_review.csv` for manual check.
- Progress is stored in a SQLite DB (`SPOTIFY_LID_DB`); re-runs resume from the last state (e.g. after an HPC time limit).

## Outputs

- **Spotify playlists**: `Indian Collection - Hindi`, `Indian Collection - Tamil` (created or updated; tracks can appear in both).
- **`indian_songs.csv`**: all tracks with at least one Indian language detected. Columns: `track_id`, `name`, `artists`, `added_at`, `languages`, `hindi_confidence`, `tamil_confidence`, `in_hindi_playlist`, `in_tamil_playlist`.
- **`needs_review.csv`**: tracks in the 0.4–0.7 confidence band.
