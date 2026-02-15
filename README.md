# Spotify Indian Languages Playlist Pipeline

Filters your Spotify **Liked Songs** into **per-language playlists** using language detection (AI4Bharat IndicLID). Supports **Hindi, Tamil, Telugu, Malayalam, and Kannada** in both native and Romanized script (e.g. Hinglish, Tanglish). Songs with multiple Indian languages appear in each corresponding playlist.

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
   - **Headless / high-performance clusters:** See **Before running on a cluster** below.

### Before running on a cluster

Cluster nodes typically have no browser, so do this once locally, then upload to the cluster:

1. **Spotify auth:** Run `python main.py` on your laptop until you see authentication succeed. This creates a **`.cache`** file in the project folder.
2. **Upload to cluster:** Copy the project to the cluster including the **`.cache`** file and your **`.env`**.
3. **In `.env` on the cluster**, set:
   - `SPOTIFY_CACHE_PATH=.cache` (so the script uses the token file you uploaded).
   - `INDICLID_MODEL_DIR=/absolute/path/to/models` (path where you extracted the IndicLID model files on the cluster).

See **Setup on a headless cluster** in `requirements.txt` for the exact IndicLID directory layout.

## Behavior

- LID runs **per line**; each track gets a confidence per language (e.g. Hindi and Tamil). A song can be assigned to **multiple** languages.
- **Confidence ≥ 0.8** for a language: track is added to that language’s playlist (e.g. **Indian Collection - Hindi**, **Indian Collection - Tamil**, **Indian Collection - Telugu**, etc.). A track can appear in multiple playlists if it has multiple languages above threshold.
- **Confidence 0.4–0.7**: track is written to `needs_review.csv` for manual check.
- Progress is stored in a SQLite DB (`SPOTIFY_LID_DB`); re-runs resume from the last state (e.g. after a cluster job time limit).

## Outputs

- **Spotify playlists**: one per language (Hindi, Tamil, Telugu, Malayalam, Kannada); tracks can appear in multiple playlists.
- **`indian_songs.csv`**: all tracks with at least one Indian language detected. Columns: `track_id`, `name`, `artists`, `added_at`, `languages`, plus per-language `{language}_confidence` and `in_{language}_playlist` for each of the five languages.
- **`needs_review.csv`**: tracks in the 0.4–0.7 confidence band.
