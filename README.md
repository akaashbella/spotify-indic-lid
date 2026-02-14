# Spotify South Asian (Hindi/Tamil) Playlist Pipeline

Filters your Spotify **Liked Songs** into a playlist (**Indian Collection**) using language detection (AI4Bharat IndicLID). Supports Hindi and Tamil in both native and Romanized script (Hinglish/Tanglish).

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
   - HPC: configure `submit_job.sh` (set `#SBATCH --account=...`) and `sbatch submit_job.sh`.

## Behavior

- **Confidence ≥ 0.8** (Hindi/Tamil): track is added to the playlist.
- **Confidence 0.4–0.7**: track is written to `needs_review.csv` for manual check.
- Progress is stored in a SQLite DB (`SPOTIFY_LID_DB`); re-runs resume from the last state (e.g. after an HPC time limit).

## Outputs

- Spotify playlist: **Indian Collection** (created or updated).
- `needs_review.csv`: tracks in the 0.4–0.7 confidence band.
