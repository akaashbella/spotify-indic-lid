#!/usr/bin/env python3
"""
Spotify Liked Songs → South Asian (Hindi/Tamil) playlist pipeline.
Uses Spotify API, Genius lyrics, and AI4Bharat IndicLID. Resume-safe via SQLite.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config (env vars preferred for cluster runs)
# -----------------------------------------------------------------------------
CONFIG = {
    "db_path": os.environ.get("SPOTIFY_LID_DB", "spotify_lid_progress.db"),
    "playlist_name": os.environ.get("SPOTIFY_PLAYLIST_NAME", "Indian Collection"),
    "confidence_auto_add": float(os.environ.get("CONFIDENCE_AUTO_ADD", "0.8")),
    "confidence_review_min": float(os.environ.get("CONFIDENCE_REVIEW_MIN", "0.4")),
    "confidence_review_max": float(os.environ.get("CONFIDENCE_REVIEW_MAX", "0.7")),
    "genius_delay": float(os.environ.get("GENIUS_DELAY", "1.2")),
    "genius_max_retries": int(os.environ.get("GENIUS_MAX_RETRIES", "5")),
    "spotify_batch_size": 100,
    "needs_review_csv": "needs_review.csv",
    "songs_csv": os.environ.get("SPOTIFY_SONGS_CSV", "indian_songs.csv"),
    "model_dir": os.environ.get("INDICLID_MODEL_DIR"),
}

# One playlist per language; each language can have multiple script codes (native + Latin/romanized).
# Covers 5 major Indian languages: Hindi, Tamil, Telugu, Malayalam, Kannada.
LANGUAGE_PLAYLISTS = {
    "Hindi": ["hin_Deva", "hin_Latn"],
    "Tamil": ["tam_Tamil", "tam_Latn"],
    "Telugu": ["tel_Telu", "tel_Latn"],
    "Malayalam": ["mal_Mlym", "mal_Latn"],
    "Kannada": ["kan_Knda", "kan_Latn"],
}

# Spotify OAuth scopes
SCOPE = "user-library-read playlist-modify-public playlist-modify-private"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# SQLite progress DB
# -----------------------------------------------------------------------------
def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tracks (
            track_id TEXT PRIMARY KEY,
            name TEXT,
            artists TEXT,
            added_at TEXT,
            lyrics TEXT,
            lid_lang TEXT,
            lid_confidence REAL,
            lid_model TEXT,
            status TEXT DEFAULT 'pending',
            languages TEXT,
            language_confidences TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
        CREATE INDEX IF NOT EXISTS idx_tracks_lyrics ON tracks(lyrics);
    """)
    # Migration: add new columns if table already existed
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(tracks)")
    cols = {row[1] for row in cur.fetchall()}
    if "languages" not in cols:
        conn.execute("ALTER TABLE tracks ADD COLUMN languages TEXT")
    if "language_confidences" not in cols:
        conn.execute("ALTER TABLE tracks ADD COLUMN language_confidences TEXT")
    conn.commit()


def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(CONFIG["db_path"])


def upsert_track(
    conn: sqlite3.Connection,
    track_id: str,
    name: str,
    artists: str,
    added_at: str,
    lyrics: str | None = None,
    lid_lang: str | None = None,
    lid_confidence: float | None = None,
    lid_model: str | None = None,
    status: str | None = None,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tracks (track_id, name, artists, added_at, lyrics, lid_lang, lid_confidence, lid_model, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, status))
        ON CONFLICT(track_id) DO UPDATE SET
            name = excluded.name,
            artists = excluded.artists,
            added_at = excluded.added_at,
            lyrics = COALESCE(excluded.lyrics, lyrics),
            lid_lang = COALESCE(excluded.lid_lang, lid_lang),
            lid_confidence = COALESCE(excluded.lid_confidence, lid_confidence),
            lid_model = COALESCE(excluded.lid_model, lid_model),
            status = COALESCE(excluded.status, status)
        """,
        (
            track_id,
            name,
            artists,
            added_at,
            lyrics,
            lid_lang,
            lid_confidence,
            lid_model,
            status,
        ),
    )
    conn.commit()


def update_language_result(
    conn: sqlite3.Connection,
    track_id: str,
    language_confidences: dict[str, float],
) -> None:
    """Store per-language confidences and set status from thresholds."""
    if not language_confidences:
        conn.execute(
            "UPDATE tracks SET languages=?, language_confidences=?, lid_lang=?, lid_confidence=?, status=? WHERE track_id=?",
            ("[]", "{}", "other", 0.0, "skip", track_id),
        )
        conn.commit()
        return
    # Primary lang for backward compat: best South Asian
    best_lang = max(language_confidences, key=language_confidences.get)
    best_conf = language_confidences[best_lang]
    languages = list(language_confidences.keys())
    if any(c >= CONFIG["confidence_auto_add"] for c in language_confidences.values()):
        status = "add"
    elif any(CONFIG["confidence_review_min"] <= c <= CONFIG["confidence_review_max"] for c in language_confidences.values()):
        status = "review"
    else:
        status = "skip"
    conn.execute(
        "UPDATE tracks SET languages=?, language_confidences=?, lid_lang=?, lid_confidence=?, lid_model=?, status=? WHERE track_id=?",
        (json.dumps(languages), json.dumps(language_confidences), best_lang, best_conf, "IndicLID", status, track_id),
    )
    conn.commit()


def get_tracks_missing_lyrics(conn: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT track_id, name, artists, COALESCE(added_at, '') FROM tracks WHERE (lyrics IS NULL OR lyrics = '') AND status != 'skip'"
    )
    return cur.fetchall()


def get_tracks_missing_lid(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT track_id, lyrics FROM tracks WHERE lyrics IS NOT NULL AND lyrics != '' AND (language_confidences IS NULL OR language_confidences = '')"
    )
    return cur.fetchall()


def get_track_uris_for_language(conn: sqlite3.Connection, lang_codes: list[str]) -> list[str]:
    """Track IDs that have any of the given lang codes with confidence >= auto_add threshold."""
    cur = conn.cursor()
    out = []
    cur.execute(
        "SELECT track_id, language_confidences FROM tracks WHERE status IN ('add', 'review') AND language_confidences IS NOT NULL AND language_confidences != ''"
    )
    for track_id, j in cur.fetchall():
        if not j:
            continue
        try:
            confs = json.loads(j)
        except json.JSONDecodeError:
            continue
        if any(confs.get(lc, 0) >= CONFIG["confidence_auto_add"] for lc in lang_codes):
            out.append(track_id)
    return out


def get_all_tracks_with_languages(conn: sqlite3.Connection) -> list[tuple]:
    """All tracks that have at least one South Asian language (add or review). For CSV export."""
    cur = conn.cursor()
    cur.execute(
        "SELECT track_id, name, artists, added_at, languages, language_confidences FROM tracks WHERE status IN ('add', 'review') AND language_confidences IS NOT NULL AND language_confidences != '' ORDER BY added_at"
    )
    return cur.fetchall()


def get_tracks_for_review(conn: sqlite3.Connection) -> list[tuple]:
    cur = conn.cursor()
    cur.execute(
        "SELECT track_id, name, artists, lid_lang, lid_confidence, lid_model FROM tracks WHERE status = 'review'"
    )
    return cur.fetchall()


# -----------------------------------------------------------------------------
# Genius lyrics with exponential backoff
# -----------------------------------------------------------------------------
def fetch_lyrics_with_backoff(genius, title: str, artist: str) -> str | None:
    delay = CONFIG["genius_delay"]
    for attempt in range(CONFIG["genius_max_retries"]):
        try:
            time.sleep(delay)
            song = genius.search_song(title, artist)
            if song and getattr(song, "lyrics", None):
                return song.lyrics.strip()
            return None
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                delay = min(60, delay * 2)
                logger.warning("Genius rate limit; backing off to %.1fs", delay)
            else:
                logger.debug("Genius error for %s - %s: %s", title, artist, e)
            if attempt == CONFIG["genius_max_retries"] - 1:
                logger.warning("Gave up lyrics for %s - %s after %d attempts", title, artist, CONFIG["genius_max_retries"])
                return None
            time.sleep(delay)
    return None


# -----------------------------------------------------------------------------
# Spotify
# -----------------------------------------------------------------------------
def get_spotify_client():
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:9090")
    cache_path = os.environ.get("SPOTIFY_CACHE_PATH", ".cache")
    if not client_id or not client_secret:
        raise RuntimeError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
    # Explicit cache_path: run once locally to create .cache, then upload it to your cluster (headless nodes have no browser).
    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=SCOPE,
        cache_path=cache_path,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def fetch_all_liked_tracks(sp) -> list[dict]:
    tracks = []
    offset = 0
    limit = 50
    while True:
        resp = sp.current_user_saved_tracks(limit=limit, offset=offset)
        items = resp.get("items", [])
        if not items:
            break
        for item in items:
            t = item.get("track") or {}
            track_id = t.get("id")
            if not track_id:
                continue
            name = t.get("name") or ""
            artists = ", ".join(a.get("name", "") for a in (t.get("artists") or []))
            added_at = (item.get("added_at") or "")[:19]
            tracks.append(
                {
                    "track_id": track_id,
                    "name": name,
                    "artists": artists,
                    "added_at": added_at,
                }
            )
        offset += limit
        if not resp.get("next"):
            break
    return tracks


def find_or_create_playlist(sp, name: str, description: str = "South Asian tracks from Liked Songs"):
    user_id = sp.current_user()["id"]
    playlists = sp.user_playlists(user_id, limit=50)
    while playlists:
        for p in playlists.get("items", []):
            if p.get("name") == name:
                return p["id"]
        if not playlists.get("next"):
            break
        playlists = sp.next(playlists)
    pl = sp.user_playlist_create(user_id, name, public=False, description=description)
    return pl["id"]


def add_tracks_to_playlist(sp, playlist_id: str, track_uris: list[str]) -> None:
    batch_size = CONFIG["spotify_batch_size"]
    for i in range(0, len(track_uris), batch_size):
        batch = track_uris[i : i + batch_size]
        sp.playlist_add_items(playlist_id, batch)
        if i + batch_size < len(track_uris):
            time.sleep(0.5)


def replace_playlist_tracks(sp, playlist_id: str, track_uris: list[str]) -> None:
    batch_size = CONFIG["spotify_batch_size"]
    sp.playlist_replace_items(playlist_id, [])
    for i in range(0, len(track_uris), batch_size):
        batch = track_uris[i : i + batch_size]
        sp.playlist_add_items(playlist_id, batch)
        if i + batch_size < len(track_uris):
            time.sleep(0.5)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run():
    conn = get_conn()
    init_db(conn)

    # ----- 1) Spotify: fetch all liked tracks and persist -----
    logger.info("Connecting to Spotify...")
    sp = get_spotify_client()
    liked = fetch_all_liked_tracks(sp)
    logger.info("Fetched %d liked tracks. Syncing to DB...", len(liked))
    for t in liked:
        upsert_track(
            conn,
            track_id=t["track_id"],
            name=t["name"],
            artists=t["artists"],
            added_at=t["added_at"],
        )
    conn.commit()

    # ----- 2) Genius: fetch lyrics for tracks missing them -----
    genius_token = os.environ.get("GENIUS_ACCESS_TOKEN")
    if genius_token:
        import lyricsgenius
        genius = lyricsgenius.Genius(genius_token, sleep_time=CONFIG["genius_delay"], retries=2)
        genius.remove_section_headers = True
        missing = get_tracks_missing_lyrics(conn)
        logger.info("Fetching lyrics for %d tracks...", len(missing))
        for track_id, name, artists, added_at in tqdm(missing, desc="Lyrics"):
            lyrics = fetch_lyrics_with_backoff(genius, name, artists.split(",")[0].strip() if artists else "")
            upsert_track(conn, track_id=track_id, name=name, artists=artists, added_at=added_at, lyrics=lyrics or "")
        conn.commit()
    else:
        logger.warning("GENIUS_ACCESS_TOKEN not set; skipping lyrics fetch. Set it for full pipeline.")

    # ----- 3) IndicLID: run LID and set status -----
    try:
        from indiclid_wrapper import IndicLIDWrapper
    except Exception as e:
        logger.error("IndicLID not available: %s. See requirements.txt.", e)
        return
    lid = IndicLIDWrapper(model_dir=CONFIG["model_dir"])
    to_lid = get_tracks_missing_lid(conn)
    logger.info("Running IndicLID on %d tracks...", len(to_lid))
    for track_id, lyrics in tqdm(to_lid, desc="LID"):
        if not lyrics or not lyrics.strip():
            update_language_result(conn, track_id, {})
            continue
        confidences = lid.get_south_asian_language_confidences(lyrics)
        update_language_result(conn, track_id, confidences)
    conn.commit()

    # ----- 4) Needs-review CSV -----
    review_rows = get_tracks_for_review(conn)
    if review_rows:
        df = pd.DataFrame(
            review_rows,
            columns=["track_id", "name", "artists", "lid_lang", "lid_confidence", "lid_model"],
        )
        df.to_csv(CONFIG["needs_review_csv"], index=False)
        logger.info("Wrote %d rows to %s", len(df), CONFIG["needs_review_csv"])

    # ----- 5) CSV of all Indian-language songs -----
    rows = get_all_tracks_with_languages(conn)
    if rows:
        # Human-readable names for IndicLID codes (native + Latin)
        lang_display = {
            "hin_Deva": "Hindi (Devanagari)", "hin_Latn": "Hindi (Latin)",
            "tam_Tamil": "Tamil", "tam_Latn": "Tamil (Latin)",
            "tel_Telu": "Telugu", "tel_Latn": "Telugu (Latin)",
            "mal_Mlym": "Malayalam", "mal_Latn": "Malayalam (Latin)",
            "kan_Knda": "Kannada", "kan_Latn": "Kannada (Latin)",
        }
        csv_rows = []
        for track_id, name, artists, added_at, languages_json, confidences_json in rows:
            try:
                langs = json.loads(languages_json or "[]")
                confs = json.loads(confidences_json or "{}")
            except json.JSONDecodeError:
                langs, confs = [], {}
            languages_str = ", ".join(lang_display.get(l, l) for l in langs)
            row = {
                "track_id": track_id,
                "name": name,
                "artists": artists,
                "added_at": added_at,
                "languages": languages_str,
            }
            for lang_name, lang_codes in LANGUAGE_PLAYLISTS.items():
                key = lang_name.lower()
                row[f"{key}_confidence"] = round(max(confs.get(lc, 0) for lc in lang_codes), 4)
                row[f"in_{key}_playlist"] = any(confs.get(lc, 0) >= CONFIG["confidence_auto_add"] for lc in lang_codes)
            csv_rows.append(row)
        df = pd.DataFrame(csv_rows)
        df.to_csv(CONFIG["songs_csv"], index=False)
        logger.info("Wrote %d songs to %s", len(df), CONFIG["songs_csv"])

    # ----- 6) Per-language playlists -----
    for lang_name, lang_codes in LANGUAGE_PLAYLISTS.items():
        track_ids = get_track_uris_for_language(conn, lang_codes)
        if not track_ids:
            logger.info("No tracks for '%s'; skipping playlist.", lang_name)
            continue
        playlist_title = f"{CONFIG['playlist_name']} - {lang_name}"
        playlist_id = find_or_create_playlist(sp, playlist_title, description=f"{lang_name} tracks from Liked Songs")
        uris = [f"spotify:track:{tid}" for tid in track_ids]
        replace_playlist_tracks(sp, playlist_id, uris)
        logger.info("Updated playlist '%s' with %d tracks.", playlist_title, len(uris))

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    run()
