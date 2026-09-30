"""
Audio from a YouTube link, via yt-dlp (the best audio-only stream; ffmpeg converts it on import).
Playlist and channel links expand to their videos, for batch import.

YouTube needs a JavaScript runtime for yt-dlp: the `deno` package from requirements puts one on
the PATH. Only YouTube links are accepted: the server should not fetch arbitrary URLs.
"""

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
                 "youtu.be", "www.youtube-nocookie.com", "youtube-nocookie.com"}
VIDEO_ID = re.compile(r"[\w-]{11}")


def video_id(url: str) -> str | None:
    """The 11-character video id of a YouTube link, or None if it is not a video link."""
    u = urlparse(url.strip())
    if u.scheme not in ("http", "https") or (u.hostname or "").lower() not in YOUTUBE_HOSTS:
        return None
    parts = [p for p in u.path.split("/") if p]
    if u.hostname.lower() == "youtu.be":
        vid = parts[0] if parts else None
    elif parts[:1] == ["watch"]:
        vid = (parse_qs(u.query).get("v") or [None])[0]
    elif len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v"):
        vid = parts[1]
    else:
        vid = None
    return vid if vid and VIDEO_ID.fullmatch(vid) else None


def canonical_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def download(vid: str, dest_dir: Path, progress=None, info: dict | None = None) -> tuple[Path, str]:
    """Download the audio of one video into dest_dir as source.<ext>. Returns (path, title);
    `info`, if given, is filled with the video's channel, channel_id, upload_date and language."""
    import yt_dlp

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress(min(1.0, d.get("downloaded_bytes", 0) / total))

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(Path(dest_dir) / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        video = ydl.extract_info(canonical_url(vid), download=True)
    if info is not None:
        info.update({k: video.get(k) for k in ("channel", "channel_id", "upload_date", "language")})
    return Path(video["requested_downloads"][0]["filepath"]), video.get("title") or vid


def collection_url(url: str) -> str | None:
    """A normalised playlist or channel link, or None. Channels list their uploads ("/videos")."""
    u = urlparse(url.strip())
    if u.scheme not in ("http", "https") or (u.hostname or "").lower() not in YOUTUBE_HOSTS:
        return None
    parts = [p for p in u.path.split("/") if p]
    lst = (parse_qs(u.query).get("list") or [None])[0]
    if lst and re.fullmatch(r"[\w-]{10,64}", lst) and parts[:1] in (["playlist"], ["watch"]):
        return f"https://www.youtube.com/playlist?list={lst}"
    if parts and (parts[0].startswith("@") or parts[0] in ("channel", "c", "user")):
        base = parts[:1] if parts[0].startswith("@") else parts[:2]
        if len(base) == (1 if parts[0].startswith("@") else 2) and all(re.fullmatch(r"[@\w.-]+", p) for p in base):
            return "https://www.youtube.com/" + "/".join(base) + "/videos"
    return None


def expand(url: str, limit: int = 50) -> list[dict]:
    """The videos of a playlist or channel: [{"id", "title", "duration"}], in the order YouTube
    lists them. Nothing is downloaded."""
    import yt_dlp

    url = collection_url(url)
    if url is None:
        raise ValueError("not a YouTube playlist or channel link")
    opts = {"extract_flat": "in_playlist", "playlistend": limit, "quiet": True,
            "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    out = []
    for e in info.get("entries") or []:
        vid = (e or {}).get("id")
        if vid and VIDEO_ID.fullmatch(vid):
            out.append({"id": vid, "title": e.get("title") or vid, "duration": e.get("duration")})
    return out[:limit]
