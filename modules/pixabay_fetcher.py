import json
import os
import re
import shutil
import subprocess
import urllib.parse
from curl_cffi import requests as cffi_requests

CHROME_VERSION = "chrome124"

BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Linux"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

WIKI_HEADERS = {
    "User-Agent": "VideoEngineBot/1.0 (Streamlit Cloud; media-pipeline@internal.dev)",
    "Accept": "application/json, text/html, */*",
}


def is_landscape_video(video_path: str) -> bool:
    """Uses ffprobe to confirm the video is landscape (width >= height)."""
    if not shutil.which("ffprobe"):
        return True

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dims = res.stdout.strip().split("x")
        if len(dims) == 2:
            return int(dims[0]) >= int(dims[1])
    except Exception:
        pass
    return True


def transcode_to_silent_mp4(temp_input: str, output_mp4: str) -> bool:
    """
    Standardizes any downloaded format (WebM/OGV/MP4) into a silent .mp4:
    - Uses '-map 0:v:0' to drop audio and avoid Opus packet header warnings.
    - Suppresses demuxer stderr output.
    """
    ext = os.path.splitext(temp_input)[1].lower()

    if ext == ".mp4":
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", temp_input,
            "-map", "0:v:0",
            "-c:v", "copy",
            "-an",
            output_mp4
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", temp_input,
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-an",
            output_mp4
        ]

    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0 and os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 0


def upgrade_mixkit_url(session: cffi_requests.Session, url: str) -> str:
    """Upgrades 360p Mixkit preview links to 720p or 1080p HD streams if available."""
    upgrades = [
        re.sub(r'-(?:360|small|thumb)\.mp4$', '-720.mp4', url),
        re.sub(r'-(?:360|small|thumb)\.mp4$', '-1080.mp4', url),
    ]
    for upgraded_url in upgrades:
        if upgraded_url != url:
            try:
                head_resp = session.head(upgraded_url, timeout=4)
                if head_resp.status_code == 200:
                    return upgraded_url
            except Exception:
                pass
    return url


def download_stream(session: cffi_requests.Session, url: str, output_path: str, headers: dict) -> bool:
    """Streams binary content to disk using curl_cffi."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        res = session.get(url, headers=headers, stream=True, timeout=35)
        if res.status_code != 200:
            return False

        with open(output_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception:
        return False


# =====================================================================
# PRIMARY VIDEO ENGINE: MIXKIT
# =====================================================================
def _search_mixkit_video(session: cffi_requests.Session, query: str, output_path: str) -> bool:
    clean_q = query.strip()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", clean_q.lower()).strip("-")

    search_urls = [
        f"https://mixkit.co/free-stock-video/{slug}/",
        f"https://mixkit.co/free-stock-video/?q={urllib.parse.quote(clean_q)}",
    ]

    html = ""
    for url in search_urls:
        try:
            resp = session.get(url, headers=BROWSER_HEADERS, timeout=12)
            if resp.status_code == 200 and "assets.mixkit.co" in resp.text:
                html = resp.text
                break
        except Exception:
            continue

    if not html:
        return False

    candidate_urls = []
    # Extract from JSON-LD schema
    json_lds = re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL)
    for jld in json_lds:
        try:
            data = json.loads(jld)
            for item in data.get("@graph", []):
                if item.get("@type") == "VideoObject":
                    vid_url = item.get("contentUrl") or item.get("embedUrl")
                    if vid_url and vid_url not in candidate_urls:
                        candidate_urls.append(vid_url)
        except Exception:
            pass

    # Regex fallback
    if not candidate_urls:
        raw_assets = re.findall(r'https://assets\.mixkit\.co/videos/[^"\'\s<>]+\.mp4', html)
        unique = {}
        for link in raw_assets:
            base = re.sub(r'-(?:large|medium|small|thumb|\d+)\.mp4$', '', link)
            if base not in unique or "-720" in link or "-1080" in link:
                unique[base] = link
        candidate_urls = list(unique.values())

    asset_headers = {
        "Accept": "*/*",
        "Referer": "https://mixkit.co/",
        "Sec-Fetch-Dest": "video",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }

    temp_path = f"{output_path}.temp.mp4"
    for raw_url in candidate_urls:
        url = upgrade_mixkit_url(session, raw_url)
        if download_stream(session, url, temp_path, asset_headers):
            if is_landscape_video(temp_path) and transcode_to_silent_mp4(temp_path, output_path):
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(f"[MediaFetcher] Mixkit video saved: {output_path}")
                return True
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return False


# =====================================================================
# FALLBACK VIDEO ENGINE: WIKIMEDIA COMMONS
# =====================================================================
def _search_wikimedia_video(session: cffi_requests.Session, query: str, output_path: str) -> bool:
    clean_q = query.strip()
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{clean_q} filetype:video",
        "gsrnamespace": 6,
        "gsrlimit": 15,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|dimensions",
    }

    try:
        resp = session.get(api_url, params=params, headers=WIKI_HEADERS, timeout=12)
        pages = resp.json().get("query", {}).get("pages", {})
    except Exception:
        return False

    candidate_urls = []
    for page_id, info in pages.items():
        image_info = info.get("imageinfo", [{}])[0]
        url = image_info.get("url")
        w = image_info.get("width", 0)
        h = image_info.get("height", 0)
        if url and w >= h and w > 0:
            candidate_urls.append(url)

    asset_headers = {
        "Accept": "*/*",
        "Referer": "https://commons.wikimedia.org/",
        "User-Agent": WIKI_HEADERS["User-Agent"],
    }

    for url in candidate_urls:
        raw_ext = os.path.splitext(urllib.parse.urlparse(url).path)[-1] or ".webm"
        temp_path = f"{output_path}.temp{raw_ext}"
        if download_stream(session, url, temp_path, asset_headers):
            if transcode_to_silent_mp4(temp_path, output_path):
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(f"[MediaFetcher] Wikimedia video saved: {output_path}")
                return True
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return False


# =====================================================================
# PUBLIC EXPORTED FUNCTIONS (Matches Pipeline Interface)
# =====================================================================
def fetch_landscape_video(query: str, output_path: str) -> str:
    """
    Fetches a landscape stock video (Mixkit first, Wikimedia fallback).
    Ensures the final file is a clean, silent .mp4 without Opus errors.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Video query cannot be empty.")

    session = cffi_requests.Session(impersonate=CHROME_VERSION)

    # 1. Try Mixkit
    print(f"[MediaFetcher] Searching Mixkit for video: '{clean_query}'...")
    if _search_mixkit_video(session, clean_query, output_path):
        return output_path

    # 2. Fallback to Wikimedia Commons
    print(f"[MediaFetcher] Mixkit yielded no landscape match. Falling back to Wikimedia for: '{clean_query}'...")
    if _search_wikimedia_video(session, clean_query, output_path):
        return output_path

    raise RuntimeError(f"No landscape video found across Mixkit or Wikimedia for: '{query}'")


def fetch_image(query: str, output_path: str) -> str:
    """
    Fetches a landscape photo/image from Wikimedia Commons.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Image query cannot be empty.")

    session = cffi_requests.Session(impersonate=CHROME_VERSION)
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"{clean_query} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": 15,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|dimensions",
    }

    print(f"[MediaFetcher] Searching Wikimedia for image: '{clean_query}'...")
    resp = session.get(api_url, params=params, headers=WIKI_HEADERS, timeout=12)
    if resp.status_code != 200:
        raise RuntimeError(f"Wikimedia image query failed (HTTP {resp.status_code})")

    pages = resp.json().get("query", {}).get("pages", {})
    landscape_urls = []
    for page_id, info in pages.items():
        image_info = info.get("imageinfo", [{}])[0]
        url = image_info.get("url")
        w = image_info.get("width", 0)
        h = image_info.get("height", 0)
        if url and (w >= h or w == 0):
            landscape_urls.append(url)

    if not landscape_urls:
        raise RuntimeError(f"No landscape image found on Wikimedia for: '{query}'")

    asset_headers = {
        "Accept": "*/*",
        "Referer": "https://commons.wikimedia.org/",
        "User-Agent": WIKI_HEADERS["User-Agent"],
    }

    for url in landscape_urls:
        if download_stream(session, url, output_path, asset_headers):
            print(f"[MediaFetcher] Wikimedia image saved: {output_path}")
            return output_path

    raise RuntimeError(f"Failed to download image from Wikimedia for: '{query}'")
