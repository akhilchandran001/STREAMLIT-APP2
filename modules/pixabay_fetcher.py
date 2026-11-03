import json
import os
import random
import re
import urllib.parse
from curl_cffi import requests as cffi_requests
import requests

# Full Chrome 124 Client-Hint and Navigation Headers
BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def get_proxy_candidates() -> list[str]:
    """
    Retrieves all available proxies from Streamlit secrets or environment variables.
    Returns a shuffled list for automatic load-balancing and retry failover.
    """
    candidates = []

    try:
        import streamlit as st

        # 1. Multi-proxy pool (e.g. PIXABAY_PROXIES = ["http://...", ...])
        pool = st.secrets.get("PIXABAY_PROXIES")
        if pool and isinstance(pool, list):
            candidates.extend([p.strip() for p in pool if isinstance(p, str) and p.strip()])

        # 2. Single proxy string (e.g. PIXABAY_PROXY = "http://...")
        single = st.secrets.get("PIXABAY_PROXY")
        if single and isinstance(single, str) and single.strip():
            if single.strip() not in candidates:
                candidates.append(single.strip())
    except Exception:
        pass

    # 3. Environment variable fallback
    env_proxy = os.getenv("PIXABAY_PROXY", "").strip()
    if env_proxy and env_proxy not in candidates:
        candidates.append(env_proxy)

    # Randomize rotation order to distribute requests across the pool
    random.shuffle(candidates)
    return candidates


def fetch_public_proxy_fallback() -> list[str]:
    """Emergency fallback pool if no Webshare/private proxies are configured in secrets."""
    url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=4000&country=all&ssl=yes&anonymity=elite"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            lines = [line.strip() for line in r.text.splitlines() if line.strip()]
            return [f"http://{ip}" for ip in lines[:10]]
    except Exception as e:
        print(f"[Pixabay Fetcher] Notice: Could not fetch public proxy pool: {e}")
    return []


def create_session(proxy: str | None = None) -> cffi_requests.Session:
    """Builds a curl_cffi session configured with Chrome 124 TLS fingerprint and optional proxy."""
    proxies = {"http": proxy, "https": proxy} if proxy else None
    return cffi_requests.Session(impersonate="chrome124", proxies=proxies)


def download_stream(url: str, output_path: str, proxy: str | None = None) -> str:
    """
    Streams media content directly to local disk in binary chunks.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    session = create_session(proxy)

    asset_headers = {
        "Accept": "*/*",
        "Referer": "https://pixabay.com/",
        "Sec-Fetch-Dest": "video" if output_path.endswith(".mp4") else "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }

    response = session.get(url, headers=asset_headers, stream=True, timeout=40)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed for {url} (HTTP {response.status_code})")

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Downloaded media file '{output_path}' is missing or empty.")

    print(f"[Pixabay Fetcher] Successfully saved: {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


def extract_next_data(html: str) -> dict | None:
    """Extracts Pixabay's __NEXT_DATA__ JSON state tree from script tags."""
    match = re.search(r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>', html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def extract_media_from_html(html: str, media_type: str = "video") -> str | None:
    """
    Locates target media URL using structured JSON tree extraction first,
    falling back to regular expression scanning against Pixabay CDN paths.
    """
    # Strategy 1: JSON Hydration parsing
    next_data = extract_next_data(html)
    if next_data:
        try:
            if media_type == "video":
                def search_video_nodes(data):
                    if isinstance(data, dict):
                        if "videos" in data and isinstance(data["videos"], dict):
                            vids = data["videos"]
                            for quality in ["large", "medium", "small"]:
                                cand = vids.get(quality, {})
                                cand_url = cand.get("url")
                                w = cand.get("width", 0)
                                h = cand.get("height", 0)
                                if cand_url and (w >= h):  # Prefer landscape
                                    return cand_url
                        for v in data.values():
                            res = search_video_nodes(v)
                            if res:
                                return res
                    elif isinstance(data, list):
                        for item in data:
                            res = search_video_nodes(item)
                            if res:
                                return res
                    return None

                found_url = search_video_nodes(next_data)
                if found_url:
                    return found_url
            else:
                def search_image_nodes(data):
                    if isinstance(data, dict):
                        for k in ["largeImageURL", "fullPXURL", "webformatURL"]:
                            if k in data and isinstance(data[k], str) and data[k].startswith("http"):
                                return data[k]
                        for v in data.values():
                            res = search_image_nodes(v)
                            if res:
                                return res
                    elif isinstance(data, list):
                        for item in data:
                            res = search_image_nodes(item)
                            if res:
                                return res
                    return None

                found_url = search_image_nodes(next_data)
                if found_url:
                    return found_url
        except Exception as e:
            print(f"[Pixabay Fetcher] Hydration state walk notice: {e}")

    # Strategy 2: Regex scanning on Pixabay CDN domain patterns
    if media_type == "video":
        candidates = re.findall(r'https://cdn\.pixabay\.com/video/[^"\'\s<>]+\.mp4', html)
        if candidates:
            # Prioritize high and medium resolution formats
            preferred = [c for c in candidates if "_large" in c or "_medium" in c or "_1080p" in c]
            return preferred[0] if preferred else candidates[0]
    else:
        candidates = re.findall(r'https://cdn\.pixabay\.com/photo/[^"\'\s<>]+\.(?:jpg|jpeg|png)', html)
        if candidates:
            preferred = [c for c in candidates if "_1280." in c or "_960." in c]
            return preferred[0] if preferred else candidates[0]

    return None


def execute_scrape_with_rotation(search_url: str, media_type: str = "video") -> tuple[str, str | None]:
    """
    Executes search query across configured proxy pool, automatically rotating
    to the next candidate if Cloudflare challenges (403) or timeouts occur.
    """
    configured_proxies = get_proxy_candidates()
    
    # If no proxies configured in secrets, try direct connection first
    attempt_queue = configured_proxies if configured_proxies else [None]

    for idx, proxy in enumerate(attempt_queue, start=1):
        proxy_label = proxy.split("@")[-1] if proxy and "@" in proxy else (proxy or "Direct Connection")
        print(f"[Pixabay Fetcher] Requesting ({idx}/{len(attempt_queue)}) via {proxy_label}...")

        try:
            session = create_session(proxy)

            # Warm-up request to establish TLS session and cookies
            warm_headers = dict(BROWSER_HEADERS)
            warm_headers["Sec-Fetch-Site"] = "none"
            session.get("https://pixabay.com/", headers=warm_headers, timeout=12)

            # Search query request
            query_headers = dict(BROWSER_HEADERS)
            query_headers["Referer"] = "https://pixabay.com/"
            resp = session.get(search_url, headers=query_headers, timeout=15)

            if resp.status_code == 200:
                media_url = extract_media_from_html(resp.text, media_type=media_type)
                if media_url:
                    print(f"[Pixabay Fetcher] Media URL resolved successfully: {media_url}")
                    return media_url, proxy
                else:
                    print(f"[Pixabay Fetcher] Warning: No media URLs found on page via {proxy_label}.")
            elif resp.status_code == 403:
                print(f"[Pixabay Fetcher] Cloudflare blocked request (403) on {proxy_label}. Rotating...")
            else:
                print(f"[Pixabay Fetcher] HTTP {resp.status_code} on {proxy_label}. Rotating...")

        except Exception as e:
            print(f"[Pixabay Fetcher] Proxy failure on {proxy_label}: {e}")
            continue

    # Emergency fallback: If user proxies failed or were absent, try public proxy pool
    print("[Pixabay Fetcher] Configured pool exhausted. Testing public proxy fallbacks...")
    public_pool = fetch_public_proxy_fallback()
    for pub_proxy in public_pool:
        try:
            session = create_session(pub_proxy)
            resp = session.get(search_url, headers=BROWSER_HEADERS, timeout=10)
            if resp.status_code == 200:
                media_url = extract_media_from_html(resp.text, media_type=media_type)
                if media_url:
                    return media_url, pub_proxy
        except Exception:
            continue

    raise RuntimeError(
        "All Pixabay scraping attempts were blocked by Cloudflare. "
        "Verify your Webshare proxies are saved in .streamlit/secrets.toml under PIXABAY_PROXIES."
    )


def fetch_landscape_video(query: str, output_path: str) -> str:
    """
    Public interface: Scrapes and downloads a landscape stock video from Pixabay.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Video search query cannot be empty.")

    encoded = urllib.parse.quote(clean_query)
    search_url = f"https://pixabay.com/videos/search/{encoded}/?order=ec"

    video_url, working_proxy = execute_scrape_with_rotation(search_url, media_type="video")
    return download_stream(video_url, output_path, proxy=working_proxy)


def fetch_image(query: str, output_path: str) -> str:
    """
    Public interface: Scrapes and downloads a stock landscape image from Pixabay.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Image search query cannot be empty.")

    encoded = urllib.parse.quote(clean_query)
    search_url = f"https://pixabay.com/photos/search/{encoded}/?order=ec"

    image_url, working_proxy = execute_scrape_with_rotation(search_url, media_type="image")
    return download_stream(image_url, output_path, proxy=working_proxy)
