import json
import os
import re
import urllib.parse
from curl_cffi import requests as cffi_requests

# Complete Chrome 124 Client-Hint and Navigation Headers
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


def get_proxy() -> str | None:
    """Loads optional proxy from Streamlit secrets or environment if AWS IP is challenged."""
    try:
        import streamlit as st
        proxy = st.secrets.get("PIXABAY_PROXY")
        if proxy and proxy.strip():
            return proxy.strip()
    except Exception:
        pass
    return os.getenv("PIXABAY_PROXY", "").strip() or None


def create_warmed_session() -> cffi_requests.Session:
    """
    Initializes a Chrome 124 impersonated session, binds optional proxies,
    and lands on Pixabay's homepage to harvest initial cookies and connection tokens.
    """
    proxy = get_proxy()
    proxies = {"http": proxy, "https": proxy} if proxy else None

    session = cffi_requests.Session(impersonate="chrome124", proxies=proxies)

    # Warm-up request to establish session state and pass initial Cloudflare clearance
    try:
        warm_headers = dict(BROWSER_HEADERS)
        warm_headers["Sec-Fetch-Site"] = "none"
        session.get("https://pixabay.com/", headers=warm_headers, timeout=15)
    except Exception as e:
        print(f"[Pixabay Scraper] Warning during session warmup: {e}")

    return session


def download_stream(url: str, output_path: str, session: cffi_requests.Session = None) -> str:
    """Streams binary media chunks to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if session is None:
        session = cffi_requests.Session(impersonate="chrome124")

    # Asset requests use empty or no-cors fetch headers
    asset_headers = {
        "Accept": "*/*",
        "Referer": "https://pixabay.com/",
        "Sec-Fetch-Dest": "video" if output_path.endswith(".mp4") else "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
    }

    response = session.get(url, headers=asset_headers, stream=True, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed for {url} (HTTP {response.status_code})")

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    if os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Downloaded file '{output_path}' is empty.")

    print(f"[Pixabay Scraper] Saved: {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


def extract_next_data(html: str) -> dict | None:
    """Extracts Pixabay's hydration JSON state object from script tags."""
    match = re.search(r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>', html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            return None
    return None


def fetch_landscape_video(query: str, output_path: str) -> str:
    """
    Scrapes Pixabay for landscape video footage using curl_cffi with full Chrome 124 client hints.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Video search query cannot be empty.")

    encoded_query = urllib.parse.quote(clean_query)
    search_url = f"https://pixabay.com/videos/search/{encoded_query}/?order=ec"

    session = create_warmed_session()

    search_headers = dict(BROWSER_HEADERS)
    search_headers["Referer"] = "https://pixabay.com/"

    print(f"[Pixabay Scraper] Querying videos for: '{clean_query}' via curl_cffi")
    response = session.get(search_url, headers=search_headers, timeout=20)

    if response.status_code == 403:
        raise RuntimeError(
            "Pixabay Cloudflare challenge triggered (HTTP 403). "
            "To bypass AWS data center filtering without using an API, "
            "add a proxy to your secrets as: PIXABAY_PROXY = 'http://user:pass@host:port'"
        )
    if response.status_code != 200:
        raise RuntimeError(f"Pixabay video search failed with HTTP {response.status_code}")

    html = response.text
    video_url = None

    # Strategy 1: JSON Hydration tree
    next_data = extract_next_data(html)
    if next_data:
        try:
            def search_media_nodes(data):
                if isinstance(data, dict):
                    if "videos" in data and isinstance(data["videos"], dict):
                        vids = data["videos"]
                        for quality in ["large", "medium", "small"]:
                            cand = vids.get(quality, {})
                            cand_url = cand.get("url")
                            w = cand.get("width", 0)
                            h = cand.get("height", 0)
                            if cand_url and (w >= h):
                                return cand_url
                    for v in data.values():
                        res = search_media_nodes(v)
                        if res:
                            return res
                elif isinstance(data, list):
                    for item in data:
                        res = search_media_nodes(item)
                        if res:
                            return res
                return None

            video_url = search_media_nodes(next_data)
        except Exception as e:
            print(f"[Pixabay Scraper] Notice: __NEXT_DATA__ parse fallback: {e}")

    # Strategy 2: Direct CDN regular expression match
    if not video_url:
        candidates = re.findall(r'https://cdn\.pixabay\.com/video/[^"\'\s<>]+\.mp4', html)
        if candidates:
            prioritized = [c for c in candidates if "_medium" in c or "_large" in c or "_1080p" in c]
            video_url = prioritized[0] if prioritized else candidates[0]

    if not video_url:
        raise RuntimeError(f"No Pixabay landscape video found for query: '{query}'")

    print(f"[Pixabay Scraper] Extracted video CDN link: {video_url}")
    return download_stream(video_url, output_path, session=session)


def fetch_image(query: str, output_path: str) -> str:
    """
    Scrapes Pixabay for landscape stock images/photos using curl_cffi with full Chrome 124 client hints.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Image search query cannot be empty.")

    encoded_query = urllib.parse.quote(clean_query)
    search_url = f"https://pixabay.com/photos/search/{encoded_query}/?order=ec"

    session = create_warmed_session()

    search_headers = dict(BROWSER_HEADERS)
    search_headers["Referer"] = "https://pixabay.com/"

    print(f"[Pixabay Scraper] Querying photos for: '{clean_query}' via curl_cffi")
    response = session.get(search_url, headers=search_headers, timeout=20)

    if response.status_code == 403:
        raise RuntimeError(
            "Pixabay Cloudflare challenge triggered (HTTP 403). "
            "Add a proxy to secrets as: PIXABAY_PROXY = 'http://user:pass@host:port'"
        )
    if response.status_code != 200:
        raise RuntimeError(f"Pixabay photo search failed with HTTP {response.status_code}")

    html = response.text
    image_url = None

    # Strategy 1: JSON Hydration tree
    next_data = extract_next_data(html)
    if next_data:
        try:
            def search_image_nodes(data):
                if isinstance(data, dict):
                    if "largeImageURL" in data:
                        return data["largeImageURL"]
                    if "fullPXURL" in data:
                        return data["fullPXURL"]
                    if "webformatURL" in data:
                        return data["webformatURL"]
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

            image_url = search_image_nodes(next_data)
        except Exception as e:
            print(f"[Pixabay Scraper] Notice: __NEXT_DATA__ image parse fallback: {e}")

    # Strategy 2: Direct CDN regular expression match
    if not image_url:
        candidates = re.findall(r'https://cdn\.pixabay\.com/photo/[^"\'\s<>]+\.(?:jpg|jpeg|png)', html)
        if candidates:
            large_candidates = [c for c in candidates if "_1280." in c or "_960." in c]
            image_url = large_candidates[0] if large_candidates else candidates[0]

    if not image_url:
        raise RuntimeError(f"No Pixabay image found for query: '{query}'")

    print(f"[Pixabay Scraper] Extracted image CDN link: {image_url}")
    return download_stream(image_url, output_path, session=session)
