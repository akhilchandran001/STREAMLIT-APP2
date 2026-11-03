import json
import os
import re
import urllib.parse
from curl_cffi import requests as cffi_requests
import requests

BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}


def get_configured_proxy() -> str | None:
    """Reads proxy from Streamlit secrets or env."""
    try:
        import streamlit as st
        proxy = st.secrets.get("PIXABAY_PROXY")
        if proxy and proxy.strip():
            return proxy.strip()
    except Exception:
        pass
    return os.getenv("PIXABAY_PROXY", "").strip() or None


def fetch_free_proxy_list() -> list[str]:
    """Fetches a fresh list of public elite HTTPS proxies."""
    url = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=4000&country=all&ssl=yes&anonymity=elite"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            lines = [line.strip() for line in r.text.splitlines() if line.strip()]
            return [f"http://{ip}" for ip in lines]
    except Exception as e:
        print(f"[Proxy Scraper] Warning: Failed to fetch public proxies: {e}")
    return []


def create_session(proxy: str = None) -> cffi_requests.Session:
    """Creates a curl_cffi session impersonating Chrome 124."""
    proxies = {"http": proxy, "https": proxy} if proxy else None
    return cffi_requests.Session(impersonate="chrome124", proxies=proxies)


def download_stream(url: str, output_path: str, proxy: str = None) -> str:
    """Streams binary file chunks to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    session = create_session(proxy)

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

    print(f"[Pixabay] Saved: {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path


def extract_media_url_from_html(html: str, media_type: str = "video") -> str | None:
    """Extracts either video or image CDN URL from HTML."""
    if media_type == "video":
        candidates = re.findall(r'https://cdn\.pixabay\.com/video/[^"\'\s<>]+\.mp4', html)
        if candidates:
            prioritized = [c for c in candidates if "_medium" in c or "_large" in c or "_1080p" in c]
            return prioritized[0] if prioritized else candidates[0]
    else:
        candidates = re.findall(r'https://cdn\.pixabay\.com/photo/[^"\'\s<>]+\.(?:jpg|jpeg|png)', html)
        if candidates:
            large_candidates = [c for c in candidates if "_1280." in c or "_960." in c]
            return large_candidates[0] if large_candidates else candidates[0]
    return None


def execute_search_request(search_url: str, media_type: str = "video") -> tuple[str, str | None]:
    """
    Attempts search request. If blocked by 403, cycles through free proxies automatically.
    """
    configured_proxy = get_configured_proxy()
    candidate_proxies = [configured_proxy] if configured_proxy else [None]

    # Try direct / configured proxy first
    for proxy in candidate_proxies:
        try:
            session = create_session(proxy)
            headers = dict(BROWSER_HEADERS)
            headers["Referer"] = "https://pixabay.com/"
            resp = session.get(search_url, headers=headers, timeout=15)
            if resp.status_code == 200:
                url = extract_media_url_from_html(resp.text, media_type)
                if url:
                    return url, proxy
        except Exception:
            pass

    # Fallback: Cycle through live elite public proxies
    print("[Pixabay] Cloud IP blocked (403). Fetching free proxy pool...")
    free_proxies = fetch_free_proxy_list()

    for idx, test_proxy in enumerate(free_proxies[:15], start=1):
        try:
            print(f"[Pixabay] Trying free proxy ({idx}/{min(len(free_proxies), 15)}): {test_proxy}")
            session = create_session(test_proxy)
            headers = dict(BROWSER_HEADERS)
            headers["Referer"] = "https://pixabay.com/"
            resp = session.get(search_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                url = extract_media_url_from_html(resp.text, media_type)
                if url:
                    print(f"[Pixabay] Successfully bypassed Cloudflare using {test_proxy}!")
                    return url, test_proxy
        except Exception:
            continue

    raise RuntimeError(
        "Pixabay Cloudflare challenge blocked all attempts. "
        "Get a free proxy from webshare.io (free 10 proxies) and set PIXABAY_PROXY in secrets."
    )


def fetch_landscape_video(query: str, output_path: str) -> str:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Video search query cannot be empty.")

    encoded_query = urllib.parse.quote(clean_query)
    search_url = f"https://pixabay.com/videos/search/{encoded_query}/?order=ec"

    video_url, working_proxy = execute_search_request(search_url, media_type="video")
    return download_stream(video_url, output_path, proxy=working_proxy)


def fetch_image(query: str, output_path: str) -> str:
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Image search query cannot be empty.")

    encoded_query = urllib.parse.quote(clean_query)
    search_url = f"https://pixabay.com/photos/search/{encoded_query}/?order=ec"

    image_url, working_proxy = execute_search_request(search_url, media_type="image")
    return download_stream(image_url, output_path, proxy=working_proxy)
