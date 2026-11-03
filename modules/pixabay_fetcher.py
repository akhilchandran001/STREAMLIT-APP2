import json
import os
import re
import urllib.parse
from curl_cffi import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://pixabay.com/",
}


def download_stream(url: str, output_path: str) -> str:
    """Streams binary content to disk using curl_cffi with Chrome impersonation."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    session = requests.Session(impersonate="chrome124")
    response = session.get(url, headers=HEADERS, stream=True, timeout=30)
    
    if response.status_code != 200:
        raise RuntimeError(f"Download failed for {url} with status {response.status_code}")
        
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                
    if os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Downloaded file '{output_path}' is empty.")
        
    print(f"Downloaded asset: {output_path} ({os.path.getsize(output_path)} bytes)")
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
    Scrapes Pixabay for landscape video footage matching the query using curl_cffi.
    No API key required.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Video search query cannot be empty.")

    encoded_query = urllib.parse.quote(clean_query)
    search_url = f"https://pixabay.com/videos/search/{encoded_query}/?order=ec"
    
    session = requests.Session(impersonate="chrome124")
    response = session.get(search_url, headers=HEADERS, timeout=20)
    
    if response.status_code != 200:
        raise RuntimeError(f"Pixabay video search blocked or failed with HTTP {response.status_code}")
        
    html = response.text
    video_url = None

    # Extraction Strategy 1: Parse __NEXT_DATA__ JSON state tree
    next_data = extract_next_data(html)
    if next_data:
        try:
            def search_media_nodes(data):
                if isinstance(data, dict):
                    # Check for video model signature
                    if "videos" in data and isinstance(data["videos"], dict):
                        vids = data["videos"]
                        # Prefer large or medium landscape stream
                        for quality in ["large", "medium", "small"]:
                            cand = vids.get(quality, {})
                            cand_url = cand.get("url")
                            w = cand.get("width", 0)
                            h = cand.get("height", 0)
                            if cand_url and (w >= h):  # Strictly landscape or square
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
            print(f"Notice: Failed parsing __NEXT_DATA__ node: {e}")

    # Extraction Strategy 2: Direct CDN regular expression parsing
    if not video_url:
        candidates = re.findall(r'https://cdn\.pixabay\.com/video/[^"\'\s<>]+\.mp4', html)
        if candidates:
            # Prioritize high/medium resolution versions over tiny mobile clips
            prioritized = [c for c in candidates if "_medium" in c or "_large" in c or "_1080p" in c]
            video_url = prioritized[0] if prioritized else candidates[0]

    if not video_url:
        raise RuntimeError(f"No Pixabay landscape video found for query: '{query}'")

    print(f"Extracted Pixabay Video CDN: {video_url}")
    return download_stream(video_url, output_path)


def fetch_image(query: str, output_path: str) -> str:
    """
    Scrapes Pixabay for landscape stock images/photos matching the query using curl_cffi.
    No API key required.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Image search query cannot be empty.")

    encoded_query = urllib.parse.quote(clean_query)
    search_url = f"https://pixabay.com/photos/search/{encoded_query}/?order=ec"
    
    session = requests.Session(impersonate="chrome124")
    response = session.get(search_url, headers=HEADERS, timeout=20)
    
    if response.status_code != 200:
        raise RuntimeError(f"Pixabay photo search blocked or failed with HTTP {response.status_code}")
        
    html = response.text
    image_url = None

    # Extraction Strategy 1: Parse __NEXT_DATA__ JSON state tree
    next_data = extract_next_data(html)
    if next_data:
        try:
            def search_image_nodes(data):
                if isinstance(data, dict):
                    # Check for image item structure
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
            print(f"Notice: Failed parsing image from __NEXT_DATA__: {e}")

    # Extraction Strategy 2: Direct CDN regular expression parsing
    if not image_url:
        candidates = re.findall(r'https://cdn\.pixabay\.com/photo/[^"\'\s<>]+\.(?:jpg|jpeg|png)', html)
        if candidates:
            # Filter out tiny thumbnails (_60, _150, _340) in favor of large assets (_960, _1280)
            large_candidates = [c for c in candidates if "_1280." in c or "_960." in c]
            image_url = large_candidates[0] if large_candidates else candidates[0]

    if not image_url:
        raise RuntimeError(f"No Pixabay image found for query: '{query}'")

    print(f"Extracted Pixabay Image CDN: {image_url}")
    return download_stream(image_url, output_path)
