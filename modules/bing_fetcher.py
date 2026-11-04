import asyncio
import os
import sys
import time
import types
from curl_cffi.requests import AsyncSession

if "rookiepy" not in sys.modules:
    sys.modules["rookiepy"] = types.ModuleType("rookiepy")

from bingart import BingArt, Model, Aspect, AuthCookieError, PromptRejectedError

BASE_PROJECT_DIR = os.getcwd()
DEFAULT_BING_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "storage", "temp", "bing_mai1")

# Global pointer for round-robin rotation across generations
_COOKIE_ROTATION_INDEX = 0


def parse_cookie_pool(cookie_input) -> list[str]:
    """Extracts a list of clean _U cookies from string, list, or secrets."""
    if not cookie_input:
        return []
    if isinstance(cookie_input, (list, tuple)):
        return [str(c).strip().strip('"').strip("'") for c in cookie_input if str(c).strip()]
    if isinstance(cookie_input, str):
        raw_list = cookie_input.replace("\n", ",").split(",")
        return [c.strip().strip('"').strip("'") for c in raw_list if c.strip() and c.strip() != "your_cookie_here"]
    return []


async def download_image(session: AsyncSession, url: str, filepath: str) -> None:
    """Downloads full-resolution image bytes using Chrome impersonation."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    for attempt in range(3):
        try:
            response = await session.get(url, timeout=20)
            if response.status_code == 200 and len(response.content) > 1000:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"[BingFetcher] Image saved successfully: {filepath}")
                return
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2)
    raise RuntimeError(f"Failed downloading image bytes from {url}")


async def generate_bing_image(
    prompt_text: str,
    output_path: str,
    cookie_u: str | list[str],
    download_all_variations: bool = False
) -> str:
    """
    Generates image using Bing Image Creator (Model.MAI1 Landscape).
    Rotates cookies round-robin for each scene generation.
    """
    global _COOKIE_ROTATION_INDEX

    clean_prompt = prompt_text.strip()
    if not clean_prompt:
        raise ValueError("Error: Prompt cannot be empty.")

    cookies = parse_cookie_pool(cookie_u)
    if not cookies:
        raise ValueError("Error: At least one Microsoft _U cookie is required.")

    target_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(target_dir, exist_ok=True)

    # Reorder cookies starting from the current rotation index
    start_idx = _COOKIE_ROTATION_INDEX % len(cookies)
    ordered_cookies = cookies[start_idx:] + cookies[:start_idx]
    _COOKIE_ROTATION_INDEX = (_COOKIE_ROTATION_INDEX + 1) % len(cookies)

    last_error = None
    for attempt_cycle in range(2):  # Try the pool up to 2 full cycles
        for idx, current_cookie in enumerate(ordered_cookies, start=1):
            print(f"[BingFetcher] Attempt with Cookie #{idx}/{len(ordered_cookies)} (Cycle {attempt_cycle + 1})...")
            try:
                async with BingArt(auth_cookie_U=current_cookie) as bing:
                    result = await bing.generate(
                        clean_prompt,
                        model=Model.MAI1,
                        aspect=Aspect.LANDSCAPE
                    )

                    image_list = result.get("images", [])
                    if not image_list:
                        print(f"[BingFetcher] Cookie #{idx} returned 0 images. Trying next...")
                        continue

                    async with AsyncSession(impersonate="chrome124") as dl_session:
                        primary_clean_url = image_list[0]["url"].split("?")[0]
                        await download_image(dl_session, primary_clean_url, output_path)

                        if download_all_variations and len(image_list) > 1:
                            tasks = []
                            for var_idx, img in enumerate(image_list[1:], start=2):
                                c_url = img["url"].split("?")[0]
                                alt_path = os.path.join(target_dir, f"mai1_var{var_idx}.jpg")
                                tasks.append(download_image(dl_session, c_url, alt_path))
                            await asyncio.gather(*tasks, return_exceptions=True)

                    return output_path

            except AuthCookieError as ace:
                print(f"[BingFetcher] Cookie #{idx} invalid or expired: {ace}")
                last_error = ace
                continue
            except PromptRejectedError:
                raise ValueError(f"Prompt violated Bing safety filters: \"{clean_prompt}\"")
            except Exception as e:
                print(f"[BingFetcher] Error on cookie #{idx}: {e}")
                last_error = e
                continue

        if attempt_cycle == 0:
            print("[BingFetcher] All cookies throttled. Cooling down for 12 seconds before retry...")
            await asyncio.sleep(12)

    raise RuntimeError(f"All Bing cookies exhausted. Last error: {last_error}")


def run_sync_generate(
    prompt_text: str,
    output_path: str,
    cookie_u: str | list[str],
    download_all_variations: bool = False
) -> str:
    """Synchronous execution helper with active loop protection."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                generate_bing_image(prompt_text, output_path, cookie_u, download_all_variations)
            ).result()
    else:
        return asyncio.run(
            generate_bing_image(prompt_text, output_path, cookie_u, download_all_variations)
        )
