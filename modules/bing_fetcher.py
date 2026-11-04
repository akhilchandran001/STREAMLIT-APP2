import asyncio
import os
import sys
import time
import types
from curl_cffi.requests import AsyncSession

# Mock rookiepy to bypass compilation issues in headless Linux / Streamlit Cloud containers
if "rookiepy" not in sys.modules:
    sys.modules["rookiepy"] = types.ModuleType("rookiepy")

from bingart import BingArt, Model, Aspect, AuthCookieError, PromptRejectedError

# Default storage path adapted for Streamlit Cloud ephemeral runtime
BASE_PROJECT_DIR = os.getcwd()
DEFAULT_BING_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "storage", "temp", "bing_mai1")


def parse_cookie_pool(cookie_input) -> list[str]:
    """Extracts a list of clean _U cookies from str, list, or comma/newline-separated strings."""
    if not cookie_input:
        return []
    if isinstance(cookie_input, (list, tuple)):
        return [str(c).strip().strip('"').strip("'") for c in cookie_input if str(c).strip()]
    if isinstance(cookie_input, str):
        raw_list = cookie_input.replace("\n", ",").split(",")
        return [c.strip().strip('"').strip("'") for c in raw_list if c.strip() and c.strip() != "your_cookie_here"]
    return []


async def download_image(session: AsyncSession, url: str, filepath: str) -> None:
    """Downloads full-resolution image bytes using curl_cffi with Chrome impersonation."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    response = await session.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Saved Bing Image: {filepath}")
    else:
        raise RuntimeError(f"Failed downloading {url} (HTTP {response.status_code})")


async def generate_bing_image(
    prompt_text: str,
    output_path: str,
    cookie_u: str | list[str],
    download_all_variations: bool = False
) -> str:
    """
    Submits prompt to Bing Image Creator with Model.MAI1 in Landscape (7:4 wide),
    strips '?pid=...' URL parameters to acquire raw full-resolution image files,
    and saves to the target filepath inside the Streamlit Cloud filesystem.
    Rotates through multiple _U cookies if one fails, expires, or gets rate-limited.
    """
    clean_prompt = prompt_text.strip()
    if not clean_prompt:
        raise ValueError("Error: Prompt cannot be empty.")

    cookies = parse_cookie_pool(cookie_u)
    if not cookies:
        raise ValueError("Error: A valid Microsoft _U cookie string or list is required.")

    target_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(target_dir, exist_ok=True)

    last_error = None
    for idx, current_cookie in enumerate(cookies, start=1):
        print(f"\n[BingArt] Attempting generation with Cookie #{idx}/{len(cookies)} (Model.MAI1 Landscape)...")
        try:
            async with BingArt(auth_cookie_U=current_cookie) as bing:
                print(f"Generating landscape image with Model.MAI1 for:\n\"{clean_prompt}\"")

                # Positional prompt with MAI1 and Landscape (7:4 widescreen)
                result = await bing.generate(
                    clean_prompt,
                    model=Model.MAI1,
                    aspect=Aspect.LANDSCAPE
                )

                image_list = result.get("images", [])
                if not image_list:
                    print(f"[BingArt] Cookie #{idx} returned 0 images. Trying next cookie...")
                    continue

                print(f"Found {len(image_list)} images from Bing generation.")
                timestamp = int(time.time())

                async with AsyncSession(impersonate="chrome124") as dl_session:
                    # Save primary selected image to exact requested output path
                    primary_clean_url = image_list[0]["url"].split("?")[0]
                    await download_image(dl_session, primary_clean_url, output_path)

                    # Optionally save remaining variations to directory for cache/fallback
                    if download_all_variations and len(image_list) > 1:
                        tasks = []
                        for var_idx, img in enumerate(image_list[1:], start=2):
                            clean_url = img["url"].split("?")[0]
                            alt_path = os.path.join(target_dir, f"mai1_{timestamp}_var{var_idx}.jpg")
                            tasks.append(download_image(dl_session, clean_url, alt_path))
                        await asyncio.gather(*tasks)

                print(f"Bing asset generation completed successfully: {output_path}")
                return output_path

        except AuthCookieError as ace:
            print(f"[BingArt] Cookie #{idx} expired or session invalidated: {ace}")
            last_error = ace
            continue
        except PromptRejectedError:
            raise ValueError(f"Prompt violated Bing content moderation filters: \"{clean_prompt}\"")
        except Exception as e:
            print(f"[BingArt] Cookie #{idx} failed: {e}. Trying next...")
            last_error = e
            continue

    raise RuntimeError(f"All ({len(cookies)}) Bing _U cookies failed. Last error: {last_error}")


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


if __name__ == "__main__":
    test_cookie = os.getenv("BING_COOKIE_U", "")
    test_prompt = "Futuristic cybernetic city floating in the sky, hyper-detailed digital art"
    test_dest = os.path.join(DEFAULT_BING_CACHE_DIR, "test_landscape.jpg")

    if not test_cookie:
        print("Please export BING_COOKIE_U environment variable to test directly.")
        sys.exit(1)

    result_file = run_sync_generate(test_prompt, test_dest, test_cookie)
    print(f"Direct test completed: {result_file}")
