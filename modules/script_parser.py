import asyncio
import concurrent.futures
import glob
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import time

import cv2
import numpy as np

# Headless Matplotlib configuration for formula rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.cartesia_tts import generate_speech
from modules.pixabay_fetcher import fetch_landscape_video, fetch_image
from modules.bing_fetcher import run_sync_generate
from animations.animation_router import render_animation_by_index
from modules.ffmpeg_compositor import (
    render_individual_scene,
    concat_all_scenes,
    get_stream_duration
)

BG_DIR = "assets/backgrounds"
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "storage/checkpoints"

os.makedirs(BG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logger = logging.getLogger("VideoEngine.Pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

ABSTRACT_AI_KEYWORDS = {
    "quantum", "subatomic", "particle", "ghost", "neutrino", "matrix", "cyber",
    "futuristic", "citadel", "portal", "dimension", "black hole", "singularity",
    "supernova", "cosmic energy", "abstract", "alien", "energy silhouette",
    "glowing energy", "laser burst", "cosmic rays", "accretion disk"
}


def compute_scene_hash(scene: dict) -> str:
    """Produces a deterministic 12-character hash for cache verification."""
    raw = f"{scene.get('sentence', '')}_{scene.get('prompt', '')}_{scene.get('query', '')}_{scene.get('equation', '')}_{json.dumps(scene.get('headline', ''))}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]


def is_valid_media_file(file_path: str, min_size_bytes: int = 15000) -> bool:
    """Checks if a file exists, is non-empty, and has valid media content."""
    return os.path.exists(file_path) and os.path.getsize(file_path) >= min_size_bytes


def sanitize_mathtext(expr: str) -> str:
    """Replaces unsupported LaTeX macros with valid mathtext equivalents."""
    cleaned = expr.strip()
    # Replace \stackrel{?}{=} with \approx
    cleaned = re.sub(r'\\stackrel\{[^}]*\}\{=\}', r'\\approx', cleaned)
    # Replace \implies with \Rightarrow
    cleaned = cleaned.replace(r'\implies', r'\Rightarrow')
    # Replace \iff with \Leftrightarrow
    cleaned = cleaned.replace(r'\iff', r'\Leftrightarrow')
    return cleaned


def render_cinematic_equation_card(
    latex_equation: str,
    output_path: str,
    formula_name: str = "",
    width: int = 1760,
    height: int = 870
) -> str:
    """
    Renders an exact LaTeX formula onto a dark obsidian cinematic card.
    Guaranteed zero-crash with multi-stage fallback.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    clean_expr = sanitize_mathtext(latex_equation)
    if not clean_expr.startswith("$"):
        clean_expr = f"${clean_expr}$"

    # Multi-stage rendering with bulletproof try/except
    fig = None
    rendered_successfully = False

    for attempt in range(2):
        try:
            fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="none")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.patch.set_alpha(0.0)

            # Stage 1: Try rendering math formula
            current_text = clean_expr if attempt == 0 else clean_expr.replace('\\', '')
            is_math = attempt == 0

            ax.text(
                0.5, 0.50, current_text,
                fontsize=46 if is_math else 34,
                color="#FFFFFF",
                ha="center", va="center",
                usetex=False
            )

            if formula_name:
                ax.text(
                    0.5, 0.18, formula_name.upper(),
                    fontsize=22,
                    color="#DC2644",
                    fontweight="bold",
                    ha="center", va="center"
                )

            fig.canvas.draw()
            rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
            plt.close(fig)
            rendered_successfully = True
            break
        except Exception as err:
            logger.warning(f"[EquationRenderer] Mathtext warning on attempt {attempt + 1}: {err}")
            if fig is not None:
                plt.close(fig)

    # Stage 3 fallback: If matplotlib failed completely, draw clean fallback image
    if not rendered_successfully:
        logger.warning(f"[EquationRenderer] Fallback to clean title card for: {formula_name}")
        fallback_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(fallback_img, formula_name or "MATHEMATICAL FORMULA", (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(output_path, fallback_img)
        return output_path

    formula_bgr = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
    formula_alpha = rgba_buffer[:, :, 3] / 255.0

    # Build dark obsidian background with radial gradient
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width * 0.5, height * 0.5
    r = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    gradient = np.clip(1.0 - 0.50 * r, 0.0, 1.0)[:, :, np.newaxis]

    dark_tone = np.array([26, 22, 18], dtype=np.float32)
    ambient_tone = np.array([10, 8, 8], dtype=np.float32)
    card_surface = (ambient_tone + (dark_tone - ambient_tone) * gradient).astype(np.uint8)

    # Gridlines for scientific aesthetic
    card_surface[::60, :, :] = np.clip(card_surface[::60, :, :].astype(np.int16) + 12, 0, 255).astype(np.uint8)
    card_surface[:, ::60, :] = np.clip(card_surface[:, ::60, :].astype(np.int16) + 12, 0, 255).astype(np.uint8)

    # Composite formula with subtle glow
    alpha_3ch = formula_alpha[:, :, np.newaxis]
    final_card = (card_surface * (1.0 - alpha_3ch) + formula_bgr * alpha_3ch).astype(np.uint8)

    glow_mask = (formula_alpha * 255).astype(np.uint8)
    glow_blurred = cv2.GaussianBlur(glow_mask, (31, 31), 0)
    glow_layer = (glow_blurred[:, :, np.newaxis] / 255.0) * np.array([120, 40, 20], dtype=np.float32)
    final_card = np.clip(final_card.astype(np.float32) + glow_layer, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, final_card)
    logger.info(f"[EquationRenderer] Rendered exact math card: {output_path}")
    return output_path


def _execute_tts_with_retry(sentence: str, target_audio_path: str, voice_name: str = None) -> tuple[str, float]:
    """Retries TTS voiceover generation up to 3 times on transient drops."""
    kwargs = {}
    if voice_name:
        kwargs["voice_name"] = voice_name

    for attempt in range(3):
        try:
            res = generate_speech(sentence, target_audio_path, **kwargs)

            if inspect.iscoroutine(res):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            res = pool.submit(asyncio.run, res).result()
                    else:
                        res = loop.run_until_complete(res)
                except RuntimeError:
                    res = asyncio.run(res)

            if isinstance(res, (tuple, list)) and len(res) == 2:
                return str(res[0]), float(res[1])
            elif isinstance(res, str) and is_valid_media_file(res, 5000):
                return res, get_stream_duration(res)
            elif is_valid_media_file(target_audio_path, 5000):
                return target_audio_path, get_stream_duration(target_audio_path)

        except Exception as e:
            logger.warning(f"[TTS] Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    raise RuntimeError(f"TTS failed completely for sentence: '{sentence[:40]}...'")


def generate_emergency_fallback_image(output_path: str):
    """Generates an emergency gradient card so pipeline never crashes on missing files."""
    h, w = 1080, 1920
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        ratio = y / h
        canvas[y, :] = [int(35 * (1 - ratio)), int(20 * (1 - ratio)), int(45 * ratio + 15)]
    cv2.circle(canvas, (w // 2, h // 2), 220, (180, 80, 20), -1)
    canvas = cv2.GaussianBlur(canvas, (101, 101), 0)
    cv2.imwrite(output_path, canvas)


def acquire_scene_image(scene: dict, raw_img_path: str, bing_cookies) -> str:
    """Acquires image using intelligent routing with complete two-way fallback."""
    if "equation" in scene and str(scene.get("equation", "")).strip():
        logger.info(f"[ImageAcquisition] Equation detected. Rendering local math card...")
        formula_latex = str(scene["equation"]).strip()
        formula_name = str(scene.get("formula_name") or scene.get("title") or "").strip()
        render_cinematic_equation_card(formula_latex, raw_img_path, formula_name=formula_name)
        return raw_img_path

    explicit = str(scene.get("image_source") or scene.get("source") or "").lower().strip()
    prompt_text = scene.get("prompt", "").strip()
    query_text = scene.get("query", "").strip()
    search_q = query_text or prompt_text

    combined_text = f"{prompt_text.lower()} {query_text.lower()}"
    prefer_ai = explicit in ["bing", "ai", "dall-e"] or any(k in combined_text for k in ABSTRACT_AI_KEYWORDS)

    if prefer_ai and prompt_text:
        try:
            logger.info(f"[ImageAcquisition] Trying Bing AI for: '{prompt_text}'")
            run_sync_generate(prompt_text, raw_img_path, cookie_u=bing_cookies)
            if is_valid_media_file(raw_img_path):
                return raw_img_path
        except Exception as e:
            logger.warning(f"[ImageAcquisition] Bing AI failed ({e}). Falling back to Wikimedia...")

        try:
            fetch_image(search_q, raw_img_path)
            if is_valid_media_file(raw_img_path):
                return raw_img_path
        except Exception as e2:
            logger.warning(f"[ImageAcquisition] Wikimedia fallback failed ({e2}). Using emergency card.")
    else:
        try:
            logger.info(f"[ImageAcquisition] Trying Wikimedia Commons for: '{search_q}'")
            fetch_image(search_q, raw_img_path)
            if is_valid_media_file(raw_img_path):
                return raw_img_path
        except Exception as e:
            logger.warning(f"[ImageAcquisition] Wikimedia failed ({e}). Falling back to Bing AI...")

        if prompt_text:
            try:
                run_sync_generate(prompt_text, raw_img_path, cookie_u=bing_cookies)
                if is_valid_media_file(raw_img_path):
                    return raw_img_path
            except Exception as e2:
                logger.warning(f"[ImageAcquisition] Bing AI fallback failed ({e2}). Using emergency card.")

    generate_emergency_fallback_image(raw_img_path)
    return raw_img_path


def get_available_backgrounds(assets_dir: str = BG_DIR) -> list[str]:
    patterns = [os.path.join(assets_dir, "*.mp4"), os.path.join(assets_dir, "*.MP4")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def process_video_pipeline(
    script_data: list[dict],
    output_video_path: str = None,
    temp_dir: str = "temp_pipeline",
    bing_cookie=None,
    bing_cookie_u=None,
    voice_name: str = None,
    rest_seconds_between_scenes: int = 45,
    progress_callback=None,
    artifact_callback=None
) -> str:
    start_time = time.time()
    effective_cookie = bing_cookie or bing_cookie_u

    if not output_video_path:
        output_video_path = os.path.join(OUTPUT_DIR, f"master_video_{int(start_time)}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    available_bgs = get_available_backgrounds(BG_DIR)
    if not available_bgs:
        raise RuntimeError(f"No background videos found in '{BG_DIR}/'.")

    scene_video_paths = []
    bg_cycle_idx = 0
    anim_cycle_idx = 0
    total_scenes = len(script_data)

    logger.info(f"Pipeline started for {total_scenes} scenes.")

    for idx, scene in enumerate(script_data):
        scene_num = idx + 1
        scene_prefix = f"scene_{scene_num:02d}"
        s_hash = compute_scene_hash(scene)

        cached_scene_mp4 = os.path.join(CHECKPOINT_DIR, f"{scene_prefix}_{s_hash}_rendered.mp4")

        # Instant Cache Skip
        if is_valid_media_file(cached_scene_mp4, 50000):
            logger.info(f"[{scene_prefix}] Found cached completed render. Reusing...")
            if progress_callback:
                progress_callback(f"Scene {scene_num}/{total_scenes}: Reusing completed cache...")
            if artifact_callback:
                artifact_callback("video", cached_scene_mp4, f"🎬 Scene {scene_num} (Cached)")
            scene_video_paths.append(cached_scene_mp4)
            continue

        sentence = scene.get("sentence", "").strip()
        query_text = scene.get("query", "").strip()
        prompt_text = scene.get("prompt", "").strip()
        headline_data = scene.get("headline") or scene.get("heading") or scene.get("title", None)

        if not sentence:
            continue

        # 1. Voiceover
        logger.info(f"[{scene_prefix}] Generating Voiceover...")
        if progress_callback:
            progress_callback(f"Scene {scene_num}/{total_scenes}: Generating Voiceover...")

        audio_target = os.path.join(CHECKPOINT_DIR, f"{scene_prefix}_{s_hash}_tts.wav")
        if not is_valid_media_file(audio_target, 5000):
            audio_path, duration = _execute_tts_with_retry(sentence, audio_target, voice_name=voice_name)
        else:
            audio_path, duration = audio_target, get_stream_duration(audio_target)

        if artifact_callback:
            artifact_callback("audio", audio_path, f"🎤 Scene {scene_num} Audio ({duration:.1f}s)")

        scene_mode = idx % 3
        rendered_scene_mp4 = cached_scene_mp4

        # MODE 0: LANDSCAPE B-ROLL VIDEO
        if scene_mode == 0:
            logger.info(f"[{scene_prefix}] Rendering Landscape Video")
            if progress_callback:
                progress_callback(f"Scene {scene_num}/{total_scenes}: Fetching Video Footage...")

            video_search_query = query_text or prompt_text or "space"
            raw_video_path = os.path.join(CHECKPOINT_DIR, f"{scene_prefix}_{s_hash}_raw_video.mp4")

            if not is_valid_media_file(raw_video_path, 20000):
                try:
                    fetch_landscape_video(video_search_query, raw_video_path)
                except Exception as e:
                    logger.warning(f"[{scene_prefix}] Stock video fetch failed ({e}). Using local background fallback...")
                    fallback_bg = available_bgs[bg_cycle_idx % len(available_bgs)]
                    shutil.copyfile(fallback_bg, raw_video_path)

            render_individual_scene(
                scene_type="video",
                audio_path=audio_path,
                output_path=rendered_scene_mp4,
                video_path=raw_video_path
            )

        # MODE 1: BACKGROUND LOOP + STAMP CARD
        elif scene_mode == 1:
            logger.info(f"[{scene_prefix}] Rendering Type 2 Stamp Drop")
            if progress_callback:
                progress_callback(f"Scene {scene_num}/{total_scenes}: Acquiring Image & Rendering Stamp...")

            raw_img_path = os.path.join(CHECKPOINT_DIR, f"{scene_prefix}_{s_hash}_image.jpg")
            if not is_valid_media_file(raw_img_path, 5000):
                acquire_scene_image(scene, raw_img_path, effective_cookie)

            if artifact_callback and is_valid_media_file(raw_img_path):
                artifact_callback("image", raw_img_path, f"🖼️ Scene {scene_num} Card")

            bg_loop_video = available_bgs[bg_cycle_idx % len(available_bgs)]
            bg_cycle_idx += 1

            render_individual_scene(
                scene_type="background_with_image",
                audio_path=audio_path,
                output_path=rendered_scene_mp4,
                background_video_path=bg_loop_video,
                foreground_image_path=raw_img_path,
                headline_data=headline_data
            )

        # MODE 2: KINETIC ANIMATION
        elif scene_mode == 2:
            logger.info(f"[{scene_prefix}] Rendering Type 3 Kinetic Animation")
            if progress_callback:
                progress_callback(f"Scene {scene_num}/{total_scenes}: Generating Visual & Animation...")

            raw_img_path = os.path.join(CHECKPOINT_DIR, f"{scene_prefix}_{s_hash}_ai_image.jpg")
            if not is_valid_media_file(raw_img_path, 5000):
                if "equation" in scene and str(scene.get("equation", "")).strip():
                    formula_latex = str(scene["equation"]).strip()
                    formula_name = str(scene.get("formula_name") or scene.get("title") or "").strip()
                    render_cinematic_equation_card(formula_latex, raw_img_path, formula_name=formula_name)
                else:
                    acquired = False
                    if prompt_text:
                        try:
                            run_sync_generate(prompt_text, raw_img_path, cookie_u=effective_cookie)
                            acquired = is_valid_media_file(raw_img_path)
                        except Exception as e:
                            logger.warning(f"Bing AI failed ({e}). Trying fallback...")

                    if not acquired:
                        try:
                            fetch_image(query_text or prompt_text, raw_img_path)
                        except Exception:
                            generate_emergency_fallback_image(raw_img_path)

            if artifact_callback and is_valid_media_file(raw_img_path):
                artifact_callback("image", raw_img_path, f"🎨 Scene {scene_num} Visual")

            anim_video_path = os.path.join(CHECKPOINT_DIR, f"{scene_prefix}_{s_hash}_animation.mp4")
            render_animation_by_index(
                image_path=raw_img_path,
                output_path=anim_video_path,
                duration_sec=duration,
                cycle_index=anim_cycle_idx,
                fps=30
            )
            anim_cycle_idx += 1

            render_individual_scene(
                scene_type="animation",
                audio_path=audio_path,
                output_path=rendered_scene_mp4,
                animation_video_path=anim_video_path
            )

        if artifact_callback and is_valid_media_file(rendered_scene_mp4):
            artifact_callback("video", rendered_scene_mp4, f"🎬 Scene {scene_num} Rendered MP4")

        scene_video_paths.append(rendered_scene_mp4)

        # Rest Cooldown Heartbeat between scenes
        if scene_num < total_scenes and rest_seconds_between_scenes > 0:
            logger.info(f"Resting for {rest_seconds_between_scenes}s to cool down CPU and Bing cookie...")
            for remaining in range(rest_seconds_between_scenes, 0, -5):
                if progress_callback:
                    progress_callback(f"⏳ Cooldown: Resting CPU & Cookies for {remaining}s...")
                time.sleep(min(5, remaining))

    logger.info(f"Concatenating {len(scene_video_paths)} scenes...")
    if progress_callback:
        progress_callback("Concatenating all scenes into final video...")

    concat_all_scenes(scene_video_paths, output_video_path)

    if artifact_callback and is_valid_media_file(output_video_path):
        artifact_callback("final_video", output_video_path, "🏆 Final Full Combined Video")

    logger.info(f"Pipeline finished in {time.time() - start_time:.2f}s -> {output_video_path}")
    return output_video_path
