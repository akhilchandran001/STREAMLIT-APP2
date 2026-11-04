import asyncio
import concurrent.futures
import glob
import inspect
import json
import logging
import os
import shutil
import time

import cv2
import numpy as np

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

os.makedirs(BG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def _execute_tts(sentence: str, target_audio_path: str) -> tuple[str, float]:
    """Safely executes generate_speech whether it is an async coroutine or sync function."""
    res = generate_speech(sentence, target_audio_path)

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
    elif isinstance(res, str) and os.path.exists(res):
        return res, get_stream_duration(res)
    else:
        return target_audio_path, get_stream_duration(target_audio_path)


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
            if os.path.exists(raw_img_path) and os.path.getsize(raw_img_path) > 1000:
                return raw_img_path
        except Exception as e:
            logger.warning(f"[ImageAcquisition] Bing AI failed ({e}). Falling back to Wikimedia...")

        try:
            fetch_image(search_q, raw_img_path)
            if os.path.exists(raw_img_path) and os.path.getsize(raw_img_path) > 1000:
                return raw_img_path
        except Exception as e2:
            logger.warning(f"[ImageAcquisition] Wikimedia fallback failed ({e2}). Using emergency card.")
    else:
        try:
            logger.info(f"[ImageAcquisition] Trying Wikimedia Commons for: '{search_q}'")
            fetch_image(search_q, raw_img_path)
            if os.path.exists(raw_img_path) and os.path.getsize(raw_img_path) > 1000:
                return raw_img_path
        except Exception as e:
            logger.warning(f"[ImageAcquisition] Wikimedia failed ({e}). Falling back to Bing AI...")

        if prompt_text:
            try:
                run_sync_generate(prompt_text, raw_img_path, cookie_u=bing_cookies)
                if os.path.exists(raw_img_path) and os.path.getsize(raw_img_path) > 1000:
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
    output_video_path: str,
    temp_dir: str = "temp_pipeline",
    bing_cookie=None,
    progress_callback=None,
    artifact_callback=None
) -> str:
    start_time = time.time()
    run_temp_dir = os.path.join(temp_dir, f"run_{int(start_time)}")
    os.makedirs(run_temp_dir, exist_ok=True)

    available_bgs = get_available_backgrounds(BG_DIR)
    if not available_bgs:
        raise RuntimeError(f"No background videos found in '{BG_DIR}/'.")

    scene_video_paths = []
    bg_cycle_idx = 0
    anim_cycle_idx = 0
    total_scenes = len(script_data)

    logger.info(f"Pipeline started for {total_scenes} scenes.")

    try:
        for idx, scene in enumerate(script_data):
            scene_num = idx + 1
            scene_prefix = f"scene_{scene_num:02d}"

            sentence = scene.get("sentence", "").strip()
            query_text = scene.get("query", "").strip()
            prompt_text = scene.get("prompt", "").strip()
            headline_data = scene.get("headline", None)

            if not sentence:
                continue

            # 1. Voiceover
            logger.info(f"[{scene_prefix}] Generating Voiceover...")
            if progress_callback:
                progress_callback(f"Scene {scene_num}/{total_scenes}: Generating Voiceover...")

            audio_target = os.path.join(run_temp_dir, f"{scene_prefix}_tts.wav")
            audio_path, duration = _execute_tts(sentence, audio_target)

            if artifact_callback:
                artifact_callback("audio", audio_path, f"🎤 Scene {scene_num} Audio ({duration:.1f}s)")

            scene_mode = idx % 3
            rendered_scene_mp4 = os.path.join(run_temp_dir, f"{scene_prefix}_rendered.mp4")

            # MODE 0: LANDSCAPE B-ROLL VIDEO
            if scene_mode == 0:
                logger.info(f"[{scene_prefix}] Rendering Landscape Video")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Fetching Video Footage...")

                video_search_query = query_text or prompt_text or "cinematic nature space"
                raw_video_path = os.path.join(run_temp_dir, f"{scene_prefix}_raw_video.mp4")
                fetch_landscape_video(video_search_query, raw_video_path)

                render_individual_scene(
                    scene_type="video",
                    audio_path=audio_path,
                    output_path=rendered_scene_mp4,
                    video_path=raw_video_path
                )

            # MODE 1: BACKGROUND LOOP + STAMP IMAGE + HEADLINE
            elif scene_mode == 1:
                logger.info(f"[{scene_prefix}] Rendering Type 2 Stamp Drop")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Acquiring Image & Rendering Stamp...")

                raw_img_path = os.path.join(run_temp_dir, f"{scene_prefix}_image.jpg")
                acquire_scene_image(scene, raw_img_path, bing_cookie)

                if artifact_callback and os.path.exists(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🖼️ Scene {scene_num} Image")

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

            # MODE 2: KINETIC ANIMATION (STRICT AI + FALLBACK)
            elif scene_mode == 2:
                logger.info(f"[{scene_prefix}] Rendering Type 3 Kinetic Animation")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Generating AI Art & Animation...")

                raw_img_path = os.path.join(run_temp_dir, f"{scene_prefix}_ai_image.jpg")
                acquired = False
                if prompt_text:
                    try:
                        run_sync_generate(prompt_text, raw_img_path, cookie_u=bing_cookie)
                        acquired = os.path.exists(raw_img_path) and os.path.getsize(raw_img_path) > 1000
                    except Exception as e:
                        logger.warning(f"Bing AI failed for Type 3 ({e}). Trying Wikimedia fallback...")

                if not acquired:
                    try:
                        fetch_image(query_text or prompt_text, raw_img_path)
                    except Exception:
                        generate_emergency_fallback_image(raw_img_path)

                if artifact_callback and os.path.exists(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🎨 Scene {scene_num} AI Art")

                anim_video_path = os.path.join(run_temp_dir, f"{scene_prefix}_animation.mp4")
                render_animation_by_index(
                    index=anim_cycle_idx,
                    image_path=raw_img_path,
                    output_path=anim_video_path,
                    duration=duration
                )
                anim_cycle_idx += 1

                render_individual_scene(
                    scene_type="animation",
                    audio_path=audio_path,
                    output_path=rendered_scene_mp4,
                    animation_video_path=anim_video_path
                )

            if artifact_callback and os.path.exists(rendered_scene_mp4):
                artifact_callback("video", rendered_scene_mp4, f"🎬 Scene {scene_num} Rendered MP4")

            scene_video_paths.append(rendered_scene_mp4)

        logger.info(f"Concatenating {len(scene_video_paths)} scenes...")
        if progress_callback:
            progress_callback("Concatenating all scenes into final video...")

        concat_all_scenes(scene_video_paths, output_video_path)

        if artifact_callback and os.path.exists(output_video_path):
            artifact_callback("final_video", output_video_path, "🏆 Final Full Combined Video")

        logger.info(f"Pipeline finished in {time.time() - start_time:.2f}s -> {output_video_path}")
        return output_video_path

    finally:
        if os.path.exists(run_temp_dir):
            shutil.rmtree(run_temp_dir, ignore_errors=True)
