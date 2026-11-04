import asyncio
import concurrent.futures
import glob
import inspect
import json
import logging
import os
import re
import shutil
import time

from modules.cartesia_tts import generate_speech
from modules.pixabay_fetcher import fetch_landscape_video, fetch_image
from modules.bing_fetcher import generate_bing_image
from animations.animation_router import render_animation_by_index
from modules.ffmpeg_compositor import (
    render_individual_scene,
    concat_all_scenes,
    get_stream_duration
)

# =====================================================================
# DIRECTORY CONSTANTS (Exported for app.py and Telegram bot)
# =====================================================================
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
    """
    Safely executes generate_speech whether it is an async coroutine or a sync function,
    and returns (audio_path, duration_in_seconds).
    """
    res = generate_speech(sentence, target_audio_path)

    # Resolve coroutine if generate_speech is an async def
    if inspect.iscoroutine(res):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    res = pool.submit(asyncio.run, res).result()
            else:
                res = loop.run_until_complete(res)
        except RuntimeError:
            res = asyncio.run(res)

    # Parse return format safely
    if isinstance(res, (tuple, list)) and len(res) == 2:
        out_path, dur = res
        return str(out_path), float(dur)
    elif isinstance(res, str) and os.path.exists(res):
        dur = get_stream_duration(res)
        return res, dur
    else:
        dur = get_stream_duration(target_audio_path)
        return target_audio_path, dur


def get_available_backgrounds(assets_dir: str = BG_DIR) -> list[str]:
    """Locates all looping background MP4s inside the assets directory."""
    patterns = [os.path.join(assets_dir, "*.mp4"), os.path.join(assets_dir, "*.MP4")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def evaluate_image_source_decision(scene: dict) -> str:
    """Intelligently determines whether an image should be AI-generated or fetched."""
    explicit = scene.get("image_source") or scene.get("source")
    if explicit:
        val = str(explicit).lower().strip()
        if val in ["bing", "ai", "dall-e", "dalle", "gen"]:
            return "bing"
        if val in ["fetch", "wikimedia", "scrape", "stock"]:
            return "fetch"

    prompt_text = scene.get("prompt", "").lower()
    query_text = scene.get("query", "").lower()
    combined_text = f"{prompt_text} {query_text}"

    if any(kw in combined_text for kw in ABSTRACT_AI_KEYWORDS):
        return "bing"

    return "fetch"


def process_video_pipeline(
    script_data: list[dict],
    output_video_path: str,
    temp_dir: str = "temp_pipeline",
    bing_cookie: str = "",
    progress_callback=None,
    artifact_callback=None
) -> str:
    """
    Renders video following the 3-beat cycle:
      - Scene (i % 3 == 0): Landscape Video (Mixkit / Wikimedia)
      - Scene (i % 3 == 1): BG Loop + Centered Image + Headline (Type 2)
      - Scene (i % 3 == 2): Procedural OpenCV Kinetic Animation (Strict AI)
    """
    start_time = time.time()
    run_temp_dir = os.path.join(temp_dir, f"run_{int(start_time)}")
    os.makedirs(run_temp_dir, exist_ok=True)

    available_bgs = get_available_backgrounds(BG_DIR)
    if not available_bgs:
        raise RuntimeError(f"No looping background videos found in '{BG_DIR}/'.")

    scene_video_paths = []
    bg_cycle_idx = 0
    anim_cycle_idx = 0
    total_scenes = len(script_data)

    logger.info(f"Pipeline started for {total_scenes} scenes in: {run_temp_dir}")

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

            # -------------------------------------------------------------
            # STEP 1: TTS Audio Generation (Async-Safe)
            # -------------------------------------------------------------
            logger.info(f"[{scene_prefix}] Generating Voiceover...")
            if progress_callback:
                progress_callback(f"Scene {scene_num}/{total_scenes}: Synthesizing Voiceover...")

            audio_target = os.path.join(run_temp_dir, f"{scene_prefix}_tts.wav")
            audio_path, duration = _execute_tts(sentence, audio_target)

            if artifact_callback:
                artifact_callback("audio", audio_path, f"🎤 Scene {scene_num} Audio ({duration:.1f}s)")

            scene_mode = idx % 3
            rendered_scene_mp4 = os.path.join(run_temp_dir, f"{scene_prefix}_rendered.mp4")

            # -------------------------------------------------------------
            # MODE 0: LANDSCAPE B-ROLL VIDEO
            # -------------------------------------------------------------
            if scene_mode == 0:
                logger.info(f"[{scene_prefix}] Rendering Landscape Video")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Fetching Cinematic Footage...")

                video_search_query = query_text or prompt_text or "cinematic nature space abstract"
                raw_video_path = os.path.join(run_temp_dir, f"{scene_prefix}_raw_video.mp4")
                fetch_landscape_video(video_search_query, raw_video_path)

                render_individual_scene(
                    scene_type="video",
                    audio_path=audio_path,
                    output_path=rendered_scene_mp4,
                    video_path=raw_video_path
                )

            # -------------------------------------------------------------
            # MODE 1: BACKGROUND LOOP + STAMP IMAGE + HEADLINE
            # -------------------------------------------------------------
            elif scene_mode == 1:
                logger.info(f"[{scene_prefix}] Rendering Type 2 Stamp Drop")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Sourcing Image & Rendering Stamp...")

                chosen_source = evaluate_image_source_decision(scene)
                raw_img_path = os.path.join(run_temp_dir, f"{scene_prefix}_image.jpg")

                if chosen_source == "bing" and prompt_text:
                    generate_bing_image(prompt_text, raw_img_path, cookie_u=bing_cookie)
                else:
                    try:
                        fetch_image(query_text or prompt_text, raw_img_path)
                    except Exception as err:
                        logger.warning(f"Fetch failed ({err}). Falling back to Bing AI...")
                        generate_bing_image(prompt_text or query_text, raw_img_path, cookie_u=bing_cookie)

                if artifact_callback and os.path.exists(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🖼️ Scene {scene_num} Image ({chosen_source.upper()})")

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

            # -------------------------------------------------------------
            # MODE 2: KINETIC ANIMATION (STRICT AI)
            # -------------------------------------------------------------
            elif scene_mode == 2:
                logger.info(f"[{scene_prefix}] Rendering Type 3 Kinetic Animation")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Generating AI Art & Animation...")

                raw_img_path = os.path.join(run_temp_dir, f"{scene_prefix}_ai_image.jpg")
                generate_bing_image(prompt_text or query_text, raw_img_path, cookie_u=bing_cookie)

                if artifact_callback and os.path.exists(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🎨 Scene {scene_num} AI Art for Animation")

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

            # Emit Scene Video Artifact
            if artifact_callback and os.path.exists(rendered_scene_mp4):
                artifact_callback("video", rendered_scene_mp4, f"🎬 Scene {scene_num} Rendered MP4")

            scene_video_paths.append(rendered_scene_mp4)

        # -----------------------------------------------------------------
        # STEP 2: Concat All Scenes
        # -----------------------------------------------------------------
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
