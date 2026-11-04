import glob
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
from modules.ffmpeg_compositor import render_individual_scene, concat_all_scenes

# =====================================================================
# DIRECTORY CONSTANTS (Required by app.py and Telegram bot)
# =====================================================================
BG_DIR = "assets/backgrounds"
OUTPUT_DIR = "outputs"

os.makedirs(BG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logger Configuration
logger = logging.getLogger("VideoEngine.Pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# Semantic keywords indicating abstract or complex imagery that requires AI generation
ABSTRACT_AI_KEYWORDS = {
    "quantum", "subatomic", "particle", "ghost", "neutrino", "matrix", "cyber",
    "futuristic", "citadel", "portal", "dimension", "black hole", "singularity",
    "supernova", "cosmic energy", "abstract", "alien", "energy silhouette",
    "glowing energy", "laser burst", "cosmic rays", "accretion disk"
}


def get_available_backgrounds(assets_dir: str = BG_DIR) -> list[str]:
    """Finds all looping background MP4s inside the configured backgrounds directory."""
    patterns = [os.path.join(assets_dir, "*.mp4"), os.path.join(assets_dir, "*.MP4")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def evaluate_image_source_decision(scene: dict) -> str:
    """
    Intelligently determines whether an image should be AI-generated ('bing')
    or scraped ('fetch') for Type 2 scenes:
    1. Direct override via 'image_source' or 'source' in JSON takes priority.
    2. Deep prompt and query inspection for abstract/sci-fi concepts -> AI.
    3. Real-world physical entities (nature, animals, historical places) -> Stock Fetch.
    """
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
    Executes the video assembly pipeline with the 3-beat rhythm:
      - Scene (idx % 3 == 0): Landscape Stock Video (Mixkit / Wikimedia)
      - Scene (idx % 3 == 1): Type 2 - Looping BG + Stamp-Drop Image + Styled Typography
      - Scene (idx % 3 == 2): Type 3 - Procedural OpenCV Kinetic Animation (Strict AI)

    Emits each generated asset (TTS audio, images, scene videos, and final mix)
    via artifact_callback for real-time Telegram delivery.
    """
    start_time = time.time()
    run_temp_dir = os.path.join(temp_dir, f"run_{int(start_time)}")
    os.makedirs(run_temp_dir, exist_ok=True)

    available_bgs = get_available_backgrounds(BG_DIR)
    if not available_bgs:
        raise RuntimeError(f"No looping background videos found in '{BG_DIR}/'. Please add BG1.mp4, BG2.mp4, etc.")

    scene_video_paths = []
    bg_cycle_idx = 0
    anim_cycle_idx = 0
    total_scenes = len(script_data)

    logger.info(f"Pipeline started for {total_scenes} scenes. Temporary workspace: {run_temp_dir}")

    try:
        for idx, scene in enumerate(script_data):
            scene_num = idx + 1
            scene_prefix = f"scene_{scene_num:02d}"

            sentence = scene.get("sentence", "").strip()
            query_text = scene.get("query", "").strip()
            prompt_text = scene.get("prompt", "").strip()
            headline_data = scene.get("headline", None)

            if not sentence:
                logger.warning(f"Scene {scene_num} is empty. Skipping.")
                continue

            # -------------------------------------------------------------
            # STEP 1: TTS Synthesis & Real-Time Audio Artifact Emission
            # -------------------------------------------------------------
            logger.info(f"[{scene_prefix}] Generating Cartesia TTS voiceover...")
            if progress_callback:
                progress_callback(f"Scene {scene_num}/{total_scenes}: Synthesizing Voiceover...")

            audio_path = os.path.join(run_temp_dir, f"{scene_prefix}_tts.wav")
            audio_path, duration = generate_speech(sentence, audio_path)

            if artifact_callback:
                artifact_callback("audio", audio_path, f"🎤 Scene {scene_num} Audio ({duration:.1f}s)")

            # Determine Scene Beat: 0 = Video, 1 = Stamp Frame, 2 = Kinetic Animation
            scene_mode = idx % 3
            rendered_scene_mp4 = os.path.join(run_temp_dir, f"{scene_prefix}_rendered.mp4")

            # -------------------------------------------------------------
            # MODE 0: LANDSCAPE B-ROLL VIDEO (Scenes 1, 4, 7...)
            # -------------------------------------------------------------
            if scene_mode == 0:
                logger.info(f"[{scene_prefix}] Mode 0: Unique Landscape Video")
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
            # MODE 1: BACKGROUND LOOP + STAMP IMAGE + HEADLINE (Scenes 2, 5...)
            # -------------------------------------------------------------
            elif scene_mode == 1:
                logger.info(f"[{scene_prefix}] Mode 1: Animated Background + Stamp Image Frame")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Sourcing Image & Rendering Stamp Scene...")

                chosen_source = evaluate_image_source_decision(scene)
                raw_img_path = os.path.join(run_temp_dir, f"{scene_prefix}_image.jpg")

                if chosen_source == "bing" and prompt_text:
                    logger.info(f"[{scene_prefix}] Complex concept identified. Generating AI image via Bing...")
                    generate_bing_image(prompt_text, raw_img_path, cookie_u=bing_cookie)
                else:
                    logger.info(f"[{scene_prefix}] Fetching stock image from Wikimedia Commons...")
                    try:
                        fetch_image(query_text or prompt_text, raw_img_path)
                    except Exception as err:
                        logger.warning(f"Stock fetch failed ({err}). Falling back to Bing AI...")
                        generate_bing_image(prompt_text or query_text, raw_img_path, cookie_u=bing_cookie)

                # Emit Image Artifact to Telegram
                if artifact_callback and os.path.exists(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🖼️ Scene {scene_num} Image ({chosen_source.upper()})")

                # Cycle through background videos
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
            # MODE 2: PROCEDURAL OPENCV KINETIC ANIMATION (Scenes 3, 6...)
            # -------------------------------------------------------------
            elif scene_mode == 2:
                logger.info(f"[{scene_prefix}] Mode 2: Kinetic Animation (Strict AI Generation)")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Generating AI Art & Kinetic Animation...")

                # Type 3 strictly uses AI generation
                raw_img_path = os.path.join(run_temp_dir, f"{scene_prefix}_ai_image.jpg")
                generate_bing_image(prompt_text or query_text, raw_img_path, cookie_u=bing_cookie)

                # Emit AI Image Artifact to Telegram
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

            # -------------------------------------------------------------
            # STEP 2: Emit Individual Scene MP4 to Telegram
            # -------------------------------------------------------------
            if artifact_callback and os.path.exists(rendered_scene_mp4):
                artifact_callback("video", rendered_scene_mp4, f"🎬 Scene {scene_num} Rendered MP4")

            scene_video_paths.append(rendered_scene_mp4)

        # -----------------------------------------------------------------
        # STEP 3: Concatenate All Scenes into Final Combined Video
        # -----------------------------------------------------------------
        logger.info(f"Assembling final video from {len(scene_video_paths)} rendered scenes...")
        if progress_callback:
            progress_callback("Concatenating all scenes into final video...")

        concat_all_scenes(scene_video_paths, output_video_path)

        # Emit Final Video Artifact to Telegram
        if artifact_callback and os.path.exists(output_video_path):
            artifact_callback("final_video", output_video_path, "🏆 Final Full Combined Video")

        logger.info(f"Pipeline finished successfully in {time.time() - start_time:.2f}s -> {output_video_path}")
        return output_video_path

    finally:
        # Workspace cleanup
        if os.path.exists(run_temp_dir):
            shutil.rmtree(run_temp_dir, ignore_errors=True)
