import glob
import json
import logging
import os
import shutil
import time

from modules.cartesia_tts import generate_speech
from modules.pixabay_fetcher import fetch_landscape_video, fetch_image
from modules.bing_fetcher import generate_bing_image
from animations.animation_router import render_animation_by_index
from modules.ffmpeg_compositor import render_individual_scene, concat_all_scenes

# Configure logger
logger = logging.getLogger("VideoEngine.Pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Concepts that warrant AI generation over stock photo search
ABSTRACT_AI_KEYWORDS = {
    "quantum", "subatomic", "particle", "ghost", "neutrino", "matrix", "cyber",
    "futuristic", "citadel", "portal", "dimension", "black hole", "singularity",
    "supernova", "cosmic energy", "abstract", "alien", "energy silhouette"
}


def evaluate_image_source_decision(scene: dict) -> str:
    """
    Decides whether an image should be AI-generated ('bing') or scraped ('fetch'):
    1. Explicit override in JSON takes precedence.
    2. Prompt complexity analysis for abstract/sci-fi concepts.
    3. Scraped stock photo for real tangible entities.
    """
    explicit = scene.get("image_source") or scene.get("source")
    if explicit and explicit.lower() in ["bing", "ai", "dall-e"]:
        return "bing"
    if explicit and explicit.lower() in ["fetch", "wikimedia", "scrape"]:
        return "fetch"

    prompt_text = scene.get("prompt", "").lower()
    query_text = scene.get("query", "").lower()
    combined_text = f"{prompt_text} {query_text}"

    if any(kw in combined_text for kw in ABSTRACT_AI_KEYWORDS):
        return "bing"

    return "fetch"


def get_available_backgrounds(assets_dir: str = "assets/backgrounds") -> list[str]:
    """Finds all looping background MP4s inside assets/backgrounds."""
    patterns = [os.path.join(assets_dir, "*.mp4"), os.path.join(assets_dir, "*.MP4")]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def process_video_pipeline(
    script_data: list[dict],
    output_video_path: str,
    temp_dir: str = "temp_pipeline",
    bing_cookie: str = "",
    progress_callback=None,
    artifact_callback=None
) -> str:
    """
    Executes the video assembly pipeline and emits each intermediate asset:
    - (artifact_callback: callable(artifact_type, file_path, caption))
    """
    start_time = time.time()
    run_temp_dir = os.path.join(temp_dir, f"run_{int(start_time)}")
    os.makedirs(run_temp_dir, exist_ok=True)

    available_bgs = get_available_backgrounds()
    if not available_bgs:
        raise RuntimeError("No background video files found in assets/backgrounds/")

    scene_video_paths = []
    bg_cycle_idx = 0
    anim_cycle_idx = 0
    total_scenes = len(script_data)

    logger.info(f"Starting pipeline execution for {total_scenes} scenes.")

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
            # STEP A: Generate TTS Audio & Emit Artifact
            # -------------------------------------------------------------
            logger.info(f"[{scene_prefix}] Generating Cartesia TTS voiceover...")
            if progress_callback:
                progress_callback(f"Scene {scene_num}/{total_scenes}: Generating Voiceover...")

            audio_path = os.path.join(run_temp_dir, f"{scene_prefix}_tts.wav")
            audio_path, duration = generate_speech(sentence, audio_path)

            if artifact_callback:
                artifact_callback("audio", audio_path, f"🎤 Scene {scene_num} Audio ({duration:.1f}s)")

            # Determine scene pattern: 0 = Video, 1 = BG + Image, 2 = Kinetic Animation
            scene_mode = idx % 3
            rendered_scene_mp4 = os.path.join(run_temp_dir, f"{scene_prefix}_rendered.mp4")

            # -------------------------------------------------------------
            # SCENE MODE 0: LANDSCAPE B-ROLL VIDEO (Scenes 1, 4, 7...)
            # -------------------------------------------------------------
            if scene_mode == 0:
                logger.info(f"[{scene_prefix}] Mode 0: Landscape B-Roll Video")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Fetching Landscape Footage...")

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
            # SCENE MODE 1: BACKGROUND LOOP + STAMP IMAGE + HEADLINE (Scenes 2, 5...)
            # -------------------------------------------------------------
            elif scene_mode == 1:
                logger.info(f"[{scene_prefix}] Mode 1: Animated Background + Stamp Image")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Acquiring Image & Rendering Stamp Drop...")

                # Intelligent AI vs Fetch Evaluation
                chosen_source = evaluate_image_source_decision(scene)
                raw_img_path = os.path.join(run_temp_dir, f"{scene_prefix}_image.jpg")

                if chosen_source == "bing" and prompt_text:
                    logger.info(f"[{scene_prefix}] Abstract prompt detected. Generating via Bing MAI1 AI...")
                    generate_bing_image(prompt_text, raw_img_path, cookie_u=bing_cookie)
                else:
                    logger.info(f"[{scene_prefix}] Fetching stock image from Wikimedia Commons...")
                    try:
                        fetch_image(query_text or prompt_text, raw_img_path)
                    except Exception as e:
                        logger.warning(f"Image fetch failed ({e}). Fallback to Bing AI...")
                        generate_bing_image(prompt_text or query_text, raw_img_path, cookie_u=bing_cookie)

                # Emit Image Artifact to Telegram
                if artifact_callback and os.path.exists(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🖼️ Scene {scene_num} Image ({chosen_source.upper()})")

                # Pick next looping background
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
            # SCENE MODE 2: PROCEDURAL OPENCV KINETIC ANIMATION (Scenes 3, 6...)
            # -------------------------------------------------------------
            elif scene_mode == 2:
                logger.info(f"[{scene_prefix}] Mode 2: Kinetic Animation (Strict AI Generation)")
                if progress_callback:
                    progress_callback(f"Scene {scene_num}/{total_scenes}: Generating AI Art & Kinetic Animation...")

                # Mode 2 strictly uses AI generation
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

            # Emit Scene Video Artifact to Telegram
            if artifact_callback and os.path.exists(rendered_scene_mp4):
                artifact_callback("video", rendered_scene_mp4, f"🎬 Scene {scene_num} Rendered MP4")

            scene_video_paths.append(rendered_scene_mp4)

        # -----------------------------------------------------------------
        # STEP B: Concatenate All Scenes
        # -----------------------------------------------------------------
        logger.info("Concatenating all intermediate scenes...")
        if progress_callback:
            progress_callback("Concatenating all rendered scenes...")

        concat_all_scenes(scene_video_paths, output_video_path)

        if artifact_callback and os.path.exists(output_video_path):
            artifact_callback("final_video", output_video_path, "🏆 Final Full Combined Video")

        return output_video_path

    finally:
        if os.path.exists(run_temp_dir):
            shutil.rmtree(run_temp_dir, ignore_errors=True)
