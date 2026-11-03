import asyncio
import glob
import json
import os
import shutil
import time

from modules.cartesia_tts import generate_speech
from modules.pixabay_fetcher import fetch_landscape_video, fetch_image
from modules.bing_fetcher import generate_bing_image
from animations.animation_router import render_animation_by_index
from modules.ffmpeg_compositor import render_individual_scene, concat_all_scenes

BASE_DIR = os.getcwd()
BG_DIR = os.path.join(BASE_DIR, "assets", "backgrounds")
TEMP_DIR = os.path.join(BASE_DIR, "storage", "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "storage", "outputs")


def get_available_backgrounds() -> list[str]:
    """Finds all looping background MP4s inside assets/backgrounds."""
    bgs = sorted(glob.glob(os.path.join(BG_DIR, "*.mp4")))
    if not bgs:
        # Fallback check directly in root assets
        bgs = sorted(glob.glob(os.path.join(BASE_DIR, "assets", "*.mp4")))
    return bgs


async def process_video_pipeline(
    script_data: list[dict] | str,
    bing_cookie_u: str,
    voice_name: str = "Shreeraj_HindiProfile",
    progress_callback=None
) -> str:
    """
    Parses sentence-by-sentence JSON script and produces the final assembled video.

    JSON Schema expected:
    [
        {
            "sentence": "Malayalam text for TTS",
            "query": "Pixabay search keywords",
            "prompt": "Bing image generation prompt (optional)",
            "source": "bing" or "pixabay" (defaults to "bing" if prompt exists, else "pixabay")
        },
        ...
    ]
    """
    # Parse JSON if passed as string
    if isinstance(script_data, str):
        scenes = json.loads(script_data)
    else:
        scenes = script_data

    if not scenes or not isinstance(scenes, list):
        raise ValueError("Script must be a non-empty JSON array of scene objects.")

    run_id = f"run_{int(time.time())}"
    run_temp_dir = os.path.join(TEMP_DIR, run_id)
    os.makedirs(run_temp_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    available_bgs = get_available_backgrounds()
    scene_video_paths = []

    total_scenes = len(scenes)
    bg_counter = 0
    animation_counter = 0

    print(f"\n=======================================================")
    print(f"Starting Video Engine Pipeline: {total_scenes} Scenes")
    print(f"=======================================================\n")

    try:
        for idx, scene in enumerate(scenes, start=1):
            scene_text = scene.get("sentence", "").strip()
            query_text = scene.get("query", "").strip()
            prompt_text = scene.get("prompt", "").strip()
            source_pref = scene.get("source", "").lower()

            if not scene_text:
                continue

            scene_prefix = f"scene_{idx:03d}"
            audio_path = os.path.join(run_temp_dir, f"{scene_prefix}_audio.wav")
            scene_mp4 = os.path.join(run_temp_dir, f"{scene_prefix}_final.mp4")

            if progress_callback:
                progress_callback(idx, total_scenes, f"Synthesizing audio for Scene {idx}/{total_scenes}...")

            # 1. Synthesize Cartesia Speech
            tts_duration = await generate_speech(
                text=scene_text,
                output_wav_path=audio_path,
                voice_name=voice_name
            )

            # -------------------------------------------------------------
            # SENTENCE 1: Standalone Unique Pixabay Landscape Video
            # -------------------------------------------------------------
            if idx == 1:
                if progress_callback:
                    progress_callback(idx, total_scenes, f"Downloading Pixabay landscape video for Scene 1...")

                raw_video_path = os.path.join(run_temp_dir, f"{scene_prefix}_raw_video.mp4")
                search_q = query_text if query_text else prompt_text
                fetch_landscape_video(search_q, raw_video_path)

                render_individual_scene(
                    scene_type="standalone",
                    visual_path=raw_video_path,
                    audio_path=audio_path,
                    duration_sec=tts_duration,
                    output_path=scene_mp4
                )
                scene_video_paths.append(scene_mp4)

            # -------------------------------------------------------------
            # SENTENCE 2, 4, 6...: Background Video + Scaled Centered Image
            # -------------------------------------------------------------
            elif idx % 2 == 0:
                if progress_callback:
                    progress_callback(idx, total_scenes, f"Acquiring image & compositing background for Scene {idx}...")

                # Fetch/Generate Image Asset
                image_path = os.path.join(run_temp_dir, f"{scene_prefix}_image.jpg")
                if (source_pref == "bing" or prompt_text) and bing_cookie_u:
                    await generate_bing_image(prompt_text or query_text, image_path, bing_cookie_u)
                else:
                    fetch_image(query_text or prompt_text, image_path)

                # Select rotating background video
                if available_bgs:
                    selected_bg = available_bgs[bg_counter % len(available_bgs)]
                    bg_counter += 1
                else:
                    raise FileNotFoundError("No background video files found in assets/backgrounds/")

                render_individual_scene(
                    scene_type="bg_image",
                    visual_path=image_path,
                    audio_path=audio_path,
                    duration_sec=tts_duration,
                    output_path=scene_mp4,
                    bg_video_path=selected_bg
                )
                scene_video_paths.append(scene_mp4)

            # -------------------------------------------------------------
            # SENTENCE 3, 5, 7...: Procedural Kinetic OpenCV Animation
            # -------------------------------------------------------------
            else:
                if progress_callback:
                    progress_callback(idx, total_scenes, f"Rendering kinetic animation for Scene {idx}...")

                # Fetch/Generate Image Asset for the Animation
                image_path = os.path.join(run_temp_dir, f"{scene_prefix}_anim_image.jpg")
                if (source_pref == "bing" or prompt_text) and bing_cookie_u:
                    await generate_bing_image(prompt_text or query_text, image_path, bing_cookie_u)
                else:
                    fetch_image(query_text or prompt_text, image_path)

                # Render procedural animation with exact TTS audio duration
                raw_anim_video = os.path.join(run_temp_dir, f"{scene_prefix}_raw_anim.mp4")
                render_animation_by_index(
                    image_path=image_path,
                    output_path=raw_anim_video,
                    duration_sec=tts_duration,
                    cycle_index=animation_counter
                )
                animation_counter += 1

                # Mux with audio and apply standard 0.4s fade-out
                render_individual_scene(
                    scene_type="animation",
                    visual_path=raw_anim_video,
                    audio_path=audio_path,
                    duration_sec=tts_duration,
                    output_path=scene_mp4
                )
                scene_video_paths.append(scene_mp4)

        # -------------------------------------------------------------
        # FINAL ZERO-LOSS STITCHING PASS
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback(total_scenes, total_scenes, "Merging all scenes into final master video...")

        final_master_path = os.path.join(OUTPUT_DIR, f"master_{run_id}.mp4")
        concat_all_scenes(scene_video_paths, final_master_path, run_temp_dir)

        print(f"\nPipeline Finished! Master video: {final_master_path}")
        return final_master_path

    finally:
        # Automatic temporary file cleanup
        try:
            shutil.rmtree(run_temp_dir, ignore_errors=True)
            print(f"Purged session temp directory: {run_temp_dir}")
        except Exception as e:
            print(f"Temp cleanup notice: {e}")
