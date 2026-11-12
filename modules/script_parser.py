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

# Whiteboard Engines
from modules.whiteboard_text_math import generate_text_math_whiteboard
from modules.whiteboard_image_anim import generate_native_whiteboard_animation

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


def compute_scene_hash(scene: dict, mode: str = "default") -> str:
    """Produces a deterministic 12-character hash for cache verification."""
    raw = f"{mode}_{scene.get('sentence', '')}_{scene.get('prompt', '')}_{scene.get('query', '')}_{scene.get('equation', '')}_{json.dumps(scene.get('headline', ''))}"
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]


def is_valid_media_file(file_path: str, min_size_bytes: int = 5000) -> bool:
    return os.path.exists(file_path) and os.path.getsize(file_path) >= min_size_bytes


def is_valid_video_file(file_path: str, min_size_bytes: int = 20000) -> bool:
    if not os.path.exists(file_path) or os.path.getsize(file_path) < min_size_bytes:
        return False
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            return False
        ret, frame = cap.read()
        cap.release()
        return bool(ret and frame is not None and frame.size > 0)
    except Exception:
        return False


def sanitize_mathtext(expr: str) -> str:
    cleaned = expr.strip()
    cleaned = re.sub(r'\\stackrel\{[^}]*\}\{=\}', r'\\approx', cleaned)
    cleaned = cleaned.replace(r'\implies', r'\Rightarrow')
    cleaned = cleaned.replace(r'\iff', r'\Leftrightarrow')
    return cleaned


def render_cinematic_equation_card(
    latex_equation: str,
    output_path: str,
    formula_name: str = "",
    width: int = 1760,
    height: int = 870
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    clean_expr = sanitize_mathtext(latex_equation)
    if not clean_expr.startswith("$"):
        clean_expr = f"${clean_expr}$"

    fig = None
    rendered_successfully = False

    for attempt in range(2):
        try:
            fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="none")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.patch.set_alpha(0.0)

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
            logger.warning(f"[EquationRenderer] Mathtext warning: {err}")
            if fig is not None:
                plt.close(fig)

    if not rendered_successfully:
        fallback_img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(fallback_img, formula_name or "MATHEMATICAL FORMULA", (width // 4, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imwrite(output_path, fallback_img)
        return output_path

    formula_bgr = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
    formula_alpha = rgba_buffer[:, :, 3] / 255.0

    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width * 0.5, height * 0.5
    r = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    gradient = np.clip(1.0 - 0.50 * r, 0.0, 1.0)[:, :, np.newaxis]

    dark_tone = np.array([26, 22, 18], dtype=np.float32)
    ambient_tone = np.array([10, 8, 8], dtype=np.float32)
    card_surface = (ambient_tone + (dark_tone - ambient_tone) * gradient).astype(np.uint8)

    card_surface[::60, :, :] = np.clip(card_surface[::60, :, :].astype(np.int16) + 12, 0, 255).astype(np.uint8)
    card_surface[:, ::60, :] = np.clip(card_surface[:, ::60, :].astype(np.int16) + 12, 0, 255).astype(np.uint8)

    alpha_3ch = formula_alpha[:, :, np.newaxis]
    final_card = (card_surface * (1.0 - alpha_3ch) + formula_bgr * alpha_3ch).astype(np.uint8)

    glow_mask = (formula_alpha * 255).astype(np.uint8)
    glow_blurred = cv2.GaussianBlur(glow_mask, (31, 31), 0)
    glow_layer = (glow_blurred[:, :, np.newaxis] / 255.0) * np.array([120, 40, 20], dtype=np.float32)
    final_card = np.clip(final_card.astype(np.float32) + glow_layer, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, final_card)
    return output_path


def _execute_tts_with_retry(sentence: str, target_audio_path: str, voice_name: str = None) -> tuple[str, float]:
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

    raise RuntimeError(f"TTS failed completely for: '{sentence[:40]}...'")


def generate_emergency_fallback_image(output_path: str):
    h, w = 1080, 1920
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (120, 100), (1800, 980), (30, 30, 30), 4)
    cv2.circle(canvas, (w // 2, h // 2), 220, (50, 50, 50), 3)
    cv2.imwrite(output_path, canvas)


def acquire_scene_image(scene: dict, raw_img_path: str, bing_cookies) -> str:
    if "equation" in scene and str(scene.get("equation", "")).strip():
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
            run_sync_generate(prompt_text, raw_img_path, cookie_u=bing_cookies)
            if is_valid_media_file(raw_img_path):
                return raw_img_path
        except Exception:
            pass

        try:
            fetch_image(search_q, raw_img_path)
            if is_valid_media_file(raw_img_path):
                return raw_img_path
        except Exception:
            pass
    else:
        try:
            fetch_image(search_q, raw_img_path)
            if is_valid_media_file(raw_img_path):
                return raw_img_path
        except Exception:
            pass

        if prompt_text:
            try:
                run_sync_generate(prompt_text, raw_img_path, cookie_u=bing_cookies)
                if is_valid_media_file(raw_img_path):
                    return raw_img_path
            except Exception:
                pass

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
    pipeline_mode: str = "default",  # "default" or "whiteboard"
    progress_callback=None,
    artifact_callback=None
) -> str:
    start_time = time.time()
    effective_cookie = bing_cookie or bing_cookie_u

    if not output_video_path:
        output_video_path = os.path.join(OUTPUT_DIR, f"master_video_{int(start_time)}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    available_bgs = []
    if pipeline_mode == "default":
        available_bgs = get_available_backgrounds(BG_DIR)
        if not available_bgs:
            raise RuntimeError(f"No background videos found in '{BG_DIR}/'.")

    # In Whiteboard mode, group scenes into 2-sentence pairs
    if pipeline_mode == "whiteboard":
        processed_units = []
        i = 0
        while i < len(script_data):
            if i + 1 < len(script_data):
                # Pair 2 sentences
                s1 = script_data[i]
                s2 = script_data[i+1]
                combined_sentence = f"{s1.get('sentence', '').strip()} {s2.get('sentence', '').strip()}"
                
                # Check for equations (prefer s1, fallback to s2)
                eq = s1.get("equation") or s2.get("equation") or ""
                formula_title = s1.get("formula_name") or s2.get("formula_name") or s1.get("title") or s2.get("title") or ""
                
                # PREFER FIRST SENTENCE (s1) PROMPT & QUERY AS REQUESTED
                combined_prompt = s1.get("prompt") or s2.get("prompt") or ""
                combined_query = s1.get("query") or s2.get("query") or ""

                processed_units.append({
                    "sentence": combined_sentence,
                    "sentence_a": s1.get("sentence", "").strip(),
                    "sentence_b": s2.get("sentence", "").strip(),
                    "equation": eq,
                    "formula_name": formula_title,
                    "prompt": combined_prompt,
                    "query": combined_query,
                    "unit_id": f"pair_{i+1}_{i+2}"
                })
                i += 2
            else:
                # Odd remaining single sentence
                s = script_data[i]
                processed_units.append({
                    "sentence": s.get("sentence", "").strip(),
                    "sentence_a": s.get("sentence", "").strip(),
                    "sentence_b": "",
                    "equation": s.get("equation", ""),
                    "formula_name": s.get("formula_name", ""),
                    "prompt": s.get("prompt", ""),
                    "query": s.get("query", ""),
                    "unit_id": f"single_{i+1}"
                })
                i += 1
    else:
        processed_units = [dict(s, unit_id=f"scene_{idx+1}") for idx, s in enumerate(script_data)]

    scene_video_paths = []
    bg_cycle_idx = 0
    anim_cycle_idx = 0
    total_units = len(processed_units)

    logger.info(f"Pipeline running {total_units} unit(s) in [{pipeline_mode.upper()}] mode.")

    def notify(msg: str):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass

    for idx, unit in enumerate(processed_units):
        unit_num = idx + 1
        unit_prefix = f"unit_{unit_num:02d}_{unit['unit_id']}"
        s_hash = compute_scene_hash(unit, mode=pipeline_mode)

        cached_mp4 = os.path.join(CHECKPOINT_DIR, f"{unit_prefix}_{s_hash}_rendered.mp4")

        # 1. Checkpoint Verification
        if is_valid_video_file(cached_mp4, 50000):
            logger.info(f"[{unit_prefix}] Reusing cached render...")
            notify(f"Unit {unit_num}/{total_units}: Reusing completed cache...")
            if artifact_callback:
                artifact_callback("video", cached_mp4, f"🎬 Scene {unit_num} (Cached)")
            scene_video_paths.append(cached_mp4)
            continue
        elif os.path.exists(cached_mp4):
            try:
                os.remove(cached_mp4)
            except Exception:
                pass

        sentence = unit.get("sentence", "").strip()
        if not sentence:
            continue

        # 2. Voiceover Generation
        notify(f"Unit {unit_num}/{total_units}: Generating Voiceover ({pipeline_mode.title()})...")
        audio_target = os.path.join(CHECKPOINT_DIR, f"{unit_prefix}_{s_hash}_tts.wav")
        if not is_valid_media_file(audio_target, 5000):
            audio_path, duration = _execute_tts_with_retry(sentence, audio_target, voice_name=voice_name)
        else:
            audio_path, duration = audio_target, get_stream_duration(audio_target)

        if artifact_callback:
            artifact_callback("audio", audio_path, f"🎤 Scene {unit_num} Audio ({duration:.1f}s)")

        rendered_scene_mp4 = cached_mp4

        # =============================================================
        # WHITEBOARD MODE: 2-Sentence Cohesive Hand-Drawing
        # =============================================================
        if pipeline_mode == "whiteboard":
            raw_wb_silent = os.path.join(CHECKPOINT_DIR, f"{unit_prefix}_{s_hash}_wb_raw.mp4")
            has_eq = bool(unit.get("equation", "").strip())

            if has_eq:
                notify(f"Unit {unit_num}/{total_units}: Drawing 2-Sentence Math Whiteboard ({duration:.1f}s)...")
                blocks = []
                title = unit.get("formula_name") or unit.get("title") or ""
                if title:
                    blocks.append({"type": "heading", "text": title, "size": 56, "space_after": 18})

                # First sentence
                if unit.get("sentence_a"):
                    blocks.append({"type": "malayalam", "text": unit["sentence_a"], "size": 42, "space_after": 16})

                # Math Formula
                blocks.append({"type": "math", "latex": unit["equation"], "size": 32, "box": True, "space_after": 18})

                # Second sentence underneath
                if unit.get("sentence_b"):
                    blocks.append({"type": "malayalam", "text": unit["sentence_b"], "size": 40, "space_after": 14})

                generate_text_math_whiteboard(
                    content_blocks=blocks,
                    output_path=raw_wb_silent,
                    total_duration_sec=duration,
                    fps=30
                )
            else:
                notify(f"Unit {unit_num}/{total_units}: Sketching Vector Line Art ({duration:.1f}s)...")
                wb_prompt = unit.get("prompt") or f"clean black and white line art vector sketch of {unit.get('query')}, pure white background, coloring book style"
                raw_img_path = os.path.join(CHECKPOINT_DIR, f"{unit_prefix}_{s_hash}_wb_image.jpg")

                if not is_valid_media_file(raw_img_path, 5000):
                    acquired = False
                    if effective_cookie:
                        try:
                            run_sync_generate(wb_prompt, raw_img_path, cookie_u=effective_cookie)
                            acquired = is_valid_media_file(raw_img_path)
                        except Exception as e:
                            logger.warning(f"Bing AI failed ({e}). Falling back...")

                    if not acquired:
                        try:
                            fetch_image(unit.get("query") or wb_prompt, raw_img_path)
                        except Exception:
                            generate_emergency_fallback_image(raw_img_path)

                if artifact_callback and is_valid_media_file(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🖼️ Scene {unit_num} Image")

                generate_native_whiteboard_animation(
                    img_path=raw_img_path,
                    output_path=raw_wb_silent,
                    total_duration_sec=duration,
                    fps=30
                )

            # Mux Whiteboard video with Cartesia audio and smooth fade
            render_individual_scene(
                scene_type="animation",
                audio_path=audio_path,
                output_path=rendered_scene_mp4,
                animation_video_path=raw_wb_silent
            )

        # =============================================================
        # DEFAULT MODE: Standard 1-Sentence Rhythm
        # =============================================================
        else:
            scene_mode = idx % 3
            query_text = unit.get("query", "").strip()
            prompt_text = unit.get("prompt", "").strip()
            headline_data = unit.get("headline", None)

            if scene_mode == 0:
                notify(f"Scene {unit_num}/{total_units}: Fetching Stock Video...")
                video_search_query = query_text or prompt_text or "space"
                raw_video_path = os.path.join(CHECKPOINT_DIR, f"{unit_prefix}_{s_hash}_raw_video.mp4")

                if not is_valid_video_file(raw_video_path, 20000):
                    if os.path.exists(raw_video_path):
                        try:
                            os.remove(raw_video_path)
                        except Exception:
                            pass
                    try:
                        fetch_landscape_video(video_search_query, raw_video_path)
                    except Exception as e:
                        logger.warning(f"Stock video fetch failed ({e}). Using background fallback...")

                    if not is_valid_video_file(raw_video_path, 20000):
                        fallback_bg = available_bgs[bg_cycle_idx % len(available_bgs)]
                        shutil.copyfile(fallback_bg, raw_video_path)

                render_individual_scene(
                    scene_type="video",
                    audio_path=audio_path,
                    output_path=rendered_scene_mp4,
                    video_path=raw_video_path
                )

            elif scene_mode == 1:
                notify(f"Scene {unit_num}/{total_units}: Rendering Stamp Card...")
                raw_img_path = os.path.join(CHECKPOINT_DIR, f"{unit_prefix}_{s_hash}_image.jpg")
                if not is_valid_media_file(raw_img_path, 5000):
                    acquire_scene_image(unit, raw_img_path, effective_cookie)

                if artifact_callback and is_valid_media_file(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🖼️ Scene {unit_num} Card")

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

            elif scene_mode == 2:
                notify(f"Scene {unit_num}/{total_units}: Generating Kinetic Animation...")
                raw_img_path = os.path.join(CHECKPOINT_DIR, f"{unit_prefix}_{s_hash}_ai_image.jpg")
                if not is_valid_media_file(raw_img_path, 5000):
                    if unit.get("equation", "").strip():
                        render_cinematic_equation_card(unit["equation"], raw_img_path, formula_name=unit.get("formula_name", ""))
                    else:
                        acquire_scene_image(unit, raw_img_path, effective_cookie)

                if artifact_callback and is_valid_media_file(raw_img_path):
                    artifact_callback("image", raw_img_path, f"🎨 Scene {unit_num} Visual")

                anim_video_path = os.path.join(CHECKPOINT_DIR, f"{unit_prefix}_{s_hash}_animation.mp4")
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

        if artifact_callback and is_valid_video_file(rendered_scene_mp4, 20000):
            artifact_callback("video", rendered_scene_mp4, f"🎬 Scene {unit_num} Rendered MP4")

        scene_video_paths.append(rendered_scene_mp4)

        if unit_num < total_units and rest_seconds_between_scenes > 0:
            logger.info(f"Cooldown resting for {rest_seconds_between_scenes}s...")
            for remaining in range(rest_seconds_between_scenes, 0, -5):
                notify(f"⏳ Cooldown: Resting CPU for {remaining}s...")
                time.sleep(min(5, remaining))

    notify("Concatenating all scenes into master video...")
    concat_all_scenes(scene_video_paths, output_video_path)

    if artifact_callback and is_valid_video_file(output_video_path, 50000):
        artifact_callback("final_video", output_video_path, "🏆 Final Full Combined Video")

    logger.info(f"Pipeline finished in {time.time() - start_time:.2f}s -> {output_video_path}")
    return output_video_path
