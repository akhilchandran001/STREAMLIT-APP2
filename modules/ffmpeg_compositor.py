from collections import Counter
import gc
import logging
import os
import subprocess
import sys
from animations.type2_stamp_drop import render_animated_stamp_drop

logger = logging.getLogger("VideoEngine.Compositor")


def run_command(cmd: list[str]) -> str:
    """Executes FFmpeg CLI commands with error traceback logging."""
    logger.debug(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg command failed:\n{result.stderr}")
        raise RuntimeError(f"FFmpeg execution failed: {result.stderr.strip()}")
    return result.stdout


def get_stream_duration(file_path: str) -> float:
    """Extracts exact float duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    output = run_command(cmd).strip()
    return float(output)


def _get_video_dimensions(file_path: str) -> tuple[int, int]:
    """Returns (width, height) of the first video stream via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        file_path
    ]
    try:
        out = run_command(cmd).strip().splitlines()[0]
        w, h = out.split("x")
        return int(w), int(h)
    except Exception:
        return (1920, 1080)


def calculate_fade_times(duration: float, fade_len: float = 0.4) -> tuple[float, float]:
    """Standardizes to a 0.4-second fade-out at the scene boundary."""
    actual_fade = min(fade_len, duration / 2.0)
    start_time = max(0.0, duration - actual_fade)
    return start_time, actual_fade


def render_individual_scene(
    scene_type: str,
    audio_path: str | None,
    output_path: str,
    video_path: str = None,
    background_video_path: str = None,
    foreground_image_path: str = None,
    animation_video_path: str = None,
    headline_data: dict = None,
    sfx_audio_path: str = None,
    sfx_volume: float = 0.45,
    fallback_duration: float = 8.0,
) -> str:
    """
    Renders any scene into a standardized MP4.
    - Preserves native generated dimensions on `animation` (no white background padding).
    - Mixes TTS + Newspaper SFX cleanly with gentle ducking so TTS is louder than SFX
      while keeping paper drop & highlighter sounds clearly audible.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    has_tts = bool(audio_path and os.path.exists(audio_path))
    has_sfx = bool(sfx_audio_path and os.path.exists(sfx_audio_path))

    if has_tts:
        duration = get_stream_duration(audio_path)
    elif has_sfx:
        duration = get_stream_duration(sfx_audio_path)
    else:
        duration = float(fallback_duration)

    fade_start, fade_len = calculate_fade_times(duration, 0.4)
    v_fade = f"fade=t=out:st={fade_start:.3f}:d={fade_len:.3f}"
    a_fade = f"afade=t=out:st={fade_start:.3f}:d={fade_len:.3f}"

    a_std = "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo"
    temp_raw = f"{output_path}.raw.mp4"

    # SCENE TYPE 1: Unique Landscape Video (Scenes 1, 4...)
    if scene_type == "video":
        filter_str = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=increase,"
            f"crop=1920:1080,fps=30,format=yuv420p,setsar=1,{v_fade}[v];"
            f"[1:a]{a_std},volume=1.20,alimiter=limit=0.97,{a_fade}[a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", video_path,
            "-i", audio_path,
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "2",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
            "-t", f"{duration:.3f}",
            output_path
        ]
        run_command(cmd)

    # SCENE TYPE 2: Animated Drop Stamp + Looping Background + Reference Typography
    elif scene_type == "background_with_image":
        render_animated_stamp_drop(
            bg_video_path=background_video_path,
            foreground_image_path=foreground_image_path,
            output_path=temp_raw,
            duration=duration,
            headline_data=headline_data,
            fps=30
        )
        filter_str = (
            f"[0:v]fps=30,format=yuv420p,setsar=1,{v_fade}[v];"
            f"[1:a]{a_std},volume=1.20,alimiter=limit=0.97,{a_fade}[a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_raw,
            "-i", audio_path,
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "2",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
            "-t", f"{duration:.3f}",
            output_path
        ]
        run_command(cmd)
        if os.path.exists(temp_raw):
            os.remove(temp_raw)

    # SCENE TYPE 3: Whiteboard / Kinetic Animation / Newspaper (Native Dimensions Preserved)
    elif scene_type == "animation":
        if has_tts and has_sfx:
            sfx_gain = max(0.0, min(0.75, float(sfx_volume) * 0.90))
            filter_str = (
                f"[0:v]fps=30,format=yuv420p,setsar=1,{v_fade}[v];"
                f"[1:a]{a_std},volume=1.35,apad,asplit=2[tts_mix][tts_sc];"
                f"[2:a]{a_std},volume={sfx_gain:.3f},apad[sfx_raw];"
                f"[sfx_raw][tts_sc]sidechaincompress=threshold=0.04:ratio=3.5:attack=10:release=160:makeup=1[sfx_ducked];"
                f"[tts_mix][sfx_ducked]amerge=inputs=2,pan=stereo|c0=c0+c2|c1=c1+c3,"
                f"alimiter=limit=0.97:attack=5:release=50,{a_fade}[a]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", animation_video_path,
                "-i", audio_path,
                "-i", sfx_audio_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "2",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
                "-t", f"{duration:.3f}",
                output_path
            ]
        elif has_sfx and not has_tts:
            sfx_gain = max(0.0, min(1.0, float(sfx_volume) if sfx_volume > 0 else 0.85))
            filter_str = f"[0:v]fps=30,format=yuv420p,setsar=1,{v_fade}[v];[1:a]{a_std},volume={sfx_gain:.3f},alimiter=limit=0.97,{a_fade}[a]"
            cmd = [
                "ffmpeg", "-y",
                "-i", animation_video_path,
                "-i", sfx_audio_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "2",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
                "-t", f"{duration:.3f}",
                output_path
            ]
        else:
            filter_str = f"[0:v]fps=30,format=yuv420p,setsar=1,{v_fade}[v];[1:a]{a_std},volume=1.20,alimiter=limit=0.97,{a_fade}[a]"
            cmd = [
                "ffmpeg", "-y",
                "-i", animation_video_path,
                "-i", audio_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "2",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
                "-t", f"{duration:.3f}",
                output_path
            ]
        run_command(cmd)

    logger.info(f"Scene successfully rendered: {output_path} ({duration:.2f}s)")
    return output_path


def concat_all_scenes(scene_paths: list[str], final_output_path: str) -> str:
    """
    Low-memory concatenation safe for 60+ scenes on Streamlit Cloud:
    1. Finds the dominant resolution among rendered scenes.
    2. Normalizes ONLY mismatched scenes one-by-one (sequentially, ~150MB RAM).
    3. Concatenates all scenes using FFmpeg concat demuxer (-c copy, ~20MB RAM).
    """
    if not scene_paths:
        raise ValueError("No rendered scene MP4s provided to concatenate.")

    os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)
    dims = [_get_video_dimensions(p) for p in scene_paths]
    unique_dims = set(dims)

    temp_normalized = []
    final_inputs = []

    if len(unique_dims) == 1:
        final_inputs = [os.path.abspath(p) for p in scene_paths]
    else:
        # Pick the most common resolution so most scenes skip re-encoding completely
        target_w, target_h = Counter(dims).most_common(1)[0][0]
        logger.info(f"Normalizing {len(scene_paths)} scenes sequentially to {target_w}x{target_h}...")

        for idx, (p, (w, h)) in enumerate(zip(scene_paths, dims)):
            abs_p = os.path.abspath(p)
            if (w, h) == (target_w, target_h):
                final_inputs.append(abs_p)
            else:
                norm_p = f"{final_output_path}.norm_{idx:03d}.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", abs_p,
                    "-vf", (
                        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                        f"crop={target_w}:{target_h},fps=30,format=yuv420p,setsar=1"
                    ),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-threads", "2",
                    "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
                    norm_p,
                ]
                run_command(cmd)
                temp_normalized.append(norm_p)
                final_inputs.append(os.path.abspath(norm_p))
                gc.collect()

    concat_list_path = f"{final_output_path}.concat.txt"
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in final_inputs:
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        final_output_path
    ]
    run_command(cmd)

    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    for tp in temp_normalized:
        if os.path.exists(tp):
            try:
                os.remove(tp)
            except Exception:
                pass

    logger.info(f"Final combined video assembled: {final_output_path}")
    return final_output_path
