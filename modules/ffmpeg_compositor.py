import os
import subprocess
import sys
import logging
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


def calculate_fade_times(duration: float, fade_len: float = 0.4) -> tuple[float, float]:
    """Standardizes to a 0.4-second fade-out at the scene boundary."""
    actual_fade = min(fade_len, duration / 2.0)
    start_time = max(0.0, duration - actual_fade)
    return start_time, actual_fade


def render_individual_scene(
    scene_type: str,
    audio_path: str,
    output_path: str,
    video_path: str = None,
    background_video_path: str = None,
    foreground_image_path: str = None,
    animation_video_path: str = None,
    headline_data: dict = None
) -> str:
    """
    Renders any scene into a standardized 1080p 30fps H.264/AAC MP4.
    Uses multi-threaded veryfast encoding with optimal CRF for compact video size.
    """
    duration = get_stream_duration(audio_path)
    fade_start, fade_len = calculate_fade_times(duration, 0.4)
    v_fade = f"fade=t=out:st={fade_start:.3f}:d={fade_len:.3f}"
    a_fade = f"afade=t=out:st={fade_start:.3f}:d={fade_len:.3f}"

    temp_raw = f"{output_path}.raw.mp4"

    # SCENE TYPE 1: Unique Landscape Video (Scenes 1, 4...)
    if scene_type == "video":
        filter_str = (
            f"[0:v]scale=1920:1080:force_original_aspect_ratio=increase,"
            f"crop=1920:1080,fps=30,format=yuv420p,{v_fade}[v];"
            f"[1:a]{a_fade}[a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", video_path,
            "-i", audio_path,
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
            "-c:a", "aac", "-b:a", "128k",
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
        # Mux with audio and apply boundary fade
        filter_str = f"[0:v]{v_fade}[v];[1:a]{a_fade}[a]"
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_raw,
            "-i", audio_path,
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
            "-c:a", "aac", "-b:a", "128k",
            "-t", f"{duration:.3f}",
            output_path
        ]
        run_command(cmd)
        if os.path.exists(temp_raw):
            os.remove(temp_raw)

    # SCENE TYPE 3: Procedural OpenCV Kinetic Animation
    elif scene_type == "animation":
        filter_str = f"[0:v]{v_fade}[v];[1:a]{a_fade}[a]"
        cmd = [
            "ffmpeg", "-y",
            "-i", animation_video_path,
            "-i", audio_path,
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
            "-c:a", "aac", "-b:a", "128k",
            "-t", f"{duration:.3f}",
            output_path
        ]
        run_command(cmd)

    logger.info(f"Scene successfully rendered: {output_path} ({duration:.2f}s)")
    return output_path


def concat_all_scenes(scene_paths: list[str], final_output_path: str) -> str:
    """Concatenates all scene MP4s using FFmpeg concat demuxer without re-encoding."""
    concat_list_path = f"{final_output_path}.concat.txt"
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for p in scene_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        final_output_path
    ]
    run_command(cmd)

    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)

    logger.info(f"Final combined video assembled: {final_output_path}")
    return final_output_path
