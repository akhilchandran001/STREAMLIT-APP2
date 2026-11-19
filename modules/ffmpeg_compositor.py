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
    Renders any scene into a standardized 1080p 30fps H.264/AAC MP4.
    When both TTS audio and background SFX are present, sidechain ducking pushes
    the SFX down whenever TTS speaks and keeps TTS voice clearly louder on top.
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

    # Standardized stereo audio format so concat_all_scenes (-c copy) never hits channel mismatches
    a_std = "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo"
    temp_raw = f"{output_path}.raw.mp4"

    # SCENE TYPE 1: Unique Landscape Video (Scenes 1, 4...)
    if scene_type == "video":
        if not has_tts:
            raise ValueError("scene_type='video' requires a valid audio_path.")
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
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
            "-t", f"{duration:.3f}",
            output_path
        ]
        run_command(cmd)

    # SCENE TYPE 2: Animated Drop Stamp + Looping Background + Reference Typography
    elif scene_type == "background_with_image":
        if not has_tts:
            raise ValueError("scene_type='background_with_image' requires a valid audio_path.")
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
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
            "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
            "-t", f"{duration:.3f}",
            output_path
        ]
        run_command(cmd)
        if os.path.exists(temp_raw):
            os.remove(temp_raw)

    # SCENE TYPE 3: Procedural Kinetic Animation / Whiteboard / Newspaper
    elif scene_type == "animation":
        v_norm = (
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white,"
            f"fps=30,format=yuv420p,setsar=1,{v_fade}[v]"
        )

        if has_tts and has_sfx:
            # Keep TTS voice clearly higher (1.40x) and aggressively duck SFX (20:1)
            # whenever TTS is speaking so the two sounds do not clash/overlap.
            clamped_sfx = max(0.0, min(0.35, float(sfx_volume) * 0.50))
            filter_str = (
                f"[0:v]{v_norm};"
                f"[1:a]{a_std},volume=1.40,apad,asplit=2[tts_mix][tts_sc];"
                f"[2:a]{a_std},volume={clamped_sfx:.3f},apad[sfx_raw];"
                f"[sfx_raw][tts_sc]sidechaincompress=threshold=0.015:ratio=20:attack=5:release=200:makeup=1[sfx_ducked];"
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
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
                "-t", f"{duration:.3f}",
                output_path
            ]
        elif has_sfx and not has_tts:
            # SFX-Only Mode (No TTS voiceover)
            clamped_sfx = max(0.0, min(1.0, float(sfx_volume) if sfx_volume > 0 else 0.85))
            filter_str = (
                f"[0:v]{v_norm};"
                f"[1:a]{a_std},volume={clamped_sfx:.3f},alimiter=limit=0.97,{a_fade}[a]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", animation_video_path,
                "-i", sfx_audio_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
                "-t", f"{duration:.3f}",
                output_path
            ]
        elif has_tts:
            # Standard TTS-Only Animation
            filter_str = (
                f"[0:v]{v_norm};"
                f"[1:a]{a_std},volume=1.20,alimiter=limit=0.97,{a_fade}[a]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", animation_video_path,
                "-i", audio_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
                "-t", f"{duration:.3f}",
                output_path
            ]
        else:
            # Silent Animation fallback (generates silent stereo track so concat never breaks)
            filter_str = (
                f"[0:v]{v_norm};"
                f"anullsrc=channel_layout=stereo:sample_rate=44100,{a_fade}[a]"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", animation_video_path,
                "-filter_complex", filter_str,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-threads", "0",
                "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100",
                "-t", f"{duration:.3f}",
                output_path
            ]

        run_command(cmd)

    else:
        raise ValueError(f"Unsupported scene_type: '{scene_type}'")

    logger.info(f"Scene successfully rendered: {output_path} ({duration:.2f}s)")
    return output_path


def concat_all_scenes(scene_paths: list[str], final_output_path: str) -> str:
    """Concatenates all scene MP4s using FFmpeg concat demuxer without re-encoding."""
    if not scene_paths:
        raise ValueError("No rendered scene MP4s provided to concatenate.")

    os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)
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
