import json
import os
import subprocess
import sys

# Standard 16:9 Landscape dimensions
TARGET_W = 1920
TARGET_H = 1080
FPS = 30
DEFAULT_MARGIN = 90  # Equal pixel margin for top, bottom, left, and right


def run_command(cmd: list[str]) -> None:
    """Executes FFmpeg/ffprobe CLI commands and captures detailed error tracebacks."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg execution error:\nCommand: {' '.join(cmd)}\nStderr:\n{proc.stderr}")


def get_stream_duration(file_path: str) -> float:
    """Uses ffprobe to extract float-precision duration in seconds directly from container metadata."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Media file not found: {file_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {file_path}: {proc.stderr}")

    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def calculate_fade_times(duration_sec: float) -> tuple[float, float]:
    """
    Computes fade-out start time and duration.
    Standardizes to a 0.4s fade-out, scaling proportionally down if duration is ultra-short.
    """
    fade_dur = 0.4 if duration_sec >= 0.8 else round(duration_sec * 0.4, 3)
    fade_start = max(0.0, round(duration_sec - fade_dur, 3))
    return fade_start, fade_dur


# =========================================================================
# UNIFIED INDIVIDUAL SCENE GENERATOR
# =========================================================================
def render_individual_scene(
    scene_type: str,
    visual_path: str,
    audio_path: str,
    duration_sec: float,
    output_path: str,
    bg_video_path: str = None,
    margin_px: int = DEFAULT_MARGIN
) -> str:
    """
    Renders any scene type into an identically encoded intermediate MP4:
    - 'standalone': Fits unique landscape video to 1920x1080, looped or trimmed to audio duration.
    - 'bg_image': Loops/trims background video, auto-scales image with equal 4-side margins, and centers it.
    - 'animation': Takes the OpenCV rendered MP4, conforms FPS/scale, and muxes TTS audio.
    Every scene receives an exact 0.4s tail fade-out on video and audio.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fade_start, fade_dur = calculate_fade_times(duration_sec)

    # Standardized output encoding flags for zero-loss demux concatenation
    encoding_flags = [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-t", f"{duration_sec:.3f}"
    ]

    # Audio standardization filter: forces stereo 44.1kHz with smooth fade-out
    audio_filter = f"aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}"

    if scene_type == "standalone":
        # SENTENCE 1: Pure landscape video from Pixabay
        video_filter = (
            f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_W}:{TARGET_H},"
            f"fps={FPS},"
            f"fade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", visual_path,
            "-i", audio_path,
            "-filter_complex", f"[0:v]{video_filter}[v];[1:a]{audio_filter}[a]",
            "-map", "[v]",
            "-map", "[a]",
            *encoding_flags,
            output_path
        ]

    elif scene_type == "bg_image":
        # SENTENCE 2, 4, 6...: Background Video (BG2.mp4, etc.) + Centered Scaled Image
        if not bg_video_path or not os.path.exists(bg_video_path):
            raise FileNotFoundError(f"Background video not found at: {bg_video_path}")

        inner_w = TARGET_W - (margin_px * 2)
        inner_h = TARGET_H - (margin_px * 2)

        # 1. Scale background video to fill 1920x1080.
        # 2. Scale image proportionally inside inner bounding box (preserves original aspect ratio).
        # 3. Center image at (W-w)/2:(H-h)/2, guaranteeing equal margins on left/right and top/bottom.
        # 4. Apply fade out.
        filter_complex = (
            f"[0:v]scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,crop={TARGET_W}:{TARGET_H},fps={FPS}[bg];"
            f"[1:v]scale={inner_w}:{inner_h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2:shortest=1,"
            f"fade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}[v];"
            f"[2:a]{audio_filter}[a]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", bg_video_path,
            "-loop", "1",
            "-i", visual_path,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            *encoding_flags,
            output_path
        ]

    elif scene_type == "animation":
        # SENTENCE 3, 5, 7...: Procedural OpenCV Kinetic Animation Video
        video_filter = (
            f"scale={TARGET_W}:{TARGET_H},"
            f"fps={FPS},"
            f"fade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", visual_path,
            "-i", audio_path,
            "-filter_complex", f"[0:v]{video_filter}[v];[1:a]{audio_filter}[a]",
            "-map", "[v]",
            "-map", "[a]",
            *encoding_flags,
            output_path
        ]

    else:
        raise ValueError(f"Unknown scene_type: '{scene_type}'. Expected 'standalone', 'bg_image', or 'animation'.")

    run_command(cmd)
    print(f"[FFmpeg Engine] Scene rendered successfully: {output_path} ({duration_sec:.2f}s)")
    return output_path


# =========================================================================
# ZERO-LOSS FINAL CONCATENATION DEMUXER
# =========================================================================
def concat_all_scenes(
    scene_video_paths: list[str],
    output_master_path: str,
    temp_dir: str
) -> str:
    """
    Combines all individually rendered scene MP4s using the FFmpeg concat demuxer.
    Because every scene is standardized with identical codecs, timebases, pixel formats,
    and audio profiles, this command executes instantly via stream copy (-c copy)
    with zero loss of quality, zero audio desync, and minimum CPU consumption.
    """
    if not scene_video_paths:
        raise ValueError("Cannot concatenate an empty list of scene videos.")

    os.makedirs(os.path.dirname(os.path.abspath(output_master_path)), exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    concat_manifest = os.path.join(temp_dir, "concat_manifest.txt")
    with open(concat_manifest, "w", encoding="utf-8") as f:
        for p in scene_video_paths:
            # Escape single quotes in filenames for the demuxer protocol
            safe_path = os.path.abspath(p).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_manifest,
        "-c", "copy",
        "-movflags", "+faststart",
        output_master_path
    ]

    print(f"\n[FFmpeg Engine] Merging {len(scene_video_paths)} scenes into master output...")
    run_command(cmd)

    total_duration = get_stream_duration(output_master_path)
    file_size_mb = os.path.getsize(output_master_path) / (1024 * 1024)
    print(f"[FFmpeg Engine] Master video assembly complete: {output_master_path}")
    print(f"Total Master Duration: {total_duration:.2f}s | File Size: {file_size_mb:.2f} MB\n")

    return output_master_path


if __name__ == "__main__":
    print("FFmpeg Compositor module initialized.")
