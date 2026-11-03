import math
import os
import sys
import cv2
import numpy as np

# ================= CONFIGURATION & PATHS =================
# Default paths adapted for Streamlit Cloud runtime
BASE_PROJECT_DIR = os.getcwd()
DEFAULT_TEMP_DIR = os.path.join(BASE_PROJECT_DIR, "storage", "temp")

# Video Settings (16:9 Landscape Widescreen)
WIDTH = 1920
HEIGHT = 1080
FPS = 30
DEFAULT_DURATION_SEC = 10.0

# Normalized Velocity Ramp Segments (Relative 0.0 -> 1.0 progression)
# Each segment: (norm_start, norm_end, start_zoom, end_zoom, (start_x, start_y), (end_x, end_y), start_tilt, end_tilt)
NORMALIZED_RAMP_SEGMENTS = [
    # 0% - 20%: Aggressive punch-in on center -> slow hypnotic glide
    (0.0, 0.20, 1.06, 1.48, (0.0, 0.0), (45.0, -25.0), 0.0, 2.0),
    # 20% - 40%: Reverse snap zoom-out with left whip -> slow drift right
    (0.20, 0.40, 1.48, 1.14, (45.0, -25.0), (-55.0, 30.0), 2.0, -2.5),
    # 40% - 60%: Diagonal rush into upper-right quadrant -> ultra-slow crawl
    (0.40, 0.60, 1.14, 1.68, (-55.0, 30.0), (60.0, 35.0), -2.5, 3.0),
    # 60% - 80%: S-curve velocity transition down-left -> slow panoramic pan
    (0.60, 0.80, 1.68, 1.22, (60.0, 35.0), (-35.0, -40.0), 3.0, -2.0),
    # 80% - 100%: Climax double-speed rush into macro crop
    (0.80, 1.00, 1.22, 1.88, (-35.0, -40.0), (0.0, 0.0), -2.0, 0.0),
]
# =========================================================


def prepare_base_canvas(img_path: str, target_w: int = WIDTH, target_h: int = HEIGHT) -> np.ndarray:
    """Loads image and fits it to 16:9 aspect ratio via center-crop & bicubic scale."""
    src = cv2.imread(img_path)
    if src is None:
        raise FileNotFoundError(f"Could not load input image at '{img_path}'")

    sh, sw = src.shape[:2]
    target_aspect = target_w / target_h
    src_aspect = sw / sh

    if src_aspect > target_aspect:
        new_w = int(sh * target_aspect)
        offset_x = (sw - new_w) // 2
        cropped = src[:, offset_x : offset_x + new_w]
    else:
        new_h = int(sw / target_aspect)
        offset_y = (sh - new_h) // 2
        cropped = src[offset_y : offset_y + new_h, :]

    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def build_vignette_mask(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Precomputes 2D radial falloff matrix for cinematic edge depth."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radial_dist = (xx**2) + ((yy * 1.25) ** 2)
    return np.clip(1.0 - 0.45 * radial_dist, 0.0, 1.0)[:, :, np.newaxis]


def evaluate_velocity_curve(u: float) -> tuple[float, float]:
    """
    Classic CapCut Velocity Curve (Fast Rush -> Sudden Brake/Slow-Mo -> Exit Snap):
    - 0.00 <= u < 0.22: High-speed acceleration rush (covers 68% of movement)
    - 0.22 <= u < 0.82: Ultra-slow motion glide (creeps through 18% of movement)
    - 0.82 <= u <= 1.00: Snappy exit ramp into the next beat (covers remaining 14%)
    Returns: (progress_value, normalized_velocity)
    """
    u_clamped = float(np.clip(u, 0.0, 1.0))

    if u_clamped < 0.22:
        s = u_clamped / 0.22
        prog = 0.68 * (s**1.6)
        vel = (0.68 * 1.6 * (max(s, 1e-4) ** 0.6)) / 0.22
    elif u_clamped < 0.82:
        s = (u_clamped - 0.22) / 0.60
        prog = 0.68 + 0.18 * (s**0.85)
        vel = (0.18 * 0.85 * (max(s, 1e-4) ** -0.15)) / 0.60
    else:
        s = (u_clamped - 0.82) / 0.18
        prog = 0.86 + 0.14 * (s**1.7)
        vel = (0.14 * 1.7 * (max(s, 1e-4) ** 0.7)) / 0.18

    norm_vel = float(np.clip((vel - 0.6) / 3.8, 0.0, 1.0))
    return float(prog), norm_vel


def apply_directional_motion_blur(
    img: np.ndarray, vec_x: float, vec_y: float, blur_mag: float
) -> np.ndarray:
    """Applies a linear 1D directional blur kernel aligned with the camera velocity vector."""
    ksize = int(blur_mag * 22.0)
    if ksize < 3:
        return img
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2

    angle = math.atan2(vec_y, vec_x + 1e-5)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    for i in range(-center, center + 1):
        kx = int(round(center + i * cos_a))
        ky = int(round(center + i * sin_a))
        if 0 <= kx < ksize and 0 <= ky < ksize:
            kernel[ky, kx] = 1.0

    k_sum = np.sum(kernel)
    if k_sum > 0:
        kernel /= k_sum
        return cv2.filter2D(img, -1, kernel)
    return img


def apply_zoom_motion_blur(
    img: np.ndarray, cx: float, cy: float, current_zoom: float, blur_mag: float
) -> np.ndarray:
    """Blends sub-scaled optical samples to simulate cinematic radial zoom blur."""
    if blur_mag < 0.08:
        return img

    delta_scale = blur_mag * 0.045
    m1 = cv2.getRotationMatrix2D((cx, cy), 0, 1.0 + delta_scale)
    m2 = cv2.getRotationMatrix2D((cx, cy), 0, 1.0 - delta_scale)

    s1 = cv2.warpAffine(
        img,
        m1,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    s2 = cv2.warpAffine(
        img,
        m2,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    weight = float(np.clip(blur_mag * 0.45, 0.0, 0.6))
    return cv2.addWeighted(img, 1.0 - weight, cv2.addWeighted(s1, 0.5, s2, 0.5, 0), weight, 0)


def generate_velocity_speed_ramp(
    input_path: str = None,
    output_path: str = None,
    duration_sec: float = DEFAULT_DURATION_SEC,
    fps: int = FPS
) -> str:
    """
    Renders the complete Velocity Speed Ramp animation scaled dynamically
    to match the exact audio duration of the TTS sentence.
    """
    if input_path is None:
        input_path = os.path.join(DEFAULT_TEMP_DIR, "input.jpg")
    if output_path is None:
        output_path = os.path.join(DEFAULT_TEMP_DIR, "velocity_speed_ramp.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Loading '{input_path}'...")
    base_image = prepare_base_canvas(input_path, WIDTH, HEIGHT)
    vignette = build_vignette_mask(WIDTH, HEIGHT)

    total_frames = max(1, int(round(fps * duration_sec)))

    print(
        f"Rendering Velocity Speed Ramp ({WIDTH}x{HEIGHT}, {fps} FPS, {duration_sec:.2f}s, {total_frames} frames)..."
    )
    print("Features: Non-linear CapCut Curves, Directional & Zoom Blurs, Beat Flash, RGB Warp")

    # Re-scale normalized segments to match the target sentence duration in seconds
    active_segments = [
        (
            s[0] * duration_sec,
            s[1] * duration_sec,
            s[2],
            s[3],
            s[4],
            s[5],
            s[6],
            s[7]
        )
        for s in NORMALIZED_RAMP_SEGMENTS
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

    if not out.isOpened():
        raise RuntimeError(f"Error: Could not open VideoWriter at '{output_path}'.")

    cx = WIDTH * 0.5
    cy = HEIGHT * 0.5

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # 1. Locate Active Ramp Segment
        seg = active_segments[-1]
        for s in active_segments:
            if s[0] <= t < s[1]:
                seg = s
                break

        t_start, t_end, z_start, z_end, (x_start, y_start), (x_end, y_end), roll_start, roll_end = seg
        seg_duration = max(t_end - t_start, 1e-4)
        u = (t - t_start) / seg_duration

        # 2. Evaluate Non-Linear Speed Ramp Profile
        progress, norm_velocity = evaluate_velocity_curve(u)

        # Interpolate camera transformation parameters
        current_zoom = z_start + (z_end - z_start) * progress
        current_x = x_start + (x_end - x_start) * progress
        current_y = y_start + (y_end - y_start) * progress
        current_roll = roll_start + (roll_end - roll_start) * progress

        # 3. Construct Affine Transformation Matrix
        M = cv2.getRotationMatrix2D((cx, cy), current_roll, current_zoom)
        M[0, 2] += current_x
        M[1, 2] += current_y

        warped = cv2.warpAffine(
            base_image,
            M,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # 4. Velocity-Dependent Motion Blurs (Zoom Blur + Directional Pan Blur)
        if norm_velocity > 0.05:
            # Linear directional streak aligned with current motion vector
            vec_x = (x_end - x_start)
            vec_y = (y_end - y_start)
            warped = apply_directional_motion_blur(warped, vec_x, vec_y, norm_velocity)

            # Radial optical zoom blur
            warped = apply_zoom_motion_blur(warped, cx, cy, current_zoom, norm_velocity)

        # 5. Chromatic Aberration / RGB Split during high-velocity snaps
        chroma_shift = int(norm_velocity * 12.0)
        if chroma_shift > 1:
            b, g, r = cv2.split(warped)
            r = np.roll(r, chroma_shift, axis=1)
            b = np.roll(b, -chroma_shift, axis=1)
            warped = cv2.merge([b, g, r])

        # 6. Sudden Impact Beat Flash
        # The beat drops right at u = 0.22 (where the rush decelerates into slow-motion)
        t_impact = t_start + 0.22 * seg_duration
        dt_impact = t - t_impact
        if dt_impact >= 0.0:
            flash_intensity = 110.0 * math.exp(-22.0 * dt_impact)
            if flash_intensity > 1.0:
                warped = cv2.add(warped, np.full_like(warped, int(flash_intensity)))

        # 7. Cinematic Depth Vignette
        final_float = (warped.astype(np.float32) / 255.0) * vignette
        final_frame = np.clip(final_float * 255.0, 0, 255).astype(np.uint8)

        out.write(final_frame)

        if (frame_idx + 1) % (fps * 2) == 0 or frame_idx == total_frames - 1:
            print(f"Rendered {frame_idx + 1}/{total_frames} frames...")

    out.release()
    print(f"Done! Video successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    test_in = sys.argv[1] if len(sys.argv) > 1 else "1.jpg"
    test_out = sys.argv[2] if len(sys.argv) > 2 else "velocity_speed_ramp.mp4"
    test_dur = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DURATION_SEC

    generate_velocity_speed_ramp(test_in, test_out, duration_sec=test_dur)
