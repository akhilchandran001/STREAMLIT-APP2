import math
import os
import random
import sys
import cv2
import numpy as np

# ================= CONFIGURATION & PATHS =================
# Adapted for Streamlit Cloud / headless container environment
BASE_PROJECT_DIR = os.getcwd()
DEFAULT_TEMP_DIR = os.path.join(BASE_PROJECT_DIR, "storage", "temp")

# Video Settings (16:9 Landscape Widescreen)
WIDTH = 1920
HEIGHT = 1080
FPS = 30
DEFAULT_DURATION_SEC = 10.0

# Inertial Segments across 10-Second Baseline Reference Timeline[span_0](start_span)[span_0](end_span)
# Structure: (t_start, t_brake, t_end, (delta_x, delta_y), delta_zoom, delta_roll, brake_power)[span_1](start_span)[span_1](end_span)
# Each segment builds momentum until t_brake, where it violently brakes with spring recoil[span_2](start_span)[span_2](end_span)
INERTIAL_SEGMENTS = [
    # Segment 1: Right-downward floating momentum cruise -> Sharp brake[span_3](start_span)[span_3](end_span)
    (0.0, 2.10, 2.80, (140.0, -75.0), 0.22, 1.8, 1.15),
    # Segment 2: Heavy momentum redirection left-upward -> Violent bass drop brake[span_4](start_span)[span_4](end_span)
    (2.80, 5.10, 5.80, (-210.0, 120.0), 0.26, -3.2, 1.45),
    # Segment 3: High-velocity diagonal surge -> Climax snap brake[span_5](start_span)[span_5](end_span)
    (5.80, 7.80, 8.50, (180.0, 95.0), 0.24, 2.5, 1.50),
    # Segment 4: Inertial drift settling into final cinematic macro lock[span_6](start_span)[span_6](end_span)
    (8.50, 9.80, 10.00, (-40.0, -25.0), 0.08, -0.6, 0.70),
]
# =========================================================


def prepare_base_canvas(img_path: str, target_w: int = WIDTH, target_h: int = HEIGHT) -> np.ndarray:
    """Loads image and fits it to 16:9 aspect ratio via center-crop & bicubic scale[span_7](start_span)[span_7](end_span)."""
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
    """Precomputes 2D radial falloff matrix for cinematic edge depth[span_8](start_span)[span_8](end_span)."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radial_dist = (xx**2) + ((yy * 1.25) ** 2)
    return np.clip(1.0 - 0.42 * radial_dist, 0.0, 1.0)[:, :, np.newaxis]


def evaluate_inertial_motion(t: float, duration_sec: float) -> tuple[float, float, float, float, float, float, float, float, int]:
    """
    Computes cumulative camera translation (X, Y), zoom, roll angle,
    instantaneous travel velocity, and brake impact shock parameters dynamically
    scaled to the exact TTS audio duration[span_9](start_span)[span_9](end_span).
    """
    time_scale = 10.0 / max(duration_sec, 0.1)
    t_ref = t * time_scale

    cur_x = 0.0
    cur_y = 0.0
    cur_zoom = 1.05
    cur_roll = 0.0

    inst_vx = 0.0
    inst_vy = 0.0
    flash_val = 0.0
    shake_mag = 0.0
    chroma_split = 0

    for t_start, t_brake, t_end, (dx, dy), d_zoom, d_roll, power in INERTIAL_SEGMENTS:
        if t_ref < t_start:
            break

        drift_dur = t_brake - t_start

        if t_ref < t_brake:
            # PHASE 1: Inertial momentum cruise (Accelerating glide)[span_10](start_span)[span_10](end_span)
            u = (t_ref - t_start) / max(drift_dur, 1e-4)
            # Cubic ease-in momentum: velocity builds smoothly[span_11](start_span)[span_11](end_span)
            prog = u**1.65
            vel_factor = (1.65 * (max(u, 1e-4) ** 0.65)) / max(drift_dur, 1e-4)

            cur_x += dx * prog
            cur_y += dy * prog
            cur_zoom += d_zoom * prog
            cur_roll += d_roll * prog

            inst_vx = dx * vel_factor
            inst_vy = dy * vel_factor
            break

        else:
            # PHASE 2: Snap Brake & Damped Spring Recoil[span_12](start_span)[span_12](end_span)
            dt = t_ref - t_brake
            recoil_dur = t_end - t_brake

            # Damped harmonic oscillation modeling physical momentum overshoot[span_13](start_span)[span_13](end_span):
            # Camera shoots forward due to inertia, then snaps back elastically[span_14](start_span)[span_14](end_span)
            damping = 8.5
            omega = 18.0
            decay = math.exp(-damping * dt)
            spring_overshoot = 0.28 * decay * math.sin(omega * dt)

            # Rebound factor reaches 1.0 at rest, with a strong forward whip at dt=0+[span_15](start_span)[span_15](end_span)
            prog = 1.0 + spring_overshoot

            cur_x += dx * prog
            cur_y += dy * prog
            cur_zoom += d_zoom * prog
            cur_roll += d_roll * prog

            # Velocity during braking recoil[span_16](start_span)[span_16](end_span)
            recoil_vel = 0.28 * decay * (omega * math.cos(omega * dt) - damping * math.sin(omega * dt))
            inst_vx = dx * recoil_vel
            inst_vy = dy * recoil_vel

            # Brake impact shockwaves (Exposure bloom, seismic rumble, RGB lateral shear)[span_17](start_span)[span_17](end_span)
            if dt < 0.35:
                env = math.exp(-18.0 * dt)
                flash_val = max(flash_val, 175.0 * power * env)
                shake_mag = max(shake_mag, 16.0 * power * env)
                chroma_split = max(chroma_split, int(18.0 * power * env))

            if t_ref <= t_end:
                break

    return cur_x, cur_y, cur_zoom, cur_roll, inst_vx, inst_vy, flash_val, shake_mag, chroma_split


def apply_directional_motion_blur(img: np.ndarray, vx: float, vy: float, speed_norm: float) -> np.ndarray:
    """Applies a 1D directional blur kernel aligned with the instantaneous momentum vector[span_18](start_span)[span_18](end_span)."""
    ksize = int(speed_norm * 24.0)
    if ksize < 3:
        return img
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2

    angle = math.atan2(vy, vx + 1e-5)
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


def generate_inertial_drift_snap_brake(
    input_path: str = None,
    output_path: str = None,
    duration_sec: float = DEFAULT_DURATION_SEC,
    fps: int = FPS
) -> str:
    """
    Renders the complete Inertial Drift & Snap Brake animation scaled dynamically
    to match the exact audio duration of the TTS sentence[span_19](start_span)[span_19](end_span).
    """
    if input_path is None:
        input_path = os.path.join(DEFAULT_TEMP_DIR, "input.jpg")
    if output_path is None:
        output_path = os.path.join(DEFAULT_TEMP_DIR, "inertial_drift_snap_brake.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Loading '{input_path}'...")
    base_image = prepare_base_canvas(input_path, WIDTH, HEIGHT)
    vignette = build_vignette_mask(WIDTH, HEIGHT)

    total_frames = max(1, int(round(fps * duration_sec)))

    print(f"Rendering Inertial Drift & Snap Brake ({WIDTH}x{HEIGHT}, {fps} FPS, {duration_sec:.2f}s, {total_frames} frames)...")
    print("Features: Momentum Vector Acceleration, Damped Harmonic Recoil, Impulse Deceleration Shock")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

    if not out.isOpened():
        raise RuntimeError(f"Error: Could not open VideoWriter at '{output_path}'.")

    cx = WIDTH * 0.5
    cy = HEIGHT * 0.5

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # 1. Evaluate Dynamic Kinematics & Brake Shock Physics[span_20](start_span)[span_20](end_span)
        (
            cam_x,
            cam_y,
            cam_zoom,
            cam_roll,
            vx,
            vy,
            flash_intensity,
            shake_intensity,
            chroma_shift,
        ) = evaluate_inertial_motion(t, duration_sec)

        # 2. Seismic Camera Jitter on Brake Impact[span_21](start_span)[span_21](end_span)
        if shake_intensity > 0.5:
            jitter_x = random.uniform(-shake_intensity, shake_intensity)
            jitter_y = random.uniform(-shake_intensity, shake_intensity)
            jitter_roll = random.uniform(-0.6, 0.6) * (shake_intensity / 16.0)
        else:
            jitter_x, jitter_y, jitter_roll = 0.0, 0.0, 0.0

        # 3. Construct Affine Transformation Matrix[span_22](start_span)[span_22](end_span)
        total_zoom = max(cam_zoom, 1.0)
        total_roll = cam_roll + jitter_roll

        M = cv2.getRotationMatrix2D((cx, cy), total_roll, total_zoom)
        M[0, 2] += cam_x + jitter_x
        M[1, 2] += cam_y + jitter_y

        warped = cv2.warpAffine(
            base_image,
            M,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # 4. Instantaneous Directional Motion Blur (High-momentum streaks & recoil smearing)[span_23](start_span)[span_23](end_span)
        speed = math.sqrt(vx * vx + vy * vy)
        speed_norm = float(np.clip(speed / 480.0, 0.0, 1.0))
        if speed_norm > 0.08:
            warped = apply_directional_motion_blur(warped, vx, vy, speed_norm)

        # 5. Chromatic Aberration Shear on Sudden Deceleration[span_24](start_span)[span_24](end_span)
        if chroma_shift > 1:
            # Channel split along inertial trajectory[span_25](start_span)[span_25](end_span)
            angle = math.atan2(vy, vx + 1e-5)
            shift_x = int(round(chroma_shift * math.cos(angle)))
            shift_y = int(round(chroma_shift * math.sin(angle)))

            b, g, r = cv2.split(warped)
            r = np.roll(np.roll(r, shift_x, axis=1), shift_y, axis=0)
            b = np.roll(np.roll(b, -shift_x, axis=1), -shift_y, axis=0)
            warped = cv2.merge([b, g, r])

        # 6. Additive White Exposure Bloom on Snap Brake Hits[span_26](start_span)[span_26](end_span)
        if flash_intensity > 1.0:
            f_val = int(min(flash_intensity, 255.0))
            warped = cv2.add(warped, np.full_like(warped, f_val))

        # 7. Cinematic Vignette Pass[span_27](start_span)[span_27](end_span)
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
    test_out = sys.argv[2] if len(sys.argv) > 2 else "inertial_drift_snap_brake.mp4"
    test_dur = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DURATION_SEC

    generate_inertial_drift_snap_brake(test_in, test_out, duration_sec=test_dur)
