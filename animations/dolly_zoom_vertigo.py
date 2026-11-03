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

# Vertigo Optical Subject Anchor[span_0](start_span)[span_0](end_span)
# Focal point coordinates (relative to frame center: upper-chest/face golden ratio)[span_1](start_span)[span_1](end_span)
FOCAL_OFFSET_X = 0.0
FOCAL_OFFSET_Y = -45.0
SUBJECT_RADIUS = 340.0   # Pixel radius of the locked foreground focal zone[span_2](start_span)[span_2](end_span)
# =========================================================


def prepare_base_canvas(img_path: str, target_w: int = WIDTH, target_h: int = HEIGHT) -> np.ndarray:
    """Loads image and fits it to 16:9 aspect ratio via center-crop & bicubic scale[span_3](start_span)[span_3](end_span)."""
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
    """Precomputes 2D radial falloff matrix for cinematic edge depth[span_4](start_span)[span_4](end_span)."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radial_dist = (xx**2) + ((yy * 1.25) ** 2)
    return np.clip(1.0 - 0.44 * radial_dist, 0.0, 1.0)[:, :, np.newaxis]


def evaluate_vertigo_timeline(t: float, duration_sec: float) -> tuple[float, float, float, float, float, int]:
    """
    Computes optical parameters for the Hitchcock Vertigo effect scaled dynamically
    to match the exact audio duration of the TTS sentence[span_5](start_span)[span_5](end_span):
    - bg_scale: Magnification of background (telephoto vs wide-angle perspective)[span_6](start_span)[span_6](end_span).
    - optical_distortion: Barrel (wide) vs Pincushion (telephoto) distortion factor[span_7](start_span)[span_7](end_span).
    - dutch_roll: Disorientation camera tilt angle (degrees)[span_8](start_span)[span_8](end_span).
    - dolly_track_jitter: High-frequency floor rail vibration[span_9](start_span)[span_9](end_span).
    - flash_val: Psychological white pulse intensity[span_10](start_span)[span_10](end_span).
    - chroma_disp: Radial chromatic dispersion[span_11](start_span)[span_11](end_span).
    """
    time_scale = 10.0 / max(duration_sec, 0.1)
    t_ref = t * time_scale

    bg_scale = 1.0
    distortion = 0.0
    dutch_roll = 0.0
    track_jitter = 0.0
    flash_val = 0.0
    chroma_disp = 0

    # =============================================================
    # PHASE 1: AMBIENT CINEMATIC INTRO (0.0% -> 15.0%)[span_12](start_span)[span_12](end_span)
    # =============================================================
    if t_ref < 1.5:
        u = t_ref / 1.5
        bg_scale = 1.0 + 0.03 * math.sin(math.pi * u)
        distortion = 0.0
        dutch_roll = 0.3 * math.sin(0.8 * t_ref)

    # =============================================================
    # PHASE 2: DOLLY OUT + ZOOM IN (15.0% -> 44.0%)[span_13](start_span)[span_13](end_span)
    # Foreground locked: Background rushes forward into telephoto compression[span_14](start_span)[span_14](end_span)
    # =============================================================
    elif t_ref < 4.4:
        dt = t_ref - 1.5
        dur = 2.9
        u = max(0.0, min(1.0, dt / dur))
        # Quintic smootherstep progression[span_15](start_span)[span_15](end_span)
        s = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)

        # Background looms from 1.0x to 2.25x[span_16](start_span)[span_16](end_span)
        bg_scale = 1.0 + (2.25 - 1.0) * s
        # Lens transitions from neutral to telephoto flat pincushion[span_17](start_span)[span_17](end_span)
        distortion = -0.18 * s
        # Subtle vertigo Dutch angle tilt[span_18](start_span)[span_18](end_span)
        dutch_roll = 3.6 * math.sin(math.pi * s)

        # Physical dolly rail rumble increases with speed[span_19](start_span)[span_19](end_span)
        s_prime = 30.0 * (u**2) * ((1.0 - u) ** 2)
        track_jitter = 4.5 * s_prime
        chroma_disp = int(18.0 * (s**1.6))

    # =============================================================
    # PHASE 3: PSYCHOLOGICAL VERTIGO HOLD (44.0% -> 56.0%)[span_20](start_span)[span_20](end_span)
    # Telephoto compression held with disorienting breathing pulses[span_21](start_span)[span_21](end_span)
    # =============================================================
    elif t_ref < 5.6:
        dt = t_ref - 4.4
        dur = 1.2
        u = max(0.0, min(1.0, dt / dur))

        # Heartbeat pulse on background magnification[span_22](start_span)[span_22](end_span)
        pulse = math.sin(2.0 * math.pi * 1.8 * dt) * math.exp(-1.5 * dt)
        bg_scale = 2.25 + 0.12 * pulse
        distortion = -0.18 + 0.04 * pulse
        dutch_roll = 3.6 * (1.0 - u) - 1.5 * u

        # Disorientation light pulse at trigger point[span_23](start_span)[span_23](end_span)
        flash_val = 145.0 * math.exp(-14.0 * dt)
        chroma_disp = int(22.0 * math.exp(-12.0 * dt))
        track_jitter = 1.2

    # =============================================================
    # PHASE 4: REVERSE DOLLY IN + ZOOM OUT (56.0% -> 82.0%)[span_24](start_span)[span_24](end_span)
    # Foreground locked: Background contracts into extreme wide-angle stretch[span_25](start_span)[span_25](end_span)
    # =============================================================
    elif t_ref < 8.2:
        dt = t_ref - 5.6
        dur = 2.6
        u = max(0.0, min(1.0, dt / dur))
        s = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)

        # Background recedes from 2.25x down to 0.62x (deep vortex stretch)[span_26](start_span)[span_26](end_span)
        bg_scale = 2.25 - (2.25 - 0.62) * s
        # Lens transitions to heavy wide-angle barrel distortion[span_27](start_span)[span_27](end_span)
        distortion = -0.18 + (0.32 - (-0.18)) * s
        # Reverse Dutch angle roll[span_28](start_span)[span_28](end_span)
        dutch_roll = -4.2 * math.sin(math.pi * s)

        s_prime = 30.0 * (u**2) * ((1.0 - u) ** 2)
        track_jitter = 5.2 * s_prime
        chroma_disp = int(24.0 * (s**1.5))

    # =============================================================
    # PHASE 5: ELASTIC DAMPED RECOIL & RESOLUTION (82.0% -> 100.0%)[span_29](start_span)[span_29](end_span)
    # Background snaps back to natural 1.0x perspective[span_30](start_span)[span_30](end_span)
    # =============================================================
    else:
        dt = t_ref - 8.2
        decay = math.exp(-5.5 * dt)
        spring = decay * math.cos(13.0 * dt)

        # Elastic harmonic return[span_31](start_span)[span_31](end_span)
        bg_scale = 1.0 - (1.0 - 0.62) * decay + 0.14 * spring
        distortion = 0.32 * decay * math.cos(10.0 * dt)
        dutch_roll = -3.2 * spring
        track_jitter = 3.0 * decay

        if dt < 0.35:
            flash_val = 120.0 * math.exp(-16.0 * dt)
            chroma_disp = int(14.0 * math.exp(-14.0 * dt))

    return bg_scale, distortion, dutch_roll, track_jitter, flash_val, chroma_disp


def generate_dolly_zoom_vertigo(
    input_path: str = None,
    output_path: str = None,
    duration_sec: float = DEFAULT_DURATION_SEC,
    fps: int = FPS
) -> str:
    """
    Renders the complete Hitchcock Dolly Zoom Vertigo Shift animation scaled dynamically
    to match the exact audio duration of the TTS sentence.
    """
    if input_path is None:
        input_path = os.path.join(DEFAULT_TEMP_DIR, "input.jpg")
    if output_path is None:
        output_path = os.path.join(DEFAULT_TEMP_DIR, "dolly_zoom_vertigo_shift.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Loading '{input_path}'...")
    base_image = prepare_base_canvas(input_path, WIDTH, HEIGHT)
    vignette = build_vignette_mask(WIDTH, HEIGHT)

    total_frames = max(1, int(round(fps * duration_sec)))

    print(f"Rendering Dolly Zoom Vertigo Shift ({WIDTH}x{HEIGHT}, {fps} FPS, {duration_sec:.2f}s, {total_frames} frames)...")
    print("Features: Dual-Plane Focal Pinning, Non-Linear Optical Remap, Barrel/Pincushion Breathing")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

    if not out.isOpened():
        raise RuntimeError(f"Error: Could not open VideoWriter at '{output_path}'.")

    cx = WIDTH * 0.5
    cy = HEIGHT * 0.5
    focal_cx = cx + FOCAL_OFFSET_X
    focal_cy = cy + FOCAL_OFFSET_Y

    # Precompute normalized spatial coordinate grids centered on the focal subject[span_32](start_span)[span_32](end_span)
    grid_y, grid_x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    dx = grid_x - focal_cx
    dy = grid_y - focal_cy
    dist_from_focal = np.sqrt(dx * dx + dy * dy)

    # Sigmoid / Gaussian radial transition mask (1.0 at subject center, 0.0 at outer canvas)[span_33](start_span)[span_33](end_span)
    # Preserves subject dimensions while warping peripheral background space[span_34](start_span)[span_34](end_span)
    subject_weight = np.exp(-0.5 * ((dist_from_focal / SUBJECT_RADIUS) ** 2.2)).astype(np.float32)
    bg_weight = (1.0 - subject_weight).astype(np.float32)

    # Normalized radial distance for optical barrel/pincushion distortion[span_35](start_span)[span_35](end_span)
    max_radius = math.sqrt(cx * cx + cy * cy)
    norm_r = (dist_from_focal / max_radius).astype(np.float32)

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # 1. Evaluate Vertigo Kinematics[span_36](start_span)[span_36](end_span)
        (
            bg_scale,
            distortion_k,
            dutch_roll,
            track_jitter,
            flash_intensity,
            chroma_shift,
        ) = evaluate_vertigo_timeline(t, duration_sec)

        # 2. Dolly Rail Mechanical Vibration[span_37](start_span)[span_37](end_span)
        jitter_x = random.uniform(-track_jitter, track_jitter) if track_jitter > 0.3 else 0.0
        jitter_y = random.uniform(-track_jitter, track_jitter) if track_jitter > 0.3 else 0.0

        # Continuous slow ambient camera drift[span_38](start_span)[span_38](end_span)
        ambient_drift_x = 10.0 * math.sin(0.6 * t)
        ambient_drift_y = 6.0 * math.cos(0.5 * t)

        # 3. Dual-Plane Scale Synthesis (Subject scale locked at 1.0, Background scales dynamically)[span_39](start_span)[span_39](end_span)
        local_scale = (subject_weight * 1.0 + bg_weight * bg_scale).astype(np.float32)
        inv_scale = 1.0 / np.maximum(local_scale, 0.25)

        # 4. Non-Linear Optical Distortion (Barrel / Pincushion Breathing)[span_40](start_span)[span_40](end_span)
        distort_factor = (1.0 + distortion_k * (norm_r**2)).astype(np.float32)

        # Compute source sampling coordinates relative to focal center[span_41](start_span)[span_41](end_span)
        sample_dx = dx * inv_scale * distort_factor
        sample_dy = dy * inv_scale * distort_factor

        # 5. Apply Rotational Dutch Angle & Translational Offsets[span_42](start_span)[span_42](end_span)
        rad_roll = math.radians(dutch_roll)
        cos_r = math.cos(rad_roll)
        sin_r = math.sin(rad_roll)

        rot_dx = sample_dx * cos_r - sample_dy * sin_r
        rot_dy = sample_dx * sin_r + sample_dy * cos_r

        map_x = (focal_cx + rot_dx + ambient_drift_x + jitter_x).astype(np.float32)
        map_y = (focal_cy + rot_dy + ambient_drift_y + jitter_y).astype(np.float32)

        # 6. Execute Sub-Pixel Optical Remap[span_43](start_span)[span_43](end_span)
        warped = cv2.remap(
            base_image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # 7. Radial Chromatic Aberration Dispersion on High-G Vertigo Shifts[span_44](start_span)[span_44](end_span)
        if chroma_shift > 1:
            scale_r = 1.0 + (chroma_shift / 1920.0) * 1.6
            scale_b = 1.0 - (chroma_shift / 1920.0) * 1.6

            M_r = cv2.getRotationMatrix2D((focal_cx, focal_cy), 0.0, scale_r)
            M_b = cv2.getRotationMatrix2D((focal_cx, focal_cy), 0.0, scale_b)

            b_ch, g_ch, r_ch = cv2.split(warped)
            r_warped = cv2.warpAffine(r_ch, M_r, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            b_warped = cv2.warpAffine(b_ch, M_b, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            warped = cv2.merge([b_warped, g_ch, r_warped])

        # 8. Psychological White Exposure Flash[span_45](start_span)[span_45](end_span)
        if flash_intensity > 1.0:
            f_val = int(min(flash_intensity, 255.0))
            warped = cv2.add(warped, np.full_like(warped, f_val))

        # 9. Cinematic Contrast S-Curve & Lens Vignette[span_46](start_span)[span_46](end_span)
        final_float = (warped.astype(np.float32) / 255.0) * vignette

        # Deepen shadows to enhance spatial vertigo depth[span_47](start_span)[span_47](end_span)
        final_float = np.where(
            final_float < 0.5,
            2.0 * (final_float**1.22),
            1.0 - 2.0 * ((1.0 - final_float) ** 1.22) * 0.5,
        )
        final_frame = np.clip(final_float * 255.0, 0, 255).astype(np.uint8)

        out.write(final_frame)

        if (frame_idx + 1) % (fps * 2) == 0 or frame_idx == total_frames - 1:
            print(f"Rendered {frame_idx + 1}/{total_frames} frames...")

    out.release()
    print(f"Done! Video successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    test_in = sys.argv[1] if len(sys.argv) > 1 else "1.jpg"
    test_out = sys.argv[2] if len(sys.argv) > 2 else "dolly_zoom_vertigo_shift.mp4"
    test_dur = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DURATION_SEC

    generate_dolly_zoom_vertigo(test_in, test_out, duration_sec=test_dur)
