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

# Tilt-Shift Miniature Optical Parameters[span_0](start_span)[span_0](end_span)
BASE_BAND_HALF_HEIGHT = 110.0   # Half-width of razor-sharp focal band (pixels)[span_1](start_span)[span_1](end_span)
FALLOFF_DISTANCE = 260.0        # Feathered transition width from sharp to deep blur[span_2](start_span)[span_2](end_span)
MAX_BOKEH_RADIUS = 32           # Blur radius for out-of-focus toy diorama depth[span_3](start_span)[span_3](end_span)
SATURATION_BOOST = 1.48         # Hyper-saturated toy paint palette factor[span_4](start_span)[span_4](end_span)
CONTRAST_GAMMA = 1.25           # Steep contrast curve for miniature plastic sheen[span_5](start_span)[span_5](end_span)
SPECULAR_CUTOFF = 205           # Luminance threshold for out-of-focus bokeh highlights[span_6](start_span)[span_6](end_span)
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
    """Precomputes 2D radial falloff matrix for diorama lens shading[span_8](start_span)[span_8](end_span)."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radial_dist = (xx**2) + ((yy * 1.25) ** 2)
    return np.clip(1.0 - 0.45 * radial_dist, 0.0, 1.0)[:, :, np.newaxis]


def apply_diorama_color_grade(img: np.ndarray) -> np.ndarray:
    """
    Applies high-vibrancy miniature grading:
    Boosts HSV saturation, pushes warm midtones, and tightens micro-contrast
    to make full-scale scenes resemble tiny hand-painted plastic models[span_9](start_span)[span_9](end_span).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Boost saturation channel[span_10](start_span)[span_10](end_span)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * SATURATION_BOOST, 0, 255)
    graded_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Steep S-curve contrast (crushed shadows, gleaming highlights)[span_11](start_span)[span_11](end_span)
    graded_bgr = np.where(
        graded_bgr < 0.5,
        2.0 * (graded_bgr**CONTRAST_GAMMA),
        1.0 - 2.0 * ((1.0 - graded_bgr) ** CONTRAST_GAMMA) * 0.5,
    )

    # Subtle warm sunlight tint on highlights[span_12](start_span)[span_12](end_span)
    graded_bgr[:, :, 2] = np.clip(graded_bgr[:, :, 2] * 1.04, 0.0, 1.0)  # Warm red[span_13](start_span)[span_13](end_span)
    graded_bgr[:, :, 0] = np.clip(graded_bgr[:, :, 0] * 0.96, 0.0, 1.0)  # Reduced blue[span_14](start_span)[span_14](end_span)

    return np.clip(graded_bgr * 255.0, 0, 255).astype(np.uint8)


def evaluate_miniature_timeline(t: float, duration_sec: float) -> tuple[float, float, float, float, float, float, float, float, int]:
    """
    Computes camera position, isometric high-angle tilt, focal plane line parameters,
    shutter flash, and stop-motion frame quantization across 5 distinct phases dynamically
    scaled to the exact TTS audio duration[span_15](start_span)[span_15](end_span):
    1. Static Miniature Intro: Horizontal focal band locked on lower-third subject[span_16](start_span)[span_16](end_span).
    2. Angled Focal Scan: Focal band rotates 14° diagonally and glides upward[span_17](start_span)[span_17](end_span).
    3. High-Angle Isometric Crane Track: Elevated skew glide with diorama step cadence[span_18](start_span)[span_18](end_span).
    4. Macro Detail Punch: Rapid crash zoom into miniature focal center + exposure pop[span_19](start_span)[span_19](end_span).
    5. Cinematic Diorama Settle: Smooth pull-out to master model framing[span_20](start_span)[span_20](end_span).
    """
    # Scale factor mapping the dynamic duration to the 10.0-second baseline trajectory[span_21](start_span)[span_21](end_span)
    time_scale = 10.0 / max(duration_sec, 0.1)
    t_ref = t * time_scale

    cam_x = 0.0
    cam_y = 0.0
    cam_zoom = 1.05
    iso_shear = 0.0
    focal_y = 620.0
    tilt_angle_deg = 0.0
    flash_val = 0.0
    shake_mag = 0.0
    chroma_disp = 0

    # =============================================================
    # PHASE 1: AMBIENT DIORAMA INTRO (0.0% -> 15.0%)[span_22](start_span)[span_22](end_span)
    # =============================================================
    if t_ref < 1.5:
        u = t_ref / 1.5
        cam_zoom = 1.05 + 0.02 * math.sin(math.pi * u)
        focal_y = 620.0 + 15.0 * math.sin(0.8 * t_ref)
        tilt_angle_deg = 0.0
        iso_shear = 0.02

    # =============================================================
    # PHASE 2: ANGLED FOCAL PLANE SCAN (15.0% -> 42.0%)[span_23](start_span)[span_23](end_span)
    # =============================================================
    elif t_ref < 4.2:
        dt = t_ref - 1.5
        dur = 2.7
        u = max(0.0, min(1.0, dt / dur))
        s = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)

        # Focal line rotates from 0° to 14.5° and climbs from y=620 to y=380[span_24](start_span)[span_24](end_span)
        focal_y = 620.0 - 240.0 * s
        tilt_angle_deg = 14.5 * math.sin(math.pi * s)
        cam_x = 45.0 * s
        cam_y = -35.0 * s
        cam_zoom = 1.07 + 0.08 * s
        iso_shear = 0.02 + 0.04 * s

    # =============================================================
    # PHASE 3: ISOMETRIC CRANE TRACK & TOY CADENCE (42.0% -> 68.0%)[span_25](start_span)[span_25](end_span)
    # =============================================================
    elif t_ref < 6.8:
        dt = t_ref - 4.2
        dur = 2.6
        u = max(0.0, min(1.0, dt / dur))

        # Sweep back diagonally across scene[span_26](start_span)[span_26](end_span)
        s = 0.5 * (1.0 - math.cos(math.pi * u))
        focal_y = 380.0 + 180.0 * s
        tilt_angle_deg = 14.5 * (1.0 - s) - 8.0 * s
        cam_x = 45.0 - 110.0 * s
        cam_y = -35.0 + 70.0 * s
        cam_zoom = 1.15 + 0.06 * math.sin(math.pi * u)
        iso_shear = 0.06 - 0.08 * s

        # Periodic miniature focus-pull micro-hits (rhythmic beat pulses)[span_27](start_span)[span_27](end_span)
        pulse_phase = math.sin(2.0 * math.pi * 1.5 * dt)
        if pulse_phase > 0.92:
            flash_val = 65.0
            chroma_disp = 8

    # =============================================================
    # PHASE 4: MACRO DETAIL PUNCH & RECOIL (68.0% -> 84.0%)[span_28](start_span)[span_28](end_span)
    # =============================================================
    elif t_ref < 8.4:
        dt = t_ref - 6.8
        t_punch = 0.22

        if dt < t_punch:
            # Accelerated crash-zoom into miniature subject[span_29](start_span)[span_29](end_span)
            u = dt / t_punch
            p = u**2.4
            cam_zoom = 1.21 + (1.75 - 1.21) * p
            cam_x = -65.0 + (15.0 - (-65.0)) * p
            cam_y = 35.0 + (-25.0 - 35.0) * p
            focal_y = 560.0 - 80.0 * p
            tilt_angle_deg = -8.0 * (1.0 - p)
            iso_shear = -0.02 * (1.0 - p)
        else:
            # Damped spring recoil settling at 1.48x[span_30](start_span)[span_30](end_span)
            dt_rebound = dt - t_punch
            decay = math.exp(-7.0 * dt_rebound)
            spring = decay * math.cos(16.0 * dt_rebound)

            cam_zoom = 1.48 + (1.75 - 1.48) * spring
            cam_x = 15.0 + 18.0 * spring
            cam_y = -25.0 - 12.0 * spring
            focal_y = 480.0 + 25.0 * spring
            tilt_angle_deg = 4.0 * spring
            iso_shear = 0.03 * spring

            # Macro punch impact shock[span_31](start_span)[span_31](end_span)
            if dt_rebound < 0.35:
                flash_val = 190.0 * math.exp(-18.0 * dt_rebound)
                shake_mag = 16.0 * math.exp(-14.0 * dt_rebound)
                chroma_disp = int(22.0 * math.exp(-15.0 * dt_rebound))

    # =============================================================
    # PHASE 5: CINEMATIC DIORAMA SETTLE (84.0% -> 100.0%)[span_32](start_span)[span_32](end_span)
    # =============================================================
    else:
        dt = t_ref - 8.4
        dur = 1.6
        u = max(0.0, min(1.0, dt / dur))
        s = 0.5 * (1.0 - math.cos(math.pi * u))

        cam_zoom = 1.48 - (1.48 - 1.06) * s
        cam_x = 15.0 * (1.0 - s)
        cam_y = -25.0 * (1.0 - s)
        focal_y = 480.0 + (580.0 - 480.0) * s
        tilt_angle_deg = 0.0
        iso_shear = 0.0

    return cam_x, cam_y, cam_zoom, iso_shear, focal_y, tilt_angle_deg, flash_val, shake_mag, chroma_disp


def apply_tilt_shift_optics(frame: np.ndarray, focal_y: float, tilt_angle_deg: float,
                            grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """
    Renders realistic tilt-shift depth of field[span_33](start_span)[span_33](end_span):
    1. Evaluates distance from an angled focal line[span_34](start_span)[span_34](end_span).
    2. Blends sharp base image with fast pyramidal downscale/upscale creamy bokeh blur[span_35](start_span)[span_35](end_span).
    3. Extracts high-luminance specular highlights in blur zones to synthesize circular bokeh discs[span_36](start_span)[span_36](end_span).
    """
    # Angled focal line geometry: Line passes through (WIDTH*0.5, focal_y) at tilt_angle_deg[span_37](start_span)[span_37](end_span)
    rad_tilt = math.radians(tilt_angle_deg)
    tan_t = math.tan(rad_tilt)
    cos_t = math.cos(rad_tilt)

    # Perpendicular distance from each pixel to the rotated focal axis[span_38](start_span)[span_38](end_span):
    # d = |(y - focal_y) - tan(theta) * (x - cx)| * cos(theta)[span_39](start_span)[span_39](end_span)
    cx = WIDTH * 0.5
    dist = np.abs((grid_y - focal_y) - tan_t * (grid_x - cx)) * cos_t

    # Compute continuous blur weight: 0.0 inside sharp focal band, rising smoothly to 1.0[span_40](start_span)[span_40](end_span)
    blur_weight = np.clip((dist - BASE_BAND_HALF_HEIGHT) / FALLOFF_DISTANCE, 0.0, 1.0)
    # Smoothstep interpolation (3w^2 - 2w^3) for optical falloff[span_41](start_span)[span_41](end_span)
    blur_weight = blur_weight * blur_weight * (3.0 - 2.0 * blur_weight)
    blur_weight_3ch = blur_weight[:, :, np.newaxis]

    # Fast pyramidal depth-of-field blur (Downscale -> Gaussian Blur -> Upscale)[span_42](start_span)[span_42](end_span)
    small_w = WIDTH // 4
    small_h = HEIGHT // 4
    downscaled = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    blurred_small = cv2.GaussianBlur(downscaled, (0, 0), sigmaX=7.5, sigmaY=7.5)
    bokeh_bg = cv2.resize(blurred_small, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)

    # Secondary deep blur tier for outer diorama periphery[span_43](start_span)[span_43](end_span)
    deep_small = cv2.GaussianBlur(blurred_small, (0, 0), sigmaX=9.0, sigmaY=9.0)
    deep_bokeh = cv2.resize(deep_small, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    bokeh_combined = cv2.addWeighted(bokeh_bg, 0.55, deep_bokeh, 0.45, 0)

    # Composite sharp focal zone with optical bokeh blur[span_44](start_span)[span_44](end_span)
    base_f = frame.astype(np.float32)
    bokeh_f = bokeh_combined.astype(np.float32)
    comp_f = base_f * (1.0 - blur_weight_3ch) + bokeh_f * blur_weight_3ch

    # Specular Bokeh Highlight Glints in Deep Blur Zones[span_45](start_span)[span_45](end_span)
    # Extract bright points that fall strictly in the out-of-focus background/foreground[span_46](start_span)[span_46](end_span)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, spec_mask = cv2.threshold(gray, SPECULAR_CUTOFF, 255, cv2.THRESH_BINARY)
    spec_mask = (spec_mask.astype(np.float32) / 255.0) * blur_weight

    if np.any(spec_mask > 0.05):
        spec_mask_small = cv2.resize(spec_mask, (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_LINEAR)
        # Circular dilation to create lens aperture discs[span_47](start_span)[span_47](end_span)
        kernel_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        spec_dilated = cv2.dilate(spec_mask_small, kernel_disc)
        spec_orbs = cv2.resize(spec_dilated, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Add luminous warm bokeh highlights[span_48](start_span)[span_48](end_span)
        bokeh_tint = np.array([255, 240, 215], dtype=np.float32)  # Incandescent white-gold[span_49](start_span)[span_49](end_span)
        orb_layer = (spec_orbs[:, :, np.newaxis] * bokeh_tint * 0.45)
        comp_f = np.clip(comp_f + orb_layer, 0, 255)

    return np.clip(comp_f, 0, 255).astype(np.uint8)


def generate_isometric_tilt_shift_miniature(
    input_path: str = None,
    output_path: str = None,
    duration_sec: float = DEFAULT_DURATION_SEC,
    fps: int = FPS
) -> str:
    """
    Renders the complete Isometric Tilt-Shift Miniature animation scaled dynamically
    to match the exact audio duration of the TTS sentence.
    """
    if input_path is None:
        input_path = os.path.join(DEFAULT_TEMP_DIR, "input.jpg")
    if output_path is None:
        output_path = os.path.join(DEFAULT_TEMP_DIR, "isometric_tilt_shift_miniature.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Loading '{input_path}'...")
    base_image = prepare_base_canvas(input_path, WIDTH, HEIGHT)
    vignette = build_vignette_mask(WIDTH, HEIGHT)

    print("Applying Miniature Toy Diorama Color Grading (High Vibrancy & Plastic Sheen)...")
    graded_image = apply_diorama_color_grade(base_image)

    total_frames = max(1, int(round(fps * duration_sec)))

    print(f"Rendering Isometric Tilt-Shift Miniature ({WIDTH}x{HEIGHT}, {fps} FPS, {duration_sec:.2f}s, {total_frames} frames)...")
    print("Features: Angled Optical Focal Band, Specular Bokeh Discs, Isometric Skew, Pyramidal DoF")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

    if not out.isOpened():
        raise RuntimeError(f"Error: Could not open VideoWriter at '{output_path}'.")

    cx = WIDTH * 0.5
    cy = HEIGHT * 0.5

    # Precompute static spatial coordinates for fast tilt-shift distance calculations[span_50](start_span)[span_50](end_span)
    grid_y, grid_x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # 1. Evaluate Dynamic Miniature Kinematics
        (
            cam_x,
            cam_y,
            cam_zoom,
            iso_shear,
            focal_y,
            tilt_angle_deg,
            flash_intensity,
            shake_intensity,
            chroma_shift,
        ) = evaluate_miniature_timeline(t, duration_sec)

        # 2. Camera Jitter on Macro Snap Point[span_51](start_span)[span_51](end_span)
        if shake_intensity > 0.5:
            jitter_x = random.uniform(-shake_intensity, shake_intensity)
            jitter_y = random.uniform(-shake_intensity, shake_intensity)
        else:
            jitter_x, jitter_y = 0.0, 0.0

        # 3. Construct 2.5D Isometric Affine Matrix (Scale + Translation + Orthographic Skew)[span_52](start_span)[span_52](end_span)
        M = np.array([
            [cam_zoom, iso_shear * cam_zoom, (1.0 - cam_zoom) * cx + cam_x + jitter_x],
            [0.0, cam_zoom, (1.0 - cam_zoom) * cy + cam_y + jitter_y],
        ], dtype=np.float32)

        warped_base = cv2.warpAffine(
            graded_image,
            M,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # 4. Render Tilt-Shift Optical Depth of Field & Specular Bokeh[span_53](start_span)[span_53](end_span)
        tilt_shifted = apply_tilt_shift_optics(
            warped_base,
            focal_y=focal_y,
            tilt_angle_deg=tilt_angle_deg,
            grid_x=grid_x,
            grid_y=grid_y
        )

        # 5. Lateral Chromatic Aberration on Beat Pulses & Impact Snaps[span_54](start_span)[span_54](end_span)
        if chroma_shift > 1:
            b, g, r = cv2.split(tilt_shifted)
            r = np.roll(r, chroma_shift, axis=1)
            b = np.roll(b, -chroma_shift, axis=1)
            tilt_shifted = cv2.merge([b, g, r])

        # 6. Additive Shutter Exposure Flash[span_55](start_span)[span_55](end_span)
        if flash_intensity > 1.0:
            f_val = int(min(flash_intensity, 255.0))
            tilt_shifted = cv2.add(tilt_shifted, np.full_like(tilt_shifted, f_val))

        # 7. Final Cinematic Lens Shading & Vignette Pass[span_56](start_span)[span_56](end_span)
        final_float = (tilt_shifted.astype(np.float32) / 255.0) * vignette
        final_frame = np.clip(final_float * 255.0, 0, 255).astype(np.uint8)

        out.write(final_frame)

        if (frame_idx + 1) % (fps * 2) == 0 or frame_idx == total_frames - 1:
            print(f"Rendered {frame_idx + 1}/{total_frames} frames...")

    out.release()
    print(f"Done! Video successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    test_in = sys.argv[1] if len(sys.argv) > 1 else "1.jpg"
    test_out = sys.argv[2] if len(sys.argv) > 2 else "isometric_tilt_shift_miniature.mp4"
    test_dur = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DURATION_SEC

    generate_isometric_tilt_shift_miniature(test_in, test_out, duration_sec=test_dur)
