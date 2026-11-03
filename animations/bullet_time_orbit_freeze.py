import math
import os
import sys
import cv2
import numpy as np

# ================= CONFIGURATION & PATHS =================
# Project root directory adapted for Streamlit Cloud / headless containers
BASE_PROJECT_DIR = os.getcwd()
DEFAULT_TEMP_DIR = os.path.join(BASE_PROJECT_DIR, "storage", "temp")

# Video Settings (16:9 Landscape Widescreen)
WIDTH = 1920
HEIGHT = 1080
FPS = 30
DEFAULT_DURATION_SEC = 10.0

# 3D Virtual Camera Optics
FOCAL_LENGTH = 1350.0   # Perspective pinhole focal length
Z_WIDE = 1350.0         # Wide distance
Z_MACRO = 880.0         # Close-up bullet-time orbit distance

# Orbit Rig Limits
MAX_ORBIT_YAW = 28.0    # Peak lateral orbit angle (degrees)
MAX_ORBIT_PITCH = 6.5   # Vertical tilt elevation during sweep (degrees)
ORBIT_RADIUS_X = 260.0  # Horizontal rig displacement along arc (pixels)
# =========================================================


def prepare_base_canvas(img_path: str, target_w: int = WIDTH, target_h: int = HEIGHT) -> np.ndarray:
    """Loads image and fits it to 16:9 aspect ratio via center-crop & bicubic scale[span_0](start_span)[span_0](end_span)."""
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
    """Precomputes 2D radial falloff matrix for cinematic edge depth[span_1](start_span)[span_1](end_span)."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radial_dist = (xx**2) + ((yy * 1.25) ** 2)
    return np.clip(1.0 - 0.42 * radial_dist, 0.0, 1.0)[:, :, np.newaxis]


def compute_3d_perspective_matrix(pitch: float, yaw: float, roll: float,
                                   trans_x: float, trans_y: float, z_dist: float) -> np.ndarray:
    """
    Computes a 3D perspective projection homography for the 16:9 image plane[span_2](start_span)[span_2](end_span).
    Simulates a circular curved camera track orbiting around the focal center[span_3](start_span)[span_3](end_span).
    """
    rad_x = math.radians(pitch)
    rad_y = math.radians(yaw)
    rad_z = math.radians(roll)

    # 3D Euler Rotation matrices[span_4](start_span)[span_4](end_span)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(rad_x), -math.sin(rad_x)],
        [0.0, math.sin(rad_x),  math.cos(rad_x)]
    ], dtype=np.float32)

    Ry = np.array([
        [ math.cos(rad_y), 0.0, math.sin(rad_y)],
        [ 0.0,             1.0, 0.0],
        [-math.sin(rad_y), 0.0, math.cos(rad_y)]
    ], dtype=np.float32)

    Rz = np.array([
        [math.cos(rad_z), -math.sin(rad_z), 0.0],
        [math.sin(rad_z),  math.cos(rad_z), 0.0],
        [0.0,              0.0,             1.0]
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx

    # Center-aligned 3D vertices of the image canvas[span_5](start_span)[span_5](end_span)
    hw = WIDTH * 0.5
    hh = HEIGHT * 0.5
    corners_3d = np.array([
        [-hw, -hh, 0.0],
        [ hw, -hh, 0.0],
        [ hw,  hh, 0.0],
        [-hw,  hh, 0.0],
    ], dtype=np.float32)

    # Apply 3D rotation and translational arc[span_6](start_span)[span_6](end_span)
    transformed = (R @ corners_3d.T).T
    transformed[:, 0] += trans_x
    transformed[:, 1] += trans_y
    transformed[:, 2] += z_dist

    # Perspective pinhole screen projection[span_7](start_span)[span_7](end_span)
    depth = np.maximum(transformed[:, 2], 20.0)
    screen_x = FOCAL_LENGTH * (transformed[:, 0] / depth) + hw
    screen_y = FOCAL_LENGTH * (transformed[:, 1] / depth) + hh

    dst_pts = np.stack([screen_x, screen_y], axis=1).astype(np.float32)
    src_pts = np.array([
        [0.0, 0.0],
        [float(WIDTH), 0.0],
        [float(WIDTH), float(HEIGHT)],
        [0.0, float(HEIGHT)]
    ], dtype=np.float32)

    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def apply_zoom_motion_blur(img: np.ndarray, cx: float, cy: float, intensity: float) -> np.ndarray:
    """Blends sub-scaled optical samples to simulate explosive radial zoom smear[span_8](start_span)[span_8](end_span)."""
    if intensity < 0.05:
        return img

    delta = intensity * 0.065
    m1 = cv2.getRotationMatrix2D((cx, cy), 0, 1.0 + delta)
    m2 = cv2.getRotationMatrix2D((cx, cy), 0, 1.0 - delta)

    s1 = cv2.warpAffine(img, m1, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    s2 = cv2.warpAffine(img, m2, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    weight = float(np.clip(intensity * 0.5, 0.0, 0.65))
    return cv2.addWeighted(img, 1.0 - weight, cv2.addWeighted(s1, 0.5, s2, 0.5, 0), weight, 0)


def generate_bullet_time_orbit_freeze(
    input_path: str = None,
    output_path: str = None,
    duration_sec: float = DEFAULT_DURATION_SEC,
    fps: int = FPS
) -> str:
    """
    Renders the complete 3D Bullet-Time Orbit Freeze animation scaled dynamically
    to match the exact audio duration of the TTS sentence.
    """
    if input_path is None:
        input_path = os.path.join(DEFAULT_TEMP_DIR, "input.jpg")
    if output_path is None:
        output_path = os.path.join(DEFAULT_TEMP_DIR, "bullet_time_orbit_freeze.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Loading '{input_path}'...")
    base_image = prepare_base_canvas(input_path, WIDTH, HEIGHT)
    vignette = build_vignette_mask(WIDTH, HEIGHT)

    total_frames = max(1, int(round(fps * duration_sec)))

    # Adaptive proportional milestones (Phase 1: 0%->20%, Phase 2: 20%->70%, Phase 3: 70%->100%)
    t_phase1_end = 0.20 * duration_sec
    t_phase2_end = 0.70 * duration_sec
    dur_freeze = max(t_phase2_end - t_phase1_end, 1e-4)
    dur_settle = max(duration_sec - t_phase2_end, 1e-4)

    print(f"Rendering Bullet-Time Orbit Freeze ({WIDTH}x{HEIGHT}, {fps} FPS, {duration_sec:.2f}s, {total_frames} frames)...")
    print(f"Timeline: 0-{t_phase1_end:.2f}s Rush-In -> {t_phase1_end:.2f}s Freeze Flash -> {t_phase1_end:.2f}-{t_phase2_end:.2f}s 3D Orbit Arc -> {t_phase2_end:.2f}s Snap-Out -> Settle")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

    if not out.isOpened():
        raise RuntimeError(f"Error: Could not open VideoWriter at '{output_path}'.")

    cx = WIDTH * 0.5
    cy = HEIGHT * 0.5

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # =============================================================
        # 1. TIMELINE STATE MACHINE (RUSH -> FREEZE ORBIT -> SNAP-OUT)
        # =============================================================
        flash_intensity = 0.0
        chroma_split = 0
        zoom_blur_mag = 0.0

        # PHASE 1: Real-Time High-Velocity Rush-In (0.0% -> 20.0%)[span_9](start_span)[span_9](end_span)
        if t < t_phase1_end:
            u = t / max(t_phase1_end, 1e-4)
            # Exponential acceleration rush towards subject[span_10](start_span)[span_10](end_span)
            speed_curve = u**2.2
            current_z = Z_WIDE - (Z_WIDE - Z_MACRO) * speed_curve
            current_yaw = -6.0 * (1.0 - u)
            current_pitch = 3.0 * math.sin(math.pi * u)
            current_roll = -1.2 * math.sin(math.pi * u)
            current_tx = 40.0 * (1.0 - u)
            current_ty = -25.0 * (1.0 - u)

            # Zoom motion smear ramps up right before freeze impact[span_11](start_span)[span_11](end_span)
            if u > 0.70:
                zoom_blur_mag = (u - 0.70) / 0.30

        # PHASE 2: Time Freeze & Sweeping 3D Orbital Arc (20.0% -> 70.0%)[span_12](start_span)[span_12](end_span)
        elif t < t_phase2_end:
            dt_freeze = t - t_phase1_end
            u = dt_freeze / dur_freeze

            # Initial Freeze Impact Strobe (Detonates right at freeze moment)[span_13](start_span)[span_13](end_span)
            if dt_freeze < 0.35:
                flash_intensity = 210.0 * math.exp(-18.0 * dt_freeze)
                chroma_split = int(22.0 * math.exp(-16.0 * dt_freeze))
                zoom_blur_mag = float(math.exp(-14.0 * dt_freeze))

            # Quintic smootherstep progression for the panoramic orbit sweep[span_14](start_span)[span_14](end_span)
            s = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)

            # Circular Rig Orbit: Yaw sweeps from left (-MAX) to right (+MAX)[span_15](start_span)[span_15](end_span)
            current_yaw = -MAX_ORBIT_YAW + (2.0 * MAX_ORBIT_YAW) * s

            # Parabolic Pitch Arc: Dips slightly then rises[span_16](start_span)[span_16](end_span)
            current_pitch = -MAX_ORBIT_PITCH * math.sin(math.pi * s) + 1.5

            # Camera Roll follows tangential curvature of the track[span_17](start_span)[span_17](end_span)
            current_roll = 2.4 * math.cos(math.pi * s)

            # Translation along circular camera array track[span_18](start_span)[span_18](end_span)
            rad_yaw = math.radians(current_yaw)
            current_tx = -ORBIT_RADIUS_X * math.sin(rad_yaw)
            current_ty = 35.0 * math.sin(math.pi * s)

            # Depth curvature: Camera arcs around the focal subject[span_19](start_span)[span_19](end_span)
            orbit_depth_sag = 60.0 * (1.0 - math.cos(rad_yaw))
            current_z = Z_MACRO + orbit_depth_sag

            # Array Camera Trigger Pulses (Subtle micro-flashes simulating multi-DSLR triggers)[span_20](start_span)[span_20](end_span)
            trigger_phase = math.sin(2.0 * math.pi * 5.0 * dt_freeze)  # 5 Hz array triggers[span_21](start_span)[span_21](end_span)
            if trigger_phase > 0.94 and dt_freeze > 0.4:
                flash_intensity = max(flash_intensity, 28.0)

        # PHASE 3: Explosive Snap-Out & Reality Unfreeze (70.0% -> 100.0%)[span_22](start_span)[span_22](end_span)
        else:
            dt_snap = t - t_phase2_end
            u = min(dt_snap / dur_settle, 1.0)

            # Snap-out exposure blast at transition moment[span_23](start_span)[span_23](end_span)
            if dt_snap < 0.40:
                flash_intensity = 220.0 * math.exp(-16.0 * dt_snap)
                chroma_split = int(24.0 * math.exp(-18.0 * dt_snap))
                zoom_blur_mag = float(math.exp(-12.0 * dt_snap))

            # Damped harmonic recoil returning to stabilized cinematic wide[span_24](start_span)[span_24](end_span)
            decay = math.exp(-5.5 * dt_snap)
            spring = decay * math.cos(12.0 * dt_snap)

            current_z = Z_MACRO + (Z_WIDE - 150.0 - Z_MACRO) * (1.0 - decay) - (120.0 * spring)
            current_yaw = MAX_ORBIT_YAW * decay * math.cos(8.0 * dt_snap)
            current_pitch = 1.0 * decay
            current_roll = -1.8 * decay * math.sin(8.0 * dt_snap)
            current_tx = 25.0 * decay * math.cos(6.0 * dt_snap)
            current_ty = -15.0 * decay

        # =============================================================
        # 2. 3D PERSPECTIVE HOMOGRAPHY WARP[span_25](start_span)[span_25](end_span)
        # =============================================================
        M = compute_3d_perspective_matrix(
            pitch=current_pitch,
            yaw=current_yaw,
            roll=current_roll,
            trans_x=current_tx,
            trans_y=current_ty,
            z_dist=current_z
        )

        warped = cv2.warpPerspective(
            base_image,
            M,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # =============================================================
        # 3. KINETIC OPTICAL EFFECTS & POST-PROCESSING[span_26](start_span)[span_26](end_span)
        # =============================================================
        # Radial Zoom Smear on rush and snap points[span_27](start_span)[span_27](end_span)
        if zoom_blur_mag > 0.05:
            warped = apply_zoom_motion_blur(warped, cx, cy, zoom_blur_mag)

        # Lateral Chromatic Aberration on impact frames[span_28](start_span)[span_28](end_span)
        if chroma_split > 1:
            b, g, r = cv2.split(warped)
            r = np.roll(r, chroma_split, axis=1)
            b = np.roll(b, -chroma_split, axis=1)
            warped = cv2.merge([b, g, r])

        # Multi-Camera Shutter Flash / Impact Bloom[span_29](start_span)[span_29](end_span)
        if flash_intensity > 1.0:
            f_val = int(min(flash_intensity, 255.0))
            warped = cv2.add(warped, np.full_like(warped, f_val))

        # Contrast Grading & Cinematic Lens Vignette[span_30](start_span)[span_30](end_span)
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
    test_out = sys.argv[2] if len(sys.argv) > 2 else "bullet_time_orbit_freeze.mp4"
    test_dur = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DURATION_SEC

    generate_bullet_time_orbit_freeze(test_in, test_out, duration_sec=test_dur)
