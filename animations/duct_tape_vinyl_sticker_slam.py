import math
import os
import random
import sys
import cv2
import numpy as np

# ================= CONFIGURATION & PATHS =================
# Adapted for Streamlit Cloud / Linux container environment
BASE_PROJECT_DIR = os.getcwd()
DEFAULT_TEMP_DIR = os.path.join(BASE_PROJECT_DIR, "storage", "temp")

# Video Settings (16:9 Landscape Widescreen)
WIDTH = 1920
HEIGHT = 1080
FPS = 30
DEFAULT_DURATION_SEC = 10.0

# Sticker Dimensions
CARD_W = 1380
CARD_H = 776
BORDER_PX = 24  # Thick white die-cut vinyl border
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
    """Precomputes 2D radial falloff matrix for tactile workshop depth[span_1](start_span)[span_1](end_span)."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radial_dist = (xx**2) + ((yy * 1.25) ** 2)
    return np.clip(1.0 - 0.42 * radial_dist, 0.0, 1.0)[:, :, np.newaxis]


def build_workbench_background(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """Synthesizes a dark, distressed industrial workbench / skate-deck surface[span_2](start_span)[span_2](end_span)."""
    np.random.seed(77)
    base = np.full((height, width, 3), (38, 42, 46), dtype=np.uint8)

    noise = np.random.uniform(-18.0, 18.0, (height, width, 1)).astype(np.float32)
    bg = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    for _ in range(40):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2 = x1 + random.randint(-220, 220)
        y2 = y1 + random.randint(-140, 140)
        color = random.choice([(25, 28, 30), (55, 60, 65), (75, 80, 85)])
        cv2.line(bg, (x1, y1), (x2, y2), color, random.randint(1, 2), cv2.LINE_AA)

    return bg


def build_vinyl_sticker_card(base_img: np.ndarray) -> np.ndarray:
    """
    Constructs a die-cut glossy vinyl sticker:
    - Centers scaled photo inside a thick, rounded white vinyl contour[span_3](start_span)[span_3](end_span).
    - Adds glossy laminate inner-bevel highlights and drop shadow[span_4](start_span)[span_4](end_span).
    """
    total_w = CARD_W + BORDER_PX * 2
    total_h = CARD_H + BORDER_PX * 2
    card = np.zeros((total_h, total_w, 4), dtype=np.uint8)

    photo_scaled = cv2.resize(base_img, (CARD_W, CARD_H), interpolation=cv2.INTER_CUBIC)

    border_mask = np.zeros((total_h, total_w), dtype=np.uint8)
    radius = 36
    cv2.rectangle(border_mask, (radius, 0), (total_w - radius, total_h), 255, -1)
    cv2.rectangle(border_mask, (0, radius), (total_w, total_h - radius), 255, -1)
    cv2.circle(border_mask, (radius, radius), radius, 255, -1)
    cv2.circle(border_mask, (total_w - radius, radius), radius, 255, -1)
    cv2.circle(border_mask, (radius, total_h - radius), radius, 255, -1)
    cv2.circle(border_mask, (total_w - radius, total_h - radius), radius, 255, -1)

    card[border_mask > 0, :3] = (248, 246, 242)
    card[border_mask > 0, 3] = 255

    card[BORDER_PX : BORDER_PX + CARD_H, BORDER_PX : BORDER_PX + CARD_W, :3] = photo_scaled

    edge_strip = cv2.Canny(border_mask, 100, 200)
    card[edge_strip > 0, :3] = (255, 255, 255)

    return card


def create_procedural_duct_tape(length: int, width: int, base_bgr: tuple,
                                 wrinkle_intensity: float = 1.0) -> np.ndarray:
    """
    Synthesizes textured industrial duct tape:
    - Micro cross-hatch fabric mesh reinforcement lines[span_5](start_span)[span_5](end_span).
    - Semi-translucent adhesive body with frayed torn jagged ends[span_6](start_span)[span_6](end_span).
    - Specular light ridge along plastic film backing[span_7](start_span)[span_7](end_span).
    """
    tape = np.zeros((width, length, 4), dtype=np.uint8)

    tape[:, :, 0] = base_bgr[0]
    tape[:, :, 1] = base_bgr[1]
    tape[:, :, 2] = base_bgr[2]
    tape[:, :, 3] = 228

    mesh_color = np.clip(np.array(base_bgr, dtype=np.float32) * 1.14 + 18, 0, 255).astype(np.uint8)
    tape[::4, :, :3] = mesh_color
    tape[:, ::4, :3] = mesh_color

    # Shape: (width, 1, 1) for 3D broadcasting against (width, length, 3)
    y_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)[:, np.newaxis, np.newaxis]
    specular_ridge = np.exp(-((y_coords / 0.32) ** 2)) * 55.0
    tape[:, :, :3] = np.clip(tape[:, :, :3].astype(np.float32) + specular_ridge, 0, 255).astype(np.uint8)

    tear_len = 22
    for y in range(width):
        jitter_left = int(math.sin(y * 0.42) * 6 + math.cos(y * 1.1) * 3 + 10)
        jitter_right = int(math.cos(y * 0.38) * 6 + math.sin(y * 1.25) * 3 + 10)
        tape[y, : max(0, jitter_left), 3] = 0
        tape[y, length - max(0, jitter_right) :, 3] = 0

    if wrinkle_intensity > 0:
        for _ in range(int(6 * wrinkle_intensity)):
            wx = random.randint(tear_len + 20, length - tear_len - 20)
            cv2.line(tape, (wx, 0), (wx + random.randint(-15, 15), width), (20, 20, 20, 70), 2, cv2.LINE_AA)
            cv2.line(tape, (wx + 1, 0), (wx + 1 + random.randint(-15, 15), width), (255, 255, 255, 90), 1, cv2.LINE_AA)

    return tape


def overlay_rgba(canvas: np.ndarray, overlay_rgba_layer: np.ndarray, M: np.ndarray,
                 draw_drop_shadow: bool = True, shadow_offset: tuple = (12, 18),
                 shadow_alpha: float = 0.55) -> np.ndarray:
    """Warps an RGBA layer and composites it over the canvas with drop shadows[span_8](start_span)[span_8](end_span)."""
    if draw_drop_shadow:
        shadow_mask = cv2.warpAffine(
            overlay_rgba_layer[:, :, 3],
            M,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        if np.any(shadow_mask > 0):
            M_sh = np.array([
                [1.0, 0.0, float(shadow_offset[0])],
                [0.0, 1.0, float(shadow_offset[1])]
            ], dtype=np.float32)
            shifted_mask = cv2.warpAffine(shadow_mask, M_sh, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
            blurred_shadow = cv2.GaussianBlur(shifted_mask, (25, 25), 0)

            s_weight = (blurred_shadow.astype(np.float32) / 255.0 * shadow_alpha)[:, :, np.newaxis]
            canvas_f = canvas.astype(np.float32)
            canvas = np.clip(canvas_f * (1.0 - s_weight), 0, 255).astype(np.uint8)

    warped_rgba = cv2.warpAffine(
        overlay_rgba_layer,
        M,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    alpha = warped_rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = warped_rgba[:, :, :3].astype(np.float32)
    bg = canvas.astype(np.float32)

    comp = bg * (1.0 - alpha) + fg * alpha
    return np.clip(comp, 0, 255).astype(np.uint8)


def evaluate_slam_timeline(t: float, duration_sec: float) -> tuple[dict, list[dict], float, float, int, float]:
    """
    Evaluates kinematic trajectory, corner tape slaps, and sheen sweep dynamically
    scaled to the target audio duration[span_9](start_span)[span_9](end_span).
    """
    # Scale factor mapping the current timeline to the master 10-second reference curve
    time_scale = 10.0 / max(duration_sec, 0.1)
    t_ref = t * time_scale

    flash_val = 0.0
    shake_mag = 0.0
    chroma_disp = 0
    sheen_x = -1.0

    card_state = {"scale": 1.0, "tx": 0.0, "ty": 0.0, "rot": -2.2, "visible": True}

    if t_ref < 1.0:
        u = t_ref / 1.0
        p = u**2.6
        card_state["scale"] = 3.2 - (3.2 - 1.0) * p
        card_state["rot"] = -14.0 + 11.8 * p
        card_state["tx"] = 65.0 * (1.0 - p)
        card_state["ty"] = -120.0 * (1.0 - p)
    else:
        dt = t_ref - 1.0
        decay = math.exp(-7.2 * dt)
        spring = decay * math.cos(16.0 * dt)
        card_state["scale"] = 1.0 + 0.18 * spring
        card_state["rot"] = -2.2 - 1.8 * spring
        card_state["tx"] = 12.0 * spring
        card_state["ty"] = -8.0 * spring

        if dt < 0.35:
            flash_val = max(flash_val, 230.0 * math.exp(-17.0 * dt))
            shake_mag = max(shake_mag, 24.0 * math.exp(-13.0 * dt))
            chroma_disp = max(chroma_disp, int(26.0 * math.exp(-14.0 * dt)))

    # Tape slam reference timings across 10-second baseline[span_10](start_span)[span_10](end_span)
    tape_slams = [
        (2.20, 0, 270, 180, -32.0, 150.0),
        (3.60, 1, 1650, 900, -28.0, 160.0),
        (5.00, 2, 1650, 180, 36.0, 170.0),
        (6.40, 3, 270, 900, 34.0, 185.0),
    ]

    tapes_render = []
    for t_hit, idx, rx, ry, r_rot, f_pwr in tape_slams:
        if t_ref < t_hit - 0.22:
            continue
        elif t_ref < t_hit:
            u = (t_ref - (t_hit - 0.22)) / 0.22
            p = u**2.2
            t_scale = 2.4 - 1.4 * p
            t_rot = r_rot + 15.0 * (1.0 - p)
            tapes_render.append({"idx": idx, "x": rx, "y": ry, "rot": t_rot, "scale": t_scale})
        else:
            dt = t_ref - t_hit
            decay = math.exp(-8.0 * dt)
            spring = decay * math.cos(18.0 * dt)
            t_scale = 1.0 + 0.14 * spring
            t_rot = r_rot - 2.5 * spring

            if dt < 0.30:
                flash_val = max(flash_val, f_pwr * math.exp(-18.0 * dt))
                shake_mag = max(shake_mag, 16.0 * (f_pwr / 185.0) * math.exp(-14.0 * dt))
                chroma_disp = max(chroma_disp, int(18.0 * math.exp(-15.0 * dt)))

            tapes_render.append({"idx": idx, "x": rx, "y": ry, "rot": t_rot, "scale": t_scale})

    if 7.2 <= t_ref < 9.4:
        sheen_x = (t_ref - 7.2) / 2.2

    return card_state, tapes_render, flash_val, shake_mag, chroma_disp, sheen_x


def apply_glossy_sheen_sweep(canvas: np.ndarray, progress: float) -> np.ndarray:
    """Simulates a diagonal sunlight gloss sweep across the laminated sticker surface[span_11](start_span)[span_11](end_span)."""
    if progress < 0.0 or progress > 1.0:
        return canvas

    center_x = int(round(progress * (WIDTH + 600) - 300))
    grid_y, grid_x = np.mgrid[0:HEIGHT:4, 0:WIDTH:4].astype(np.float32)
    band = (grid_x - grid_y * 0.45) - center_x
    sheen = np.exp(-((band / 90.0) ** 2)) * 65.0

    sheen_full = cv2.resize(sheen, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    sheen_3ch = sheen_full[:, :, np.newaxis]

    canvas_f = canvas.astype(np.float32) + sheen_3ch
    return np.clip(canvas_f, 0, 255).astype(np.uint8)


def generate_duct_tape_vinyl_sticker_slam(
    input_path: str = None,
    output_path: str = None,
    duration_sec: float = DEFAULT_DURATION_SEC,
    fps: int = FPS
) -> str:
    """
    Renders the complete Duct Tape & Vinyl Sticker Slam animation scaled dynamically
    to match the exact audio duration of the TTS sentence.
    """
    if input_path is None:
        input_path = os.path.join(DEFAULT_TEMP_DIR, "input.jpg")
    if output_path is None:
        output_path = os.path.join(DEFAULT_TEMP_DIR, "duct_tape_vinyl_sticker_slam.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Loading '{input_path}'...")
    base_image = prepare_base_canvas(input_path, WIDTH, HEIGHT)
    vignette = build_vignette_mask(WIDTH, HEIGHT)

    print("Building Industrial Workbench & Vinyl Die-Cut Sticker Card...")
    workbench = build_workbench_background(WIDTH, HEIGHT)
    sticker_card = build_vinyl_sticker_card(base_image)

    print("Synthesizing 4 Textured Duct Tape Elements with Frayed Edges...")
    tape_library = [
        create_procedural_duct_tape(380, 85, (175, 178, 180), wrinkle_intensity=1.2),  # Silver[span_12](start_span)[span_12](end_span)
        create_procedural_duct_tape(410, 85, (45, 215, 245), wrinkle_intensity=1.0),   # Caution Yellow[span_13](start_span)[span_13](end_span)
        create_procedural_duct_tape(380, 80, (245, 185, 40), wrinkle_intensity=0.8),   # Neon Cyan[span_14](start_span)[span_14](end_span)
        create_procedural_duct_tape(400, 88, (32, 35, 38), wrinkle_intensity=1.4),     # Black Gaffer[span_15](start_span)[span_15](end_span)
    ]

    total_frames = max(1, int(round(fps * duration_sec)))

    print(f"Rendering Duct Tape & Vinyl Sticker Slam ({WIDTH}x{HEIGHT}, {fps} FPS, {duration_sec:.2f}s, {total_frames} frames)...")
    print("Features: Gravitational Card Slam, 4x Corner Tape Slaps, Damped Rebounds, Specular Gloss Sweep")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (WIDTH, HEIGHT))

    if not out.isOpened():
        raise RuntimeError(f"Error: Could not open VideoWriter at '{output_path}'.")

    cx = WIDTH * 0.5
    cy = HEIGHT * 0.5

    card_cx = sticker_card.shape[1] * 0.5
    card_cy = sticker_card.shape[0] * 0.5

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        norm_t = frame_idx / float(total_frames)

        cam_zoom = 1.03 + 0.08 * (0.5 * (1.0 - math.cos(math.pi * norm_t)))
        cam_tx = 15.0 * math.sin(0.7 * t)
        cam_ty = 10.0 * math.cos(0.5 * t)

        (
            card_state,
            tapes_render,
            flash_intensity,
            shake_intensity,
            chroma_shift,
            sheen_progress,
        ) = evaluate_slam_timeline(t, duration_sec)

        if shake_intensity > 0.5:
            jitter_x = random.uniform(-shake_intensity, shake_intensity)
            jitter_y = random.uniform(-shake_intensity, shake_intensity)
            jitter_roll = random.uniform(-0.5, 0.5) * (shake_intensity / 24.0)
        else:
            jitter_x, jitter_y, jitter_roll = 0.0, 0.0, 0.0

        frame_canvas = workbench.copy()

        if card_state["visible"]:
            M_card = cv2.getRotationMatrix2D(
                (card_cx, card_cy),
                card_state["rot"],
                card_state["scale"]
            )
            M_card[0, 2] += (cx - card_cx) + card_state["tx"]
            M_card[1, 2] += (cy - card_cy) + card_state["ty"]

            sh_alpha = float(np.clip(0.65 / max(card_state["scale"], 1.0), 0.15, 0.65))
            sh_dist_x = int(14 * card_state["scale"])
            sh_dist_y = int(24 * card_state["scale"])

            frame_canvas = overlay_rgba(
                frame_canvas,
                sticker_card,
                M_card,
                draw_drop_shadow=True,
                shadow_offset=(sh_dist_x, sh_dist_y),
                shadow_alpha=sh_alpha
            )

        for tape_info in tapes_render:
            tape_raw = tape_library[tape_info["idx"]]
            tc_x = tape_raw.shape[1] * 0.5
            tc_y = tape_raw.shape[0] * 0.5

            M_tape = cv2.getRotationMatrix2D((tc_x, tc_y), tape_info["rot"], tape_info["scale"])
            M_tape[0, 2] += (tape_info["x"] - tc_x)
            M_tape[1, 2] += (tape_info["y"] - tc_y)

            frame_canvas = overlay_rgba(
                frame_canvas,
                tape_raw,
                M_tape,
                draw_drop_shadow=True,
                shadow_offset=(8, 10),
                shadow_alpha=0.50
            )

        if sheen_progress >= 0.0:
            frame_canvas = apply_glossy_sheen_sweep(frame_canvas, sheen_progress)

        M_cam = cv2.getRotationMatrix2D((cx, cy), jitter_roll, max(cam_zoom, 1.0))
        M_cam[0, 2] += cam_tx + jitter_x
        M_cam[1, 2] += cam_ty + jitter_y

        warped_final = cv2.warpAffine(
            frame_canvas,
            M_cam,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        if chroma_shift > 1:
            b, g, r = cv2.split(warped_final)
            r = np.roll(r, chroma_shift, axis=1)
            b = np.roll(b, -chroma_shift, axis=1)
            warped_final = cv2.merge([b, g, r])

        if flash_intensity > 1.0:
            f_val = int(min(flash_intensity, 255.0))
            warped_final = cv2.add(warped_final, np.full_like(warped_final, f_val))

        final_float = (warped_final.astype(np.float32) / 255.0) * vignette
        final_float = np.where(
            final_float < 0.5,
            2.0 * (final_float**1.20),
            1.0 - 2.0 * ((1.0 - final_float) ** 1.20) * 0.5,
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
    test_out = sys.argv[2] if len(sys.argv) > 2 else "duct_tape_vinyl_sticker_slam.mp4"
    test_dur = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DURATION_SEC

    generate_duct_tape_vinyl_sticker_slam(test_in, test_out, duration_sec=test_dur)
