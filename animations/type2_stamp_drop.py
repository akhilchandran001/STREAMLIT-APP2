import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Theme Palette (exact match to reference card)
COLOR_TEXT_BLACK = (20, 20, 20)      # #141414 Dark Charcoal
COLOR_ACCENT_RED = (200, 29, 59)     # #C81D3B Crimson Red
COLOR_BAR_RED = (200, 29, 59)
CANVAS_W = 1920
CANVAS_H = 1080


def _get_font(size: int, italic: bool = False, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Attempts to locate system sans-serif fonts, with synthetic fallback."""
    candidate_paths = []
    if italic:
        candidate_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
            "/system/fonts/Roboto-BoldItalic.ttf",
        ]
    else:
        candidate_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/system/fonts/Roboto-Bold.ttf",
        ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # Fallback to default
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def render_headline_banner(
    text_data: dict,
    canvas_w: int = CANVAS_W,
    banner_h: int = 240
) -> Image.Image:
    """
    Renders the exact typography from the reference card:
    e.g. 'For' [Black Bold] + 'better' [Red Bold Italic] + 'experience' [Black Bold]
    followed by a centered crimson underline bar.
    """
    banner = Image.new("RGBA", (canvas_w, banner_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(banner)

    font_regular = _get_font(size=72, italic=False, bold=True)
    font_italic = _get_font(size=74, italic=True, bold=True)

    prefix = text_data.get("prefix", "").strip()
    highlight = text_data.get("highlight", "").strip()
    suffix = text_data.get("suffix", "").strip()

    # Measure text segment widths
    def get_width(txt, font):
        if not txt:
            return 0
        bbox = draw.textbbox((0, 0), txt, font=font)
        return bbox[2] - bbox[0]

    space_w = 16
    w_prefix = get_width(prefix, font_regular) if prefix else 0
    w_highlight = get_width(highlight, font_italic) if highlight else 0
    w_suffix = get_width(suffix, font_regular) if suffix else 0

    total_w = w_prefix + (space_w if prefix else 0) + w_highlight + (space_w if suffix else 0) + w_suffix
    start_x = (canvas_w - total_w) // 2
    baseline_y = 75

    cur_x = start_x
    if prefix:
        draw.text((cur_x, baseline_y), prefix, font=font_regular, fill=COLOR_TEXT_BLACK + (255,))
        cur_x += w_prefix + space_w

    if highlight:
        draw.text((cur_x, baseline_y - 2), highlight, font=font_italic, fill=COLOR_ACCENT_RED + (255,))
        cur_x += w_highlight + space_w

    if suffix:
        draw.text((cur_x, baseline_y), suffix, font=font_regular, fill=COLOR_TEXT_BLACK + (255,))

    # Centered crimson underline bar
    bar_width = max(110, w_highlight + 10)
    bar_x1 = (canvas_w - bar_width) // 2
    bar_x2 = bar_x1 + bar_width
    bar_y = baseline_y + 90
    draw.rounded_rectangle([bar_x1, bar_y, bar_x2, bar_y + 6], radius=3, fill=COLOR_BAR_RED + (255,))

    return banner


def build_drop_shadow(img_rgba: np.ndarray, blur_ksize: int = 51, opacity: float = 0.45) -> np.ndarray:
    """Creates a soft feathered drop shadow underneath the image card."""
    h, w = img_rgba.shape[:2]
    alpha = img_rgba[:, :, 3]

    shadow = np.zeros((h + 80, w + 80, 4), dtype=np.uint8)
    shadow[40:40+h, 40:40+w, 3] = (alpha.astype(np.float32) * opacity).astype(np.uint8)
    shadow[:, :, :3] = 0

    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    shadow = cv2.GaussianBlur(shadow, (ksize, ksize), 0)
    return shadow


def render_animated_stamp_drop(
    bg_video_path: str,
    foreground_image_path: str,
    output_path: str,
    duration: float,
    headline_data: dict = None,
    fps: int = 30
) -> str:
    """
    Renders an animated Type 2 scene:
    - Loops background video
    - Stamps image down with elastic impact overshoot during the first 0.4s
    - Displays styled headline with red accent and underline bar
    - Subtle resting float for remainder of duration
    """
    total_frames = max(1, int(duration * fps))

    # 1. Load foreground image
    raw_img = cv2.imread(foreground_image_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        raise ValueError(f"Could not load foreground image: {foreground_image_path}")

    if raw_img.shape[2] == 3:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2BGRA)

    # 2. Compute proportional layout without distortion
    max_w, max_h = 1080, 620
    img_h, img_w = raw_img.shape[:2]
    scale_fit = min(max_w / img_w, max_h / img_h, 1.0)
    target_w = int(img_w * scale_fit)
    target_h = int(img_h * scale_fit)

    card_base = cv2.resize(raw_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    shadow_base = build_drop_shadow(card_base, blur_ksize=45, opacity=0.40)

    # 3. Render headline banner if text is provided
    banner_np = None
    if headline_data and any(headline_data.values()):
        banner_pil = render_headline_banner(headline_data, canvas_w=CANVAS_W, banner_h=230)
        banner_np = cv2.cvtColor(np.array(banner_pil), cv2.COLOR_RGBA2BGRA)

    # 4. Open looping background video
    bg_cap = cv2.VideoCapture(bg_video_path)
    if not bg_cap.isOpened():
        raise RuntimeError(f"Cannot open background video: {bg_video_path}")

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (CANVAS_W, CANVAS_H))

    # Center placement for image card
    center_x = CANVAS_W // 2
    center_y = 660  # Pushed down to balance with top headline banner

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # Read background loop
        ret, bg_frame = bg_cap.read()
        if not ret:
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bg_frame = bg_cap.read()
            if not ret:
                bg_frame = np.full((CANVAS_H, CANVAS_W, 3), 245, dtype=np.uint8)

        if (bg_frame.shape[1], bg_frame.shape[0]) != (CANVAS_W, CANVAS_H):
            bg_frame = cv2.resize(bg_frame, (CANVAS_W, CANVAS_H))

        # Stamp drop animation curve (Elastic slam from scale 1.4 -> 0.96 -> 1.0)
        slam_duration = 0.38
        if t < slam_duration:
            progress = t / slam_duration
            # Overshoot elastic bounce
            scale_factor = 1.0 + 0.45 * math.exp(-6.0 * progress) * math.cos(progress * math.pi * 2.2)
            alpha_stamp = min(1.0, progress * 2.5)
        else:
            # Subtle smooth breathing drift
            drift_t = t - slam_duration
            scale_factor = 1.0 + 0.015 * math.sin(drift_t * 1.8)
            alpha_stamp = 1.0

        # Scale card
        cur_w = int(target_w * scale_factor)
        cur_h = int(target_h * scale_factor)
        if cur_w > 0 and cur_h > 0:
            cur_card = cv2.resize(card_base, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
            cur_shadow = cv2.resize(shadow_base, (cur_w + 80, cur_h + 80), interpolation=cv2.INTER_LINEAR)

            x1 = center_x - cur_w // 2
            y1 = center_y - cur_h // 2
            sx1 = x1 - 40
            sy1 = y1 - 30  # Shadow cast down

            # Composite Shadow
            _alpha_blend(bg_frame, cur_shadow, sx1, sy1, alpha_stamp * 0.8)
            # Composite Card
            _alpha_blend(bg_frame, cur_card, x1, y1, alpha_stamp)

        # Composite Top Headline Banner
        if banner_np is not None:
            _alpha_blend(bg_frame, banner_np, 0, 10, 1.0)

        out.write(bg_frame)

    bg_cap.release()
    out.release()
    return output_path


def _alpha_blend(dst: np.ndarray, src_rgba: np.ndarray, x: int, y: int, opacity: float = 1.0):
    """Clips and blends an RGBA layer onto a BGR canvas with alpha."""
    dh, dw = dst.shape[:2]
    sh, sw = src_rgba.shape[:2]

    # Bounding box intersection
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(dw, x + sw), min(dh, y + sh)
    if x1 >= x2 or y1 >= y2:
        return

    src_x1 = x1 - x
    src_y1 = y1 - y
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    src_crop = src_rgba[src_y1:src_y2, src_x1:src_x2]
    dst_crop = dst[y1:y2, x1:x2]

    alpha = (src_crop[:, :, 3].astype(np.float32) / 255.0) * opacity
    alpha_3d = np.dstack([alpha, alpha, alpha])

    dst[y1:y2, x1:x2] = (src_crop[:, :, :3] * alpha_3d + dst_crop * (1.0 - alpha_3d)).astype(np.uint8)
