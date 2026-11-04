import glob
import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# High-Intensity Cinema Color Palette
COLOR_PURE_BLACK   = (0, 0, 0)          # #000000
COLOR_PURE_WHITE   = (255, 255, 255)    # #FFFFFF
COLOR_VIBRANT_RED  = (235, 15, 45)      # #EB0F2D High-Intensity Crimson Red

CANVAS_W = 1920
CANVAS_H = 1080


def _load_font_or_fallback(is_italic: bool = False, size: int = 70) -> tuple[ImageFont.FreeTypeFont, bool]:
    """
    Loads custom user fonts from assets/fonts/ first, then checks system fallbacks.
    Returns (font_object, needs_synthetic_shear).
    """
    font_dir = os.path.join(os.getcwd(), "assets", "fonts")
    if is_italic:
        user_italic = glob.glob(os.path.join(font_dir, "*[iI]talic*.ttf")) + \
                      glob.glob(os.path.join(font_dir, "*[iI]talic*.otf")) + \
                      [os.path.join(font_dir, "Italic.ttf")]
        for p in user_italic:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size), False
                except Exception:
                    pass

        sys_italic = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
            "C:\\Windows\\Fonts\\arialbi.ttf",
            "/Library/Fonts/Arial Bold Italic.ttf"
        ]
        for p in sys_italic:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size), False
                except Exception:
                    pass

    user_bold = glob.glob(os.path.join(font_dir, "*[bB]old*.ttf")) + \
                glob.glob(os.path.join(font_dir, "*[bB]old*.otf")) + \
                [os.path.join(font_dir, "Bold.ttf")]
    for p in user_bold:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size), is_italic
            except Exception:
                pass

    sys_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "/Library/Fonts/Arial Bold.ttf"
    ]
    for p in sys_bold:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size), is_italic
            except Exception:
                pass

    try:
        return ImageFont.load_default(size=size), is_italic
    except Exception:
        return ImageFont.load_default(), is_italic


def parse_headline_input(headline_input) -> tuple[str, str, str]:
    """Normalizes string or dictionary input into (prefix, highlight, suffix)."""
    if not headline_input:
        return "", "", ""

    if isinstance(headline_input, dict):
        if "prefix" in headline_input or "highlight" in headline_input or "suffix" in headline_input:
            return (
                str(headline_input.get("prefix", "")).strip(),
                str(headline_input.get("highlight", "")).strip(),
                str(headline_input.get("suffix", "")).strip()
            )
        raw_text = str(headline_input.get("title") or headline_input.get("text") or "").strip()
    else:
        raw_text = str(headline_input).strip()

    if not raw_text:
        return "", "", ""

    words = raw_text.split()
    if len(words) == 1:
        return "", words[0], ""
    elif len(words) == 2:
        return words[0], words[1], ""
    elif len(words) == 3:
        return words[0], words[1], words[2]
    else:
        return words[0], words[1], " ".join(words[2:])


def render_headline_banner(
    headline_data,
    bg_video_path: str = "",
    canvas_w: int = CANVAS_W,
    banner_h: int = 140
) -> tuple[Image.Image, int]:
    """
    Renders styled headline banner:
    - High-intensity black/white text with intense red highlight.
    - Mathematical baseline alignment so all three words sit on the exact same horizontal line.
    - Contextual background plate (pure white for BG1/BG3, deep black for others).
    """
    prefix, highlight, suffix = parse_headline_input(headline_data)
    banner = Image.new("RGBA", (canvas_w, banner_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(banner)

    # Determine theme based on background filename
    bg_name = os.path.basename(bg_video_path).upper() if bg_video_path else ""
    is_light_bg = ("BG1" in bg_name) or ("BG3" in bg_name)

    if is_light_bg:
        text_color = COLOR_PURE_BLACK
        plate_color = (255, 255, 255, 240)       # High-opacity Pure White Plate
        plate_border = (220, 220, 220, 255)
    else:
        text_color = COLOR_PURE_WHITE
        plate_color = (0, 0, 0, 230)             # High-opacity Pitch Black Plate
        plate_border = (50, 50, 50, 255)

    font_base, _ = _load_font_or_fallback(is_italic=False, size=68)
    font_italic, needs_shear = _load_font_or_fallback(is_italic=True, size=70)

    def get_width(txt: str, fnt):
        if not txt:
            return 0
        b = draw.textbbox((0, 0), txt, font=fnt)
        return b[2] - b[0]

    space_w = 20
    w_prefix = get_width(prefix, font_base) if prefix else 0
    w_highlight = get_width(highlight, font_italic) if highlight else 0
    w_suffix = get_width(suffix, font_base) if suffix else 0

    total_text_w = w_prefix + (space_w if prefix and (highlight or suffix) else 0) + \
                   w_highlight + (space_w if highlight and suffix else 0) + w_suffix

    # Mathematical Baseline Lock:
    # Uses capital 'H' baseline offset so Montserrat and Playfair sit on the exact same pixel line
    target_baseline_y = 66
    bbox_base_h = font_base.getbbox("H")
    bbox_italic_h = font_italic.getbbox("H")

    y_base = target_baseline_y - bbox_base_h[3]
    y_italic = target_baseline_y - bbox_italic_h[3]

    # Draw Background Contrast Plate
    plate_pad_x = 42
    plate_pad_y = 12
    plate_w = total_text_w + plate_pad_x * 2
    plate_h = 92
    plate_x1 = (canvas_w - plate_w) // 2
    plate_y1 = max(6, target_baseline_y - 52)
    plate_x2 = plate_x1 + plate_w
    plate_y2 = plate_y1 + plate_h

    # Rounded background plate
    draw.rounded_rectangle(
        [plate_x1, plate_y1, plate_x2, plate_y2],
        radius=18,
        fill=plate_color,
        outline=plate_border,
        width=2
    )

    cur_x = plate_x1 + plate_pad_x

    # 1. Prefix
    if prefix:
        draw.text((cur_x, y_base), prefix, font=font_base, fill=text_color + (255,))
        cur_x += w_prefix + space_w

    # 2. Highlight (Single-line baseline locked)
    highlight_start_x = cur_x
    if highlight:
        draw.text((cur_x, y_italic), highlight, font=font_italic, fill=COLOR_VIBRANT_RED + (255,))
        cur_x += w_highlight + space_w

    # 3. Suffix
    if suffix:
        draw.text((cur_x, y_base), suffix, font=font_base, fill=text_color + (255,))

    # 4. Underline bar anchored directly below the highlighted word
    bar_width = max(90, w_highlight + 14)
    bar_x1 = highlight_start_x + (w_highlight - bar_width) // 2 if highlight else (canvas_w - bar_width) // 2
    bar_x2 = bar_x1 + bar_width
    bar_y = target_baseline_y + 14
    draw.rounded_rectangle([bar_x1, bar_y, bar_x2, bar_y + 5], radius=3, fill=COLOR_VIBRANT_RED + (255,))

    return banner, bar_y + 5


def build_card_with_overlay(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Wraps photo in a die-cut rounded card with inner bevel highlight."""
    border_px = 14
    radius = 24
    inner_w = max(10, target_w - border_px * 2)
    inner_h = max(10, target_h - border_px * 2)

    scaled_photo = cv2.resize(img_bgr, (inner_w, inner_h), interpolation=cv2.INTER_AREA)

    card = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    mask = np.zeros((target_h, target_w), dtype=np.uint8)

    cv2.rectangle(mask, (radius, 0), (target_w - radius, target_h), 255, -1)
    cv2.rectangle(mask, (0, radius), (target_w, target_h - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (target_w - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, target_h - radius), radius, 255, -1)
    cv2.circle(mask, (target_w - radius, target_h - radius), radius, 255, -1)

    card[mask > 0, :3] = (255, 255, 255)
    card[mask > 0, 3] = 255
    card[border_px:border_px + inner_h, border_px:border_px + inner_w, :3] = scaled_photo

    edges = cv2.Canny(mask, 100, 200)
    card[edges > 0, :3] = (220, 220, 220)
    return card


def build_vignette_mask(width: int = CANVAS_W, height: int = CANVAS_H) -> np.ndarray:
    """Generates a smooth 2D radial falloff matrix."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    radial_dist = (xx**2) + ((yy * 1.25) ** 2)
    return np.clip(1.0 - 0.32 * radial_dist, 0.0, 1.0)[:, :, np.newaxis]


def build_drop_shadow(card_rgba: np.ndarray, blur_radius: int = 41, opacity: float = 0.55) -> np.ndarray:
    """Builds a soft feathered drop shadow underneath the card."""
    h, w = card_rgba.shape[:2]
    alpha = card_rgba[:, :, 3]

    pad = 70
    shadow = np.zeros((h + pad * 2, w + pad * 2, 4), dtype=np.uint8)
    shadow[pad:pad + h, pad:pad + w, 3] = (alpha.astype(np.float32) * opacity).astype(np.uint8)
    shadow[:, :, :3] = (5, 5, 5)

    ksize = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    return cv2.GaussianBlur(shadow, (ksize, ksize), 0)


def apply_card_specular_sheen(card_rgba: np.ndarray, progress: float) -> np.ndarray:
    """Simulates a diagonal sunlight gloss sweep across the card."""
    if progress < 0.0 or progress > 1.0:
        return card_rgba

    h, w = card_rgba.shape[:2]
    center_x = int(progress * (w + 500) - 250)

    step = 4
    grid_y, grid_x = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
    band = (grid_x - grid_y * 0.45) - center_x
    sheen = np.exp(-((band / 75.0) ** 2)) * 50.0

    sheen_full = cv2.resize(sheen, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (card_rgba[:, :, 3] > 0).astype(np.float32)

    result = card_rgba.copy()
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c].astype(np.float32) + sheen_full * mask, 0, 255).astype(np.uint8)
    return result


def _fast_roi_blend(dst: np.ndarray, src_rgba: np.ndarray, x: int, y: int, opacity: float = 1.0):
    """Blends RGBA onto BGR using fast integer math on bounding-box ROIs."""
    dh, dw = dst.shape[:2]
    sh, sw = src_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(dw, x + sw), min(dh, y + sh)
    if x1 >= x2 or y1 >= y2:
        return

    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

    src_crop = src_rgba[sy1:sy2, sx1:sx2]
    dst_crop = dst[y1:y2, x1:x2]

    alpha = ((src_crop[:, :, 3].astype(np.uint16) * int(opacity * 256)) >> 8)[:, :, np.newaxis]
    inv_alpha = 256 - alpha

    dst[y1:y2, x1:x2] = (
        (src_crop[:, :, :3].astype(np.uint16) * alpha + dst_crop.astype(np.uint16) * inv_alpha) >> 8
    ).astype(np.uint8)


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
    - Maximum dimensions (1760x870).
    - Perfect horizontal baseline alignment between regular and italic fonts.
    - Intense colors with contextual background plate.
    - Dynamic floating hover and specular glass sheen.
    """
    total_frames = max(1, int(round(duration * fps)))

    raw_img = cv2.imread(foreground_image_path)
    if raw_img is None:
        raise ValueError(f"Could not load foreground image: {foreground_image_path}")

    # Render headline banner and get underline coordinate
    banner_np = None
    underline_bar_bottom_y = 96
    if headline_data:
        banner_pil, underline_bar_bottom_y = render_headline_banner(
            headline_data=headline_data,
            bg_video_path=bg_video_path,
            canvas_w=CANVAS_W,
            banner_h=135
        )
        banner_np = cv2.cvtColor(np.array(banner_pil), cv2.COLOR_RGBA2BGRA)

    # Maximum image dimensions
    max_w, max_h = 1760, 870
    img_h, img_w = raw_img.shape[:2]
    scale_fit = min(max_w / img_w, max_h / img_h, 1.0)
    target_w = int(img_w * scale_fit)
    target_h = int(img_h * scale_fit)

    card_base = build_card_with_overlay(raw_img, target_w, target_h)
    shadow_base = build_drop_shadow(card_base, blur_radius=41, opacity=0.55)
    vignette = build_vignette_mask(CANVAS_W, CANVAS_H)

    # Tight spacing: starts ~18px below underline bar
    center_x = CANVAS_W // 2
    base_center_y = min(CANVAS_H - target_h // 2 - 12, underline_bar_bottom_y + 18 + target_h // 2)

    bg_cap = cv2.VideoCapture(bg_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (CANVAS_W, CANVAS_H))

    slam_frames = int(round(0.38 * fps))

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        ret, bg_frame = bg_cap.read()
        if not ret:
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bg_frame = bg_cap.read()
            if not ret:
                bg_frame = np.full((CANVAS_H, CANVAS_W, 3), 15, dtype=np.uint8)

        if (bg_frame.shape[1], bg_frame.shape[0]) != (CANVAS_W, CANVAS_H):
            bg_frame = cv2.resize(bg_frame, (CANVAS_W, CANVAS_H))

        # 1. Apply Vignette ONLY to the Background Frame (Preserves text and card color intensity)
        bg_frame = np.clip(bg_frame.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

        # 2. Dynamic Kinematics: Elastic Slam -> Continuous Ambient Floating Drift
        if frame_idx < slam_frames:
            prog = frame_idx / max(slam_frames, 1)
            scale = 1.0 + 0.32 * math.exp(-6.0 * prog) * math.cos(prog * math.pi * 2.2)
            alpha_stamp = min(1.0, prog * 2.5)
            float_y = 0.0
            float_roll = 0.0
            cw = max(10, int(target_w * scale))
            ch = max(10, int(target_h * scale))
            cur_card = cv2.resize(card_base, (cw, ch), interpolation=cv2.INTER_LINEAR)
            cur_shadow = cv2.resize(shadow_base, (cw + 140, ch + 140), interpolation=cv2.INTER_LINEAR)
        else:
            drift_t = t - 0.38
            float_y = 6.0 * math.sin(drift_t * 1.5)
            float_roll = 0.35 * math.sin(drift_t * 1.1)
            alpha_stamp = 1.0
            cw, ch = target_w, target_h

            # Subtle hovering rotation
            M_card = cv2.getRotationMatrix2D((cw * 0.5, ch * 0.5), float_roll, 1.0)
            cur_card = cv2.warpAffine(card_base, M_card, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cur_shadow = shadow_base

        # 3. Specular Sheen Sweep
        sheen_prog = (t - 0.30 * duration) / max(0.40 * duration, 0.1)
        if 0.0 <= sheen_prog <= 1.0:
            cur_card = apply_card_specular_sheen(cur_card, sheen_prog)

        cur_center_y = int(base_center_y + float_y)
        x1 = center_x - cw // 2
        y1 = cur_center_y - ch // 2
        sx1 = x1 - 70
        sy1 = y1 - 48 + int(float_y * 0.5)

        # 4. Composite Drop Shadow and Card Over Background
        _fast_roi_blend(bg_frame, cur_shadow, sx1, sy1, alpha_stamp * 0.8)
        _fast_roi_blend(bg_frame, cur_card, x1, y1, alpha_stamp)

        # 5. Composite Headline Banner on Top with Maximum Saturation
        if banner_np is not None:
            _fast_roi_blend(bg_frame, banner_np, 0, 10, 1.0)

        out.write(bg_frame)

    bg_cap.release()
    out.release()
    return output_path
