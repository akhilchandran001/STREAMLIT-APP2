import glob
import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Theme Palettes
COLOR_TEXT_BLACK = (20, 20, 20)       # #141414 For BG1 & BG3
COLOR_TEXT_WHITE = (245, 245, 245)   # For all other / dark backgrounds
COLOR_ACCENT_RED = (220, 38, 68)     # #DC2644 Crimson Red
COLOR_BAR_RED    = (220, 38, 68)

CANVAS_W = 1920
CANVAS_H = 1080


def _load_font_or_fallback(is_italic: bool = False, size: int = 72) -> tuple[ImageFont.FreeTypeFont, bool]:
    """
    Checks assets/fonts/ first for user fonts, then system paths.
    Returns (font, needs_synthetic_shear).
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

        # Check system italic candidates
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

    # Bold / Regular candidates
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


def _draw_forward_italic_word(
    word: str,
    font: ImageFont.FreeTypeFont,
    fill_color: tuple,
    slant: float = 0.24
) -> tuple[Image.Image, int, int]:
    """
    Renders text and applies a positive forward affine shear transformation
    so italics lean forward to the right (standard typography).
    """
    dummy = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), word, font=font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])

    pad = int(th * abs(slant) + 20)
    img_word = Image.new("RGBA", (tw + pad * 2, th + 24), (255, 255, 255, 0))
    draw_w = ImageDraw.Draw(img_word)
    draw_w.text((pad - bbox[0], 12 - bbox[1]), word, font=font, fill=fill_color)

    # Positive shear for forward slant: x_src = x_dst + slant * (h - y_dst)
    w_orig, h_orig = img_word.size
    affine_matrix = (1.0, slant, -slant * h_orig, 0.0, 1.0, 0.0)
    sheared = img_word.transform(
        (w_orig + pad, h_orig),
        Image.AFFINE,
        affine_matrix,
        resample=Image.BICUBIC
    )

    alpha = np.array(sheared)[:, :, 3]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    if len(cols) > 0:
        sheared = sheared.crop((cols[0], 0, cols[-1] + 1, h_orig))
    return sheared, sheared.width, sheared.height


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
    banner_h: int = 150
) -> tuple[Image.Image, int]:
    """
    Renders styled headline banner with contextual text coloring:
    - Black text for BG1 and BG3
    - White text for all other backgrounds
    - Crimson Red for the highlighted italic word + anchored underline bar
    """
    prefix, highlight, suffix = parse_headline_input(headline_data)
    banner = Image.new("RGBA", (canvas_w, banner_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(banner)

    # Contextual color switching
    bg_name = os.path.basename(bg_video_path).upper() if bg_video_path else ""
    if "BG1" in bg_name or "BG3" in bg_name:
        text_color = COLOR_TEXT_BLACK
    else:
        text_color = COLOR_TEXT_WHITE

    font_base, _ = _load_font_or_fallback(is_italic=False, size=68)
    font_italic, needs_shear = _load_font_or_fallback(is_italic=True, size=70)

    def measure(txt: str, fnt):
        if not txt:
            return 0
        b = draw.textbbox((0, 0), txt, font=fnt)
        return b[2] - b[0]

    space_w = 18
    w_prefix = measure(prefix, font_base) if prefix else 0
    w_suffix = measure(suffix, font_base) if suffix else 0

    highlight_img = None
    if highlight:
        if needs_shear:
            highlight_img, w_highlight, _ = _draw_forward_italic_word(
                highlight, font_base, COLOR_ACCENT_RED + (255,), slant=0.24
            )
        else:
            w_highlight = measure(highlight, font_italic)
    else:
        w_highlight = 0

    total_w = w_prefix + (space_w if prefix and (highlight or suffix) else 0) + \
              w_highlight + (space_w if highlight and suffix else 0) + w_suffix
    start_x = max(20, (canvas_w - total_w) // 2)
    baseline_y = 22

    cur_x = start_x
    if prefix:
        draw.text((cur_x, baseline_y), prefix, font=font_base, fill=text_color + (255,))
        cur_x += w_prefix + space_w

    highlight_x_start = cur_x
    if highlight:
        if highlight_img:
            banner.paste(highlight_img, (cur_x, baseline_y - 6), highlight_img)
        else:
            draw.text((cur_x, baseline_y), highlight, font=font_italic, fill=COLOR_ACCENT_RED + (255,))
        cur_x += w_highlight + space_w

    if suffix:
        draw.text((cur_x, baseline_y), suffix, font=font_base, fill=text_color + (255,))

    # Underline bar anchored specifically beneath the highlighted word
    bar_width = max(90, w_highlight + 14)
    bar_x1 = highlight_x_start + (w_highlight - bar_width) // 2 if highlight else (canvas_w - bar_width) // 2
    bar_x2 = bar_x1 + bar_width
    bar_y = baseline_y + 80
    draw.rounded_rectangle([bar_x1, bar_y, bar_x2, bar_y + 5], radius=3, fill=COLOR_BAR_RED + (255,))

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

    # Rounded rectangle mask
    cv2.rectangle(mask, (radius, 0), (target_w - radius, target_h), 255, -1)
    cv2.rectangle(mask, (0, radius), (target_w, target_h - radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (target_w - radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, target_h - radius), radius, 255, -1)
    cv2.circle(mask, (target_w - radius, target_h - radius), radius, 255, -1)

    card[mask > 0, :3] = (248, 248, 248)
    card[mask > 0, 3] = 255
    card[border_px:border_px + inner_h, border_px:border_px + inner_w, :3] = scaled_photo

    # Subtle inner bevel outline
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


def build_drop_shadow(card_rgba: np.ndarray, blur_radius: int = 39, opacity: float = 0.50) -> np.ndarray:
    """Builds soft feathered drop shadow underneath the card."""
    h, w = card_rgba.shape[:2]
    alpha = card_rgba[:, :, 3]

    pad = 70
    shadow = np.zeros((h + pad * 2, w + pad * 2, 4), dtype=np.uint8)
    shadow[pad:pad + h, pad:pad + w, 3] = (alpha.astype(np.float32) * opacity).astype(np.uint8)
    shadow[:, :, :3] = (12, 12, 12)

    ksize = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    return cv2.GaussianBlur(shadow, (ksize, ksize), 0)


def apply_card_specular_sheen(card_rgba: np.ndarray, progress: float) -> np.ndarray:
    """Simulates diagonal sunlight gloss sweep across the card."""
    if progress < 0.0 or progress > 1.0:
        return card_rgba

    h, w = card_rgba.shape[:2]
    center_x = int(progress * (w + 400) - 200)

    # Lightweight grid evaluation
    step = 4
    grid_y, grid_x = np.mgrid[0:h:step, 0:w:step].astype(np.float32)
    band = (grid_x - grid_y * 0.42) - center_x
    sheen = np.exp(-((band / 70.0) ** 2)) * 45.0

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
    - Maximum width and height card layout
    - Contextual text color matching background
    - Kinetic elastic slam (first 0.38s) followed by smooth continuous floating drift
    - Tactile glass sheen sweep and cinematic vignette overlay
    """
    total_frames = max(1, int(round(duration * fps)))

    raw_img = cv2.imread(foreground_image_path)
    if raw_img is None:
        raise ValueError(f"Could not load foreground image: {foreground_image_path}")

    # Render headline banner
    banner_np = None
    underline_bar_bottom_y = 95
    if headline_data:
        banner_pil, underline_bar_bottom_y = render_headline_banner(
            headline_data=headline_data,
            bg_video_path=bg_video_path,
            canvas_w=CANVAS_W,
            banner_h=140
        )
        banner_np = cv2.cvtColor(np.array(banner_pil), cv2.COLOR_RGBA2BGRA)

    # Maximum image dimensions
    max_w, max_h = 1720, 860
    img_h, img_w = raw_img.shape[:2]
    scale_fit = min(max_w / img_w, max_h / img_h, 1.0)
    target_w = int(img_w * scale_fit)
    target_h = int(img_h * scale_fit)

    card_base = build_card_with_overlay(raw_img, target_w, target_h)
    shadow_base = build_drop_shadow(card_base, blur_radius=39, opacity=0.50)
    vignette = build_vignette_mask(CANVAS_W, CANVAS_H)

    # Position image card ~24px below underline bar
    center_x = CANVAS_W // 2
    base_center_y = min(CANVAS_H - target_h // 2 - 16, underline_bar_bottom_y + 24 + target_h // 2)

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
                bg_frame = np.full((CANVAS_H, CANVAS_W, 3), 20, dtype=np.uint8)

        if (bg_frame.shape[1], bg_frame.shape[0]) != (CANVAS_W, CANVAS_H):
            bg_frame = cv2.resize(bg_frame, (CANVAS_W, CANVAS_H))

        # Dynamic Kinematics: 0.38s Elastic Slam -> Continuous Ambient Floating Drift
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
            # Ambient hovering drift
            drift_t = t - 0.38
            float_y = 5.0 * math.sin(drift_t * 1.6)
            float_roll = 0.35 * math.sin(drift_t * 1.2)
            alpha_stamp = 1.0
            cw, ch = target_w, target_h

            # Gentle card transformation
            M_card = cv2.getRotationMatrix2D((cw * 0.5, ch * 0.5), float_roll, 1.0)
            cur_card = cv2.warpAffine(card_base, M_card, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cur_shadow = shadow_base

        # Specular sheen sweep between 35% and 75% of scene duration
        sheen_prog = (t - 0.35 * duration) / max(0.40 * duration, 0.1)
        if 0.0 <= sheen_prog <= 1.0:
            cur_card = apply_card_specular_sheen(cur_card, sheen_prog)

        cur_center_y = int(base_center_y + float_y)
        x1 = center_x - cw // 2
        y1 = cur_center_y - ch // 2
        sx1 = x1 - 70
        sy1 = y1 - 50 + int(float_y * 0.5)

        # Composite Drop Shadow & Framed Card
        _fast_roi_blend(bg_frame, cur_shadow, sx1, sy1, alpha_stamp * 0.8)
        _fast_roi_blend(bg_frame, cur_card, x1, y1, alpha_stamp)

        # Composite Headline Banner
        if banner_np is not None:
            _fast_roi_blend(bg_frame, banner_np, 0, 10, 1.0)

        # Final Vignette Overlay
        bg_frame = np.clip(bg_frame.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

        out.write(bg_frame)

    bg_cap.release()
    out.release()
    return output_path
