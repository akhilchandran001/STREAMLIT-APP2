import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Theme Palette (matches reference card)
COLOR_TEXT_BLACK = (20, 20, 20)      # #141414 Dark Charcoal
COLOR_ACCENT_RED = (200, 29, 59)     # #C81D3B Crimson Red
COLOR_BAR_RED = (200, 29, 59)
CANVAS_W = 1920
CANVAS_H = 1080


def _get_system_font(size: int = 72) -> ImageFont.FreeTypeFont:
    """Finds any standard bold sans font on Linux/macOS/Windows, or falls back to Pillow default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _draw_italic_word(
    word: str,
    font: ImageFont.FreeTypeFont,
    fill_color: tuple,
    slant: float = 0.22
) -> tuple[Image.Image, int, int]:
    """
    Renders text and applies an affine shear transformation to guarantee
    true, uniform italics at any font size even without an italic font file.
    """
    dummy = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0, 0), word, font=font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])

    pad = int(th * abs(slant) + 16)
    img_word = Image.new("RGBA", (tw + pad * 2, th + 20), (255, 255, 255, 0))
    draw_w = ImageDraw.Draw(img_word)
    draw_w.text((pad - bbox[0], 10 - bbox[1]), word, font=font, fill=fill_color)

    # Affine shear matrix: x' = x - slant * (y - baseline)
    w_orig, h_orig = img_word.size
    affine_matrix = (1.0, -slant, slant * (h_orig / 2), 0.0, 1.0, 0.0)
    sheared = img_word.transform(
        (w_orig + 10, h_orig),
        Image.AFFINE,
        affine_matrix,
        resample=Image.BICUBIC
    )

    alpha = np.array(sheared)[:, :, 3]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    if len(cols) > 0:
        left, right = cols[0], cols[-1]
        sheared = sheared.crop((left, 0, right + 1, h_orig))
    return sheared, sheared.width, sheared.height


def parse_headline_input(headline_input) -> tuple[str, str, str]:
    """
    Normalizes strings or dicts into (prefix, highlight_word, suffix).
    Examples:
      "Solar Journey" -> ("Solar", "Journey", "")
      "For better experience" -> ("For", "better", "experience")
    """
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
        # For longer titles: first word prefix, second word highlight, rest suffix
        return words[0], words[1], " ".join(words[2:])


def render_headline_banner(
    headline_data,
    canvas_w: int = CANVAS_W,
    banner_h: int = 160
) -> tuple[Image.Image, int]:
    """
    Renders styled headline typography with the crimson accent line placed
    directly underneath the highlighted word.
    Returns (banner_image, underline_bar_bottom_y).
    """
    prefix, highlight, suffix = parse_headline_input(headline_data)
    banner = Image.new("RGBA", (canvas_w, banner_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(banner)

    font_base = _get_system_font(size=72)

    def measure(txt: str):
        if not txt:
            return 0
        b = draw.textbbox((0, 0), txt, font=font_base)
        return b[2] - b[0]

    space_w = 20
    w_prefix = measure(prefix) if prefix else 0
    w_suffix = measure(suffix) if suffix else 0

    sheared_img, w_highlight, h_highlight = (None, 0, 0)
    if highlight:
        sheared_img, w_highlight, h_highlight = _draw_italic_word(
            highlight,
            font_base,
            COLOR_ACCENT_RED + (255,),
            slant=0.22
        )

    total_w = w_prefix + (space_w if prefix and (highlight or suffix) else 0) + \
              w_highlight + (space_w if highlight and suffix else 0) + w_suffix
    start_x = max(20, (canvas_w - total_w) // 2)
    baseline_y = 28

    cur_x = start_x
    if prefix:
        draw.text((cur_x, baseline_y), prefix, font=font_base, fill=COLOR_TEXT_BLACK + (255,))
        cur_x += w_prefix + space_w

    highlight_x_start = cur_x
    if sheared_img:
        banner.paste(sheared_img, (cur_x, baseline_y - 8), sheared_img)
        cur_x += w_highlight + space_w

    if suffix:
        draw.text((cur_x, baseline_y), suffix, font=font_base, fill=COLOR_TEXT_BLACK + (255,))

    # Draw crimson underline bar anchored directly beneath the highlighted word
    bar_width = max(110, w_highlight + 12)
    bar_x1 = highlight_x_start + (w_highlight - bar_width) // 2 if highlight else (canvas_w - bar_width) // 2
    bar_x2 = bar_x1 + bar_width
    bar_y = baseline_y + 82
    draw.rounded_rectangle([bar_x1, bar_y, bar_x2, bar_y + 6], radius=3, fill=COLOR_BAR_RED + (255,))

    return banner, bar_y + 6


def build_card_with_overlay(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Wraps the image in a die-cut rounded white card frame with inner highlights."""
    border_px = 16
    radius = 24
    inner_w = target_w - border_px * 2
    inner_h = target_h - border_px * 2

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

    card[mask > 0, :3] = (250, 250, 250)
    card[mask > 0, 3] = 255

    # Insert scaled photo
    card[border_px:border_px + inner_h, border_px:border_px + inner_w, :3] = scaled_photo

    # Subtle 1px crisp outline
    edges = cv2.Canny(mask, 100, 200)
    card[edges > 0, :3] = (220, 220, 220)

    return card


def build_drop_shadow(card_rgba: np.ndarray, blur_radius: int = 35, opacity: float = 0.40) -> np.ndarray:
    """Precomputes a soft feathered drop shadow underneath the card."""
    h, w = card_rgba.shape[:2]
    alpha = card_rgba[:, :, 3]

    pad = 60
    shadow = np.zeros((h + pad * 2, w + pad * 2, 4), dtype=np.uint8)
    shadow[pad:pad + h, pad:pad + w, 3] = (alpha.astype(np.float32) * opacity).astype(np.uint8)
    shadow[:, :, :3] = (15, 15, 15)

    ksize = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    return cv2.GaussianBlur(shadow, (ksize, ksize), 0)


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
    - Large die-cut framed card with minimal gap below headline
    - Elastic slam down in first 0.38s, stable resting float for remaining duration
    - Fast ROI rendering pipeline
    """
    total_frames = max(1, int(round(duration * fps)))

    raw_img = cv2.imread(foreground_image_path)
    if raw_img is None:
        raise ValueError(f"Could not load foreground image: {foreground_image_path}")

    # Render headline banner and get underline coordinate
    banner_np = None
    underline_bar_bottom_y = 100
    if headline_data:
        banner_pil, underline_bar_bottom_y = render_headline_banner(headline_data, canvas_w=CANVAS_W, banner_h=150)
        banner_np = cv2.cvtColor(np.array(banner_pil), cv2.COLOR_RGBA2BGRA)

    # Sizing: Larger card with minimal gap from heading
    max_w, max_h = 1460, 740
    img_h, img_w = raw_img.shape[:2]
    scale_fit = min(max_w / img_w, max_h / img_h, 1.0)
    target_w = int(img_w * scale_fit)
    target_h = int(img_h * scale_fit)

    card_base = build_card_with_overlay(raw_img, target_w, target_h)
    shadow_base = build_drop_shadow(card_base, blur_radius=35, opacity=0.45)

    # Tight spacing: Image card starts ~36px below the headline underline bar
    center_x = CANVAS_W // 2
    center_y = min(CANVAS_H - target_h // 2 - 20, underline_bar_bottom_y + 36 + target_h // 2)

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
                bg_frame = np.full((CANVAS_H, CANVAS_W, 3), 245, dtype=np.uint8)

        if (bg_frame.shape[1], bg_frame.shape[0]) != (CANVAS_W, CANVAS_H):
            bg_frame = cv2.resize(bg_frame, (CANVAS_W, CANVAS_H))

        # Kinematic timeline: 0.38s elastic bounce -> rest
        if frame_idx < slam_frames:
            prog = frame_idx / max(slam_frames, 1)
            scale = 1.0 + 0.35 * math.exp(-6.0 * prog) * math.cos(prog * math.pi * 2.2)
            alpha_stamp = min(1.0, prog * 2.5)
            cw, ch = max(10, int(target_w * scale)), max(10, int(target_h * scale))
            cur_card = cv2.resize(card_base, (cw, ch), interpolation=cv2.INTER_LINEAR)
            cur_shadow = cv2.resize(shadow_base, (cw + 120, ch + 120), interpolation=cv2.INTER_LINEAR)
        else:
            cur_card = card_base
            cur_shadow = shadow_base
            cw, ch = target_w, target_h
            alpha_stamp = 1.0

        x1 = center_x - cw // 2
        y1 = center_y - ch // 2
        sx1 = x1 - 60
        sy1 = y1 - 45  # Downward shadow cast

        # Composite Drop Shadow and Die-Cut Card
        _fast_roi_blend(bg_frame, cur_shadow, sx1, sy1, alpha_stamp * 0.8)
        _fast_roi_blend(bg_frame, cur_card, x1, y1, alpha_stamp)

        # Composite Headline Banner
        if banner_np is not None:
            _fast_roi_blend(bg_frame, banner_np, 0, 8, 1.0)

        out.write(bg_frame)

    bg_cap.release()
    out.release()
    return output_path
