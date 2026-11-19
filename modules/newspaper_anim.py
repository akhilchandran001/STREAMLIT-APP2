import datetime
import math
import os
import re
import subprocess
import shutil
import wave
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MODULE_DIR)

NEWSPAPER_ASSETS_DIR = os.path.join(REPO_ROOT, "assets", "newspaper")
LOGOS_DIR = os.path.join(NEWSPAPER_ASSETS_DIR, "logos")
FONTS_DIR = os.path.join(REPO_ROOT, "assets", "fonts")

os.makedirs(NEWSPAPER_ASSETS_DIR, exist_ok=True)
os.makedirs(LOGOS_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)

VIDEO_W, VIDEO_H = 1920, 1080
SAMPLE_RATE = 44100


# ==========================================
# 1. ASSET & FONT RESOLVERS
# ==========================================
def resolve_table_path(custom_path: str = None) -> str | None:
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    for name in ("table.png", "table.jpg", "table.jpeg"):
        candidates.append(os.path.join(NEWSPAPER_ASSETS_DIR, name))
        candidates.append(os.path.join(REPO_ROOT, "assets", name))
        candidates.append(os.path.join(REPO_ROOT, name))
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def resolve_logo_path(newspaper_id: str) -> str | None:
    nid = (newspaper_id or "manorama").lower().strip()
    if "mathru" in nid:
        names = ["Mathrubhumi.png", "mathrubhumi.png", "mathrubhumi.jpg"]
    else:
        names = [
            "Malayala Manorma.png",
            "Malayala Manorama.png",
            "manorama.png",
            "manorama.jpg",
        ]
    for name in names:
        for base in (
            os.path.join(REPO_ROOT, "assets", "logos"),
            LOGOS_DIR,
            NEWSPAPER_ASSETS_DIR,
            os.path.join(REPO_ROOT, "assets"),
        ):
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    return None


def resolve_newspaper_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    if kind == "heading":
        candidates = [
            os.path.join(FONTS_DIR, "NotoSansMalayalam-ExtraBold.ttf"),
            os.path.join(FONTS_DIR, "NotoSansMalayalam-Bold.ttf"),
            os.path.join(FONTS_DIR, "Chilanka-Regular.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSansMalayalam-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMalayalam-Regular.ttf",
            "/usr/share/fonts/truetype/malayalam/Chilanka-Regular.ttf",
            "/usr/share/fonts/truetype/smc/Chilanka-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        candidates = [
            os.path.join(FONTS_DIR, "NotoSerifMalayalam-Regular.ttf"),
            os.path.join(FONTS_DIR, "NotoSansMalayalam-Regular.ttf"),
            os.path.join(FONTS_DIR, "Chilanka-Regular.ttf"),
            "/usr/share/fonts/truetype/noto/NotoSerifMalayalam-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansMalayalam-Regular.ttf",
            "/usr/share/fonts/truetype/malayalam/Meera-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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


def _has_malayalam(text: str) -> bool:
    return any("\u0d00" <= ch <= "\u0d7f" for ch in (text or ""))


# ==========================================
# 2. ANY-TEXT NORMALIZER
# ==========================================
def normalize_news_data(unit: dict, defaults: dict = None) -> dict:
    defaults = defaults or {}
    sentence = str(unit.get("sentence") or unit.get("sentence_a") or "").strip()
    sentence_b = str(unit.get("sentence_b") or "").strip()
    full_text = f"{sentence} {sentence_b}".strip() if sentence_b else sentence
    is_ml = _has_malayalam(full_text)

    raw_hl = unit.get("headline")
    if isinstance(raw_hl, dict):
        headline = " ".join(
            p for p in [raw_hl.get("prefix", ""), raw_hl.get("highlight", ""), raw_hl.get("suffix", "")] if p
        ).strip()
    else:
        headline = str(raw_hl or unit.get("title") or unit.get("formula_name") or "").strip()

    if not headline:
        if ":" in sentence and len(sentence.split(":", 1)[0]) <= 70:
            headline = sentence.split(":", 1)[0].strip()
        else:
            words = sentence.split()
            headline = " ".join(words[: min(7, len(words))])

    subheadline = str(unit.get("subheadline") or "").strip()
    if not subheadline:
        if sentence_b:
            subheadline = sentence_b
        else:
            sents = [s.strip() for s in re.split(r"[.!?\u0964]+", full_text) if s.strip()]
            if len(sents) > 1:
                subheadline = sents[1]
            elif len(full_text) > len(headline) + 15:
                subheadline = full_text[len(headline):].lstrip(" :,-.")[:140]

    yh = unit.get("yellow_highlights")
    if isinstance(yh, str):
        yellow_highlights = [x.strip() for x in re.split(r"[,;\n|]+", yh) if x.strip()]
    elif isinstance(yh, list):
        yellow_highlights = [str(x).strip() for x in yh if str(x).strip()]
    else:
        yellow_highlights = []

    if not yellow_highlights and full_text:
        clauses = [c.strip() for c in re.split(r"[,;.\n:\-—]+", full_text) if len(c.strip()) >= 6]
        for c in clauses[:4]:
            yellow_highlights.append(" ".join(c.split()[:6]))
        if not yellow_highlights:
            yellow_highlights = [headline]

    bp = unit.get("body_paragraphs") or unit.get("body")
    if isinstance(bp, list):
        body_paragraphs = [str(p).strip() for p in bp if str(p).strip()]
    elif isinstance(bp, str) and bp.strip():
        body_paragraphs = [p.strip() for p in bp.splitlines() if p.strip()]
    else:
        raw_paras = [p.strip() for p in full_text.splitlines() if p.strip()]
        if len(raw_paras) == 1 and len(raw_paras[0]) > 180:
            sents = [s.strip() + "." for s in re.split(r"[.!\u0964]+", raw_paras[0]) if s.strip()]
            mid = max(1, len(sents) // 2)
            body_paragraphs = [" ".join(sents[:mid]), " ".join(sents[mid:])]
            body_paragraphs = [p for p in body_paragraphs if p.strip()]
        else:
            body_paragraphs = raw_paras or [headline]

    today_str = datetime.date.today().strftime("%Y %B %d")
    date_line = str(unit.get("date_line") or defaults.get("date_line") or today_str).strip()
    edition_info = str(
        unit.get("edition_info")
        or defaults.get("edition_info")
        or ("തിരുവനന്തപുരം • പേജ് 1 • വില ₹8.00" if is_ml else "SPECIAL EDITION • PAGE 1")
    ).strip()
    story_heading = str(
        unit.get("story_heading")
        or ("വിശദാംശങ്ങൾ" if is_ml else "DETAILS")
    ).strip()

    return {
        "date_line": date_line,
        "edition_info": edition_info,
        "headline": headline,
        "subheadline": subheadline,
        "yellow_highlights": yellow_highlights[:5],
        "story_heading": story_heading,
        "body_paragraphs": body_paragraphs,
    }


# ==========================================
# 3. EXACT GLYPH BOUNDING & ANTI-OVERFLOW ENGINE
# ==========================================
_HAS_RAQM = features.check("raqm")


def glyph_box(draw, text, font, x=0, y=0):
    if not text:
        return (x, y, x, y)
    if _HAS_RAQM and _has_malayalam(text):
        try:
            return draw.textbbox((x, y), text, font=font, language="ml")
        except TypeError:
            pass
    return draw.textbbox((x, y), text, font=font)


def draw_text_shaped(draw, xy, text, font, fill):
    if _HAS_RAQM and _has_malayalam(text):
        try:
            draw.text(xy, text, font=font, fill=fill, language="ml")
            return
        except TypeError:
            pass
    draw.text(xy, text, font=font, fill=fill)


def wrap_strict(draw, text, font, max_width):
    lines = []
    for para in text.split("\n"):
        words = para.strip().split()
        cur = ""
        for w in words:
            candidate = f"{cur} {w}".strip()
            x0, _, x1, _ = glyph_box(draw, candidate, font)
            if (x1 - x0) <= max_width:
                cur = candidate
            else:
                if cur:
                    lines.append(cur)
                wx0, _, wx1, _ = glyph_box(draw, w, font)
                if (wx1 - wx0) <= max_width:
                    cur = w
                else:
                    part = ""
                    for ch in w:
                        cx0, _, cx1, _ = glyph_box(draw, part + ch, font)
                        if (cx1 - cx0) <= max_width:
                            part += ch
                        else:
                            if part:
                                lines.append(part)
                            part = ch
                    cur = part
        if cur:
            lines.append(cur)
    return lines


def fit_heading_auto(draw, text, max_w, max_h, start_size, min_size=24, gap_px=14):
    for size in range(start_size, min_size - 1, -2):
        font = resolve_newspaper_font("heading", size)
        lines = wrap_strict(draw, text, font, max_w)
        if not lines:
            return font, [], 0
        line_heights = []
        max_lw = 0
        for ln in lines:
            x0, y0, x1, y1 = glyph_box(draw, ln, font, 0, 0)
            max_lw = max(max_lw, x1 - x0)
            line_heights.append(y1 - y0)
        total_h = sum(line_heights) + gap_px * (len(lines) - 1)
        if max_lw <= max_w and total_h <= max_h:
            return font, lines, gap_px
    font = resolve_newspaper_font("heading", min_size)
    return font, wrap_strict(draw, text, font, max_w), gap_px


def paginate_body_across_columns(draw, paragraphs, col_rects, font_size, line_gap=11, para_gap=16):
    font = resolve_newspaper_font("body", font_size)
    _, ref_y0, _, ref_y1 = glyph_box(draw, "കേരളത്തിൽ വികസന പദ്ധതികൾ ശ്രീ ABCgjpqy", font, 0, 0)
    glyph_h = max(font_size, ref_y1 - ref_y0)
    step_y = glyph_h + line_gap

    col_idx = 0
    cx, cy, cw, c_max_y = col_rects[col_idx]
    items = []

    for p_idx, para in enumerate(paragraphs):
        lines = wrap_strict(draw, para, font, cw)
        for l_idx, ln in enumerate(lines):
            if cy + glyph_h > c_max_y:
                col_idx += 1
                if col_idx >= len(col_rects):
                    return False, items
                cx, cy, cw, c_max_y = col_rects[col_idx]

            is_last_line = (l_idx == len(lines) - 1)
            items.append((col_idx, cx, cy, cw, ln, not is_last_line))
            cy += step_y

        if p_idx < len(paragraphs) - 1:
            cy += para_gap
            if cy + glyph_h > c_max_y and col_idx + 1 < len(col_rects):
                col_idx += 1
                cx, cy, cw, c_max_y = col_rects[col_idx]

    return True, items


def draw_justified_line(draw, text, font, x, y, max_w, justify=True, fill=(22, 20, 18)):
    words = text.split()
    if not justify or len(words) <= 1:
        draw_text_shaped(draw, (x, y), text, font, fill)
        return glyph_box(draw, text, font, x, y)

    word_widths = []
    for w in words:
        wx0, _, wx1, _ = glyph_box(draw, w, font, 0, 0)
        word_widths.append(wx1 - wx0)
    total_word_w = sum(word_widths)
    gap = (max_w - total_word_w) / (len(words) - 1)

    if gap > 32 or total_word_w > max_w:
        draw_text_shaped(draw, (x, y), text, font, fill)
        return glyph_box(draw, text, font, x, y)

    cur_x = float(x)
    min_y0, max_y1 = 10**9, -10**9
    for w, ww in zip(words, word_widths):
        ix = int(round(cur_x))
        draw_text_shaped(draw, (ix, y), w, font, fill)
        _, wy0, _, wy1 = glyph_box(draw, w, font, ix, y)
        min_y0 = min(min_y0, wy0)
        max_y1 = max(max_y1, wy1)
        cur_x += ww + gap

    return (x, min_y0, x + max_w, max_y1)


# ==========================================
# 4. NEWSPRINT & TABLE.PNG
# ==========================================
def remove_white_bg(img: Image.Image, threshold=228) -> Image.Image:
    rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    mask = (arr[:, :, 0] > threshold) & (arr[:, :, 1] > threshold) & (arr[:, :, 2] > threshold)
    arr[mask, 3] = 0
    clean = Image.fromarray(arr, "RGBA")
    bbox = clean.getbbox()
    return clean.crop(bbox) if bbox else clean


def generate_wrinkled_newsprint(w: int, h: int) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wrinkle = (
        np.sin(xx * 0.010 + yy * 0.008) * 4.0
        + np.sin(xx * 0.025 - yy * 0.018) * 2.5
        + np.cos(yy * 0.040 + np.sin(xx * 0.014) * 1.8) * 2.0
    )
    fold_h = -10.0 * np.exp(-((yy - h * 0.50) ** 2) / (2 * 14.0 ** 2))
    fold_v = -7.0 * np.exp(-((xx - w * 0.50) ** 2) / (2 * 18.0 ** 2))

    rng = np.random.default_rng(42)
    grain = rng.normal(0, 2.8, (h, w)).astype(np.float32)

    shade = wrinkle + fold_h + fold_v + grain
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = 236.0 + shade
    base[:, :, 1] = 230.0 + shade
    base[:, :, 2] = 215.0 + shade

    base = cv2.GaussianBlur(np.clip(base, 0, 255).astype(np.uint8), (3, 3), 0.5)
    return Image.fromarray(base, "RGB")


def load_table_background(custom_table_path: str = None, target_w: int = VIDEO_W, target_h: int = VIDEO_H) -> np.ndarray:
    found = resolve_table_path(custom_table_path)
    img = cv2.imread(found, cv2.IMREAD_COLOR) if found else None

    if img is None:
        yy, xx = np.mgrid[0:target_h, 0:target_w].astype(np.float32)
        grain = np.sin(yy * 0.14 + np.sin(xx * 0.012) * 3.0) * 8.0 + np.sin(yy * 0.45) * 3.0
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 2.5, (target_h, target_w)).astype(np.float32)
        wood = np.zeros((target_h, target_w, 3), dtype=np.float32)
        wood[:, :, 0] = 28.0 + grain * 0.6 + noise
        wood[:, :, 1] = 44.0 + grain * 0.8 + noise
        wood[:, :, 2] = 68.0 + grain * 1.1 + noise
        cropped = np.clip(wood, 0, 255).astype(np.uint8)
    else:
        src_h, src_w = img.shape[:2]
        scale = max(target_w / src_w, target_h / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        x0 = (new_w - target_w) // 2
        y0 = (new_h - target_h) // 2
        cropped = resized[y0 : y0 + target_h, x0 : x0 + target_w]

    yy, xx = np.mgrid[0:target_h, 0:target_w].astype(np.float32)
    dist = np.sqrt(((xx - target_w * 0.5) / (target_w * 0.62)) ** 2 + ((yy - target_h * 0.5) / (target_h * 0.62)) ** 2)
    vignette = np.clip(1.03 - 0.25 * dist, 0.72, 1.03)[:, :, None]
    return np.clip(cropped.astype(np.float32) * vignette, 0, 255).astype(np.uint8)


# ==========================================
# 5. EXACT 3-COLUMN NEWSPAPER SHEET BUILDER
# ==========================================
def build_newspaper_sheet(news_data: dict, newspaper_id: str = "manorama"):
    W, H = 1680, 1160
    margin = 55
    content_w = W - 2 * margin  # 1570 px
    ink = (20, 18, 16)
    rule_col = (85, 80, 74)

    paper = generate_wrinkled_newsprint(W, H)
    draw = ImageDraw.Draw(paper)

    anim_targets = {
        "headline_underlines": [],
        "yellow_highlights": [],
        "article_underlines": [],
    }

    # --- TOP METADATA BAR ---
    f_meta = resolve_newspaper_font("heading", 22)
    y = 22
    draw_text_shaped(draw, (margin, y), news_data["date_line"], f_meta, ink)
    ex0, _, ex1, ey1 = glyph_box(draw, news_data["edition_info"], f_meta, 0, y)
    draw_text_shaped(draw, (W - margin - (ex1 - ex0), y), news_data["edition_info"], f_meta, ink)
    y = ey1 + 10
    draw.line([(margin, y), (W - margin, y)], fill=ink, width=3)
    y += 10

    # --- MASTHEAD LOGO ---
    banner_h = 122
    logo_path = resolve_logo_path(newspaper_id)
    if logo_path:
        logo_clean = remove_white_bg(Image.open(logo_path))
        scale = min((content_w - 80) / max(1, logo_clean.width), (banner_h - 8) / max(1, logo_clean.height))
        nw = max(1, int(logo_clean.width * scale))
        nh = max(1, int(logo_clean.height * scale))
        logo_resized = logo_clean.resize((nw, nh), Image.Resampling.LANCZOS)
        paper.paste(logo_resized, ((W - nw) // 2, y + (banner_h - nh) // 2), logo_resized)
    else:
        masthead_txt = "മാതൃഭൂമി" if "mathru" in newspaper_id.lower() else "മലയാള മനോരമ"
        f_mast = resolve_newspaper_font("heading", 72)
        mx0, my0, mx1, my1 = glyph_box(draw, masthead_txt, f_mast, 0, 0)
        draw_text_shaped(draw, ((W - (mx1 - mx0)) // 2, y + (banner_h - (my1 - my0)) // 2), masthead_txt, f_mast, ink)

    y += banner_h + 8
    draw.line([(margin, y), (W - margin, y)], fill=ink, width=4)
    y += 12

    # --- DYNAMIC MAIN HEADLINE ---
    f_head, head_lines, head_gap = fit_heading_auto(
        draw, news_data["headline"], content_w, max_h=165, start_size=62, min_size=32, gap_px=16
    )
    for line in head_lines:
        draw_text_shaped(draw, (margin, y), line, f_head, ink)
        hx0, hy0, hx1, hy1 = glyph_box(draw, line, f_head, margin, y)
        underline_y = hy1 + 5
        anim_targets["headline_underlines"].append((hx0, underline_y, hx1, underline_y))
        y = underline_y + head_gap

    # --- DYNAMIC SUB-HEADLINE ---
    if news_data.get("subheadline"):
        f_sub, sub_lines, sub_gap = fit_heading_auto(
            draw, news_data["subheadline"], content_w, max_h=75, start_size=27, min_size=18, gap_px=10
        )
        for line in sub_lines:
            draw_text_shaped(draw, (margin, y), line, f_sub, ink)
            _, _, _, sy1 = glyph_box(draw, line, f_sub, margin, y)
            y = sy1 + sub_gap

    y += 4
    draw.line([(margin, y), (W - margin, y)], fill=rule_col, width=2)
    y += 14

    # --- 3 SYMMETRICAL TEXT COLUMNS (EXACT ORIGINAL STRUCTURE) ---
    col_gap = 30
    col_w = (content_w - 2 * col_gap) // 3  # 503 px each
    cols_x = [margin + i * (col_w + col_gap) for i in range(3)]
    grid_top = y
    grid_bottom = H - 38

    draw.line([(cols_x[1] - col_gap // 2, grid_top), (cols_x[1] - col_gap // 2, grid_bottom)], fill=rule_col, width=2)
    draw.line([(cols_x[2] - col_gap // 2, grid_top), (cols_x[2] - col_gap // 2, grid_bottom)], fill=rule_col, width=2)

    # --- COLUMN 1 TOP: NON-OVERLAPPING YELLOW HIGHLIGHT BULLETS ---
    cy1 = grid_top
    draw_text_shaped(draw, (cols_x[0], cy1), news_data["date_line"], f_meta, ink)
    _, _, _, dy1 = glyph_box(draw, news_data["date_line"], f_meta, cols_x[0], cy1)
    cy1 = dy1 + 14

    highlights = news_data.get("yellow_highlights", [])
    if highlights:
        max_hl_budget = int((grid_bottom - cy1) * 0.56)
        hl_font_size = 28
        while hl_font_size >= 16:
            f_hl = resolve_newspaper_font("heading", hl_font_size)
            test_y = cy1
            for hl_txt in highlights:
                hl_lines = wrap_strict(draw, hl_txt, f_hl, col_w - 16)
                for ln in hl_lines:
                    _, _, _, ly1 = glyph_box(draw, ln, f_hl, cols_x[0] + 8, test_y)
                    test_y = ly1 + 16
                test_y += 8
            if (test_y - cy1) <= max_hl_budget:
                break
            hl_font_size -= 2

        f_hl = resolve_newspaper_font("heading", hl_font_size)
        for hl_txt in highlights:
            hl_lines = wrap_strict(draw, hl_txt, f_hl, col_w - 16)
            for ln in hl_lines:
                if cy1 + 26 > grid_bottom - 40:
                    break
                draw_text_shaped(draw, (cols_x[0] + 8, cy1), ln, f_hl, ink)
                lx0, ly0, lx1, ly1 = glyph_box(draw, ln, f_hl, cols_x[0] + 8, cy1)
                box_x1 = max(cols_x[0], lx0 - 6)
                box_y1 = ly0 - 3
                box_x2 = min(cols_x[0] + col_w, lx1 + 6)
                box_y2 = ly1 + 4
                anim_targets["yellow_highlights"].append((box_x1, box_y1, box_x2, box_y2))
                cy1 = box_y2 + 9
            cy1 += 8

        draw.line([(cols_x[0], cy1), (cols_x[0] + col_w, cy1)], fill=rule_col, width=2)
        cy1 += 12

    # --- COLUMN 2 TOP: STORY SUBHEADING ---
    cy2 = grid_top
    if news_data.get("story_heading"):
        f_sh, sh_lines, sh_gap = fit_heading_auto(
            draw, news_data["story_heading"], col_w, 65, start_size=32, min_size=22, gap_px=10
        )
        for ln in sh_lines:
            draw_text_shaped(draw, (cols_x[1], cy2), ln, f_sh, ink)
            sx0, _, sx1, sy1 = glyph_box(draw, ln, f_sh, cols_x[1], cy2)
            anim_targets["article_underlines"].append((sx0, sy1 + 5, sx1, sy1 + 5))
            cy2 = sy1 + 5 + sh_gap

    # --- AUTO-FLOW ALL BODY CONTENT ACROSS REMAINING SPACE IN COLS 1, 2, 3 ---
    col_rects = [
        (cols_x[0], cy1, col_w, grid_bottom),
        (cols_x[1], cy2, col_w, grid_bottom),
        (cols_x[2], grid_top, col_w, grid_bottom),
    ]

    chosen_body_size = 14
    chosen_items = []
    for test_size in range(30, 13, -1):
        l_gap = 11 if test_size >= 20 else max(5, int(test_size * 0.45))
        p_gap = 16 if test_size >= 20 else max(8, int(test_size * 0.65))
        fits, items = paginate_body_across_columns(draw, news_data["body_paragraphs"], col_rects, test_size, l_gap, p_gap)
        if fits:
            chosen_body_size = test_size
            chosen_items = items
            break
    else:
        _, chosen_items = paginate_body_across_columns(draw, news_data["body_paragraphs"], col_rects, 14, 6, 8)

    f_body = resolve_newspaper_font("body", chosen_body_size)
    col2_drawn_boxes = []
    for (c_idx, lx, ly, lw, ln_text, justify) in chosen_items:
        bx0, by0, bx1, by1 = draw_justified_line(draw, ln_text, f_body, lx, ly, lw, justify=justify, fill=ink)
        if c_idx == 1:
            col2_drawn_boxes.append((bx0, by0, bx1, by1))

    for (bx0, _, bx1, by1) in col2_drawn_boxes[:3]:
        anim_targets["article_underlines"].append((bx0, by1 + 4, bx1, by1 + 4))

    draw.line([(margin, grid_bottom + 6), (W - margin, grid_bottom + 6)], fill=ink, width=4)

    paper_bgr = cv2.cvtColor(np.array(paper), cv2.COLOR_RGB2BGR)
    return paper_bgr, anim_targets


# ==========================================
# 6. EXACT 8.0s REFERENCE CHOREOGRAPHY + REAL-TIME SFX SYNC
# ==========================================
def _to_ref_time(t_sec: float, duration_sec: float) -> float:
    """
    Keeps the physical paper drop + bounce crisp in real-time (0.0s -> 1.05s),
    then smoothly maps [1.05s .. duration_sec] onto [1.05s .. 8.00s] so the exact
    choreography & camera pan/zoom match the original 8s master feel at any TTS length.
    """
    if t_sec <= 1.05:
        return t_sec
    post_dur = max(0.5, duration_sec - 1.05)
    return 1.05 + ((t_sec - 1.05) / post_dur) * (8.0 - 1.05)


def _bandpass_noise(rng, n_samples, low_hz, high_hz, sample_rate=SAMPLE_RATE):
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.normal(0.0, 1.0, n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(white)
    center = (low_hz + high_hz) * 0.5
    bw = max(1.0, (high_hz - low_hz) * 0.5)
    mask = np.exp(-0.5 * ((freqs - center) / bw) ** 2)
    mask[0] = 0.0
    filtered = np.fft.irfft(spectrum * mask, n=n_samples)
    peak = np.max(np.abs(filtered)) + 1e-8
    return (filtered / peak).astype(np.float32)


def _add_stereo_sound(stereo_buf, start_sec, mono_wave, pan=0.0, gain=1.0):
    start_idx = max(0, int(start_sec * SAMPLE_RATE))
    if start_idx >= stereo_buf.shape[0] or len(mono_wave) == 0:
        return
    end_idx = min(stereo_buf.shape[0], start_idx + len(mono_wave))
    seg = mono_wave[: end_idx - start_idx] * gain
    left_g = math.cos((pan + 1.0) * math.pi / 4.0)
    right_g = math.sin((pan + 1.0) * math.pi / 4.0)
    stereo_buf[start_idx:end_idx, 0] += seg * left_g
    stereo_buf[start_idx:end_idx, 1] += seg * right_g


def generate_newspaper_sfx_wav(wav_path: str, duration_sec: float, yellow_boxes: list, paper_w: int):
    duration_sec = max(2.5, float(duration_sec))
    total_samples = max(1, int(round(duration_sec * SAMPLE_RATE)))
    stereo = np.zeros((total_samples, 2), dtype=np.float32)
    rng = np.random.default_rng(101)

    # 1. Soft Paper Drop Whoosh (0.0s -> 0.80s) + Gentle Desk Thud (0.80s)
    whoosh_n = int(0.80 * SAMPLE_RATE)
    t_w = np.linspace(0, 0.80, whoosh_n, endpoint=False)
    air = _bandpass_noise(rng, whoosh_n, 200, 820)
    env_w = np.sin(np.pi * (t_w / 0.80)) ** 1.8
    _add_stereo_sound(stereo, 0.0, air * env_w, pan=0.0, gain=0.28)

    thud_n = int(0.24 * SAMPLE_RATE)
    t_th = np.linspace(0, 0.24, thud_n, endpoint=False)
    freq_sweep = 120.0 * np.exp(-t_th * 20.0) + 44.0
    phase = 2.0 * np.pi * np.cumsum(freq_sweep) / SAMPLE_RATE
    thud_body = np.sin(phase) * np.exp(-t_th * 18.0)
    paper_tap = _bandpass_noise(rng, thud_n, 1500, 4200) * np.exp(-t_th * 18.0)
    _add_stereo_sound(stereo, 0.80, thud_body * 0.65 + paper_tap * 0.35, pan=0.0, gain=0.72)

    # 2. Felt-Tip Yellow Highlighter Marker Strokes synced to t_ref = 2.10 + i * step_ref
    scale_post = max(0.2, (duration_sec - 1.05) / (8.0 - 1.05))
    n_yel = len(yellow_boxes)
    step_ref = min(0.32, 1.60 / max(1, n_yel)) if n_yel > 4 else 0.32

    for i, (x1, _, x2, _) in enumerate(yellow_boxes):
        st_ref = 2.10 + i * step_ref
        st_real = 1.05 + (st_ref - 1.05) * scale_post
        dur_real = max(0.15, 0.34 * scale_post)
        n = int(dur_real * SAMPLE_RATE)
        tt = np.linspace(0, dur_real, n, endpoint=False)
        felt = _bandpass_noise(rng, n, 1400, 3100)
        env = np.sin(np.pi * (tt / dur_real)) ** 0.75
        pan = ((x1 + x2) * 0.5 / max(1, paper_w)) * 2.0 - 1.0
        _add_stereo_sound(stereo, st_real, felt * env, pan=pan, gain=0.28)

    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.85
    pcm_data = (stereo * 32767.0).astype(np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data.tobytes())


# ==========================================
# 7. SYMMETRICAL DROP & HIGHLIGHT ANIMATOR
# ==========================================
def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def compute_symmetrical_drop_rect(desk_w, desk_h, t_sec):
    rest_w, rest_h = 1520.0, 1050.0
    cx, cy = desk_w * 0.5, desk_h * 0.5

    if t_sec < 0.80:
        height_z = 1.0 - ease_out_cubic(t_sec / 0.80)
    elif t_sec < 1.05:
        b = (t_sec - 0.80) / 0.25
        height_z = 0.025 * math.sin(b * math.pi)
    else:
        height_z = 0.0

    scale = 1.0 + 0.40 * height_z
    cur_w = rest_w * scale
    cur_h = rest_h * scale
    return (
        int(round(cx - cur_w * 0.5)),
        int(round(cy - cur_h * 0.5)),
        int(round(cx + cur_w * 0.5)),
        int(round(cy + cur_h * 0.5)),
    ), height_z


def apply_camera_zoom(frame, zoom, focus_x, focus_y):
    if zoom <= 1.001:
        return frame
    h, w = frame.shape[:2]
    crop_w = max(16, int(w / zoom))
    crop_h = max(16, int(h / zoom))
    x1 = int(np.clip(focus_x - crop_w // 2, 0, w - crop_w))
    y1 = int(np.clip(focus_y - crop_h // 2, 0, h - crop_h))
    cropped = frame[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _composite_shadow_fast(desk_bg, dx1, dy1, dx2, dy2, height_z):
    pad_s = int(14 + 60 * height_z)
    off_y = int(12 + 65 * height_z)
    sx1, sy1 = max(0, dx1 - pad_s), max(0, dy1 - pad_s + off_y)
    sx2, sy2 = min(VIDEO_W, dx2 + pad_s), min(VIDEO_H, dy2 + pad_s + off_y)

    sw, sh = VIDEO_W // 4, VIDEO_H // 4
    small_mask = np.zeros((sh, sw), dtype=np.float32)
    cv2.rectangle(small_mask, (sx1 // 4, sy1 // 4), (sx2 // 4, sy2 // 4), 0.58 - 0.22 * height_z, -1)
    blur_k = max(3, int((21 + 52 * height_z) / 4)) | 1
    small_mask = cv2.GaussianBlur(small_mask, (blur_k, blur_k), 0)
    shadow_mask = cv2.resize(small_mask, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)
    return (desk_bg.astype(np.float32) * (1.0 - shadow_mask[:, :, None])).astype(np.uint8)


def generate_newspaper_animation(
    unit: dict,
    output_video_path: str,
    output_sfx_wav_path: str = None,
    duration_sec: float = 8.0,
    newspaper_id: str = "manorama",
    table_path: str = None,
    defaults: dict = None,
    fps: int = 30,
) -> tuple[str, str | None]:
    duration_sec = max(2.5, float(duration_sec))
    news_data = normalize_news_data(unit, defaults=defaults)
    desk_bg = load_table_background(table_path, VIDEO_W, VIDEO_H)
    paper_base, targets = build_newspaper_sheet(news_data, newspaper_id=newspaper_id)
    pw, ph = paper_base.shape[1], paper_base.shape[0]

    if output_sfx_wav_path:
        generate_newspaper_sfx_wav(output_sfx_wav_path, duration_sec, targets["yellow_highlights"], pw)

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    total_frames = max(1, int(round(fps * duration_sec)))

    ffmpeg_bin = shutil.which("ffmpeg")
    proc = None
    writer = None
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{VIDEO_W}x{VIDEO_H}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            output_video_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (VIDEO_W, VIDEO_H))

    yellow_mult = np.array([0.24, 0.96, 1.0], dtype=np.float32)
    red_ink_bgr = (26, 26, 220)

    settled_desk_cache = None
    settled_rect_cache = None

    n_yel = len(targets["yellow_highlights"])
    step_yel = min(0.32, 1.60 / max(1, n_yel)) if n_yel > 4 else 0.32

    for f in range(total_frames):
        t_real = f / float(fps)
        t = _to_ref_time(t_real, duration_sec)
        sheet = paper_base.copy()

        # 1. Headline Red Underlines (1.15s -> 1.95s)
        for i, (x1, y1, x2, y2) in enumerate(targets["headline_underlines"]):
            prog = ease_out_cubic((t - (1.15 + i * 0.35)) / 0.35)
            if prog > 0:
                cur_x2 = int(x1 + (x2 - x1) * prog)
                cv2.line(sheet, (x1, y1), (cur_x2, y2), red_ink_bgr, thickness=5, lineType=cv2.LINE_AA)

        # 2. Yellow Marker Highlighting (2.10s -> 3.80s, synced with marker SFX)
        for i, (x1, y1, x2, y2) in enumerate(targets["yellow_highlights"]):
            prog = ease_out_cubic((t - (2.10 + i * step_yel)) / 0.34)
            if prog > 0:
                cur_x2 = int(x1 + (x2 - x1) * prog)
                if cur_x2 > x1:
                    roi = sheet[y1:y2, x1:cur_x2].astype(np.float32)
                    sheet[y1:y2, x1:cur_x2] = np.clip(roi * yellow_mult, 0, 255).astype(np.uint8)
                    if prog < 0.98:
                        cv2.line(sheet, (cur_x2, y1 + 2), (cur_x2, y2 - 2), (40, 235, 255), 3)

        # 3. Story Red Underlines (4.30s -> 5.80s)
        for i, (x1, y1, x2, y2) in enumerate(targets["article_underlines"]):
            prog = ease_out_cubic((t - (4.30 + i * 0.30)) / 0.32)
            if prog > 0:
                cur_x2 = int(x1 + (x2 - x1) * prog)
                thick = 4 if i == 0 else 3
                cv2.line(sheet, (x1, y1), (cur_x2, y2), red_ink_bgr, thickness=thick, lineType=cv2.LINE_AA)

        # 4. Symmetrical Drop onto Wooden Table (real-time 0.80s + 0.25s bounce)
        (dx1, dy1, dx2, dy2), height_z = compute_symmetrical_drop_rect(VIDEO_W, VIDEO_H, t_real)
        if height_z == 0.0:
            if settled_desk_cache is None:
                settled_desk_cache = _composite_shadow_fast(desk_bg, dx1, dy1, dx2, dy2, 0.0)
                settled_rect_cache = (dx1, dy1, dx2, dy2)
            frame = settled_desk_cache.copy()
            dx1, dy1, dx2, dy2 = settled_rect_cache
        else:
            frame = _composite_shadow_fast(desk_bg, dx1, dy1, dx2, dy2, height_z)

        target_w, target_h = max(1, dx2 - dx1), max(1, dy2 - dy1)
        resized_sheet = cv2.resize(
            sheet, (target_w, target_h), interpolation=cv2.INTER_AREA if target_w < pw else cv2.INTER_LINEAR
        )

        vx1, vy1 = max(0, dx1), max(0, dy1)
        vx2, vy2 = min(VIDEO_W, dx2), min(VIDEO_H, dy2)
        if vx2 > vx1 and vy2 > vy1:
            frame[vy1:vy2, vx1:vx2] = resized_sheet[vy1 - dy1 : vy2 - dy1, vx1 - dx1 : vx2 - dx1]

        # 5. Exact Original Smooth Camera Pan/Zoom Choreography
        if t < 1.15:
            zoom, fx, fy = 1.0, VIDEO_W * 0.50, VIDEO_H * 0.50
        elif t < 2.15:
            p = ease_in_out_cubic((t - 1.15) / 1.00)
            zoom = 1.0 + 0.24 * p
            fx = VIDEO_W * (0.50 - 0.08 * p)
            fy = VIDEO_H * (0.50 - 0.14 * p)
        elif t < 4.20:
            p = ease_in_out_cubic((t - 2.15) / 2.05)
            zoom = 1.24 + 0.06 * p
            fx = VIDEO_W * (0.42 - 0.08 * p)
            fy = VIDEO_H * (0.36 + 0.14 * p)
        elif t < 6.40:
            p = ease_in_out_cubic((t - 4.20) / 2.20)
            zoom = 1.30
            fx = VIDEO_W * (0.34 + 0.24 * p)
            fy = VIDEO_H * (0.50 + 0.04 * p)
        else:
            p = ease_in_out_cubic((t - 6.40) / 1.60)
            zoom = 1.30 - 0.30 * p
            fx = VIDEO_W * (0.58 - 0.08 * p)
            fy = VIDEO_H * (0.54 - 0.04 * p)

        frame = apply_camera_zoom(frame, zoom, fx, fy)

        if proc is not None:
            proc.stdin.write(frame.tobytes())
        else:
            writer.write(frame)

    if proc is not None:
        proc.stdin.close()
        proc.wait()
    if writer is not None:
        writer.release()

    return output_video_path, output_sfx_wav_path
