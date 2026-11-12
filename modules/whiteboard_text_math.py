import math
import os
import sys
import time
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

BASE_DIR = os.getcwd()
DEFAULT_EN_FONT = os.path.join(BASE_DIR, "assets", "fonts", "PatrickHand-Regular.ttf")
DEFAULT_ML_FONT = os.path.join(BASE_DIR, "assets", "fonts", "Chilanka-Regular.ttf")
DEFAULT_HAND_IMG = os.path.join(BASE_DIR, "assets", "whiteboard", "alpha-hand.png")


def _resolve_font(font_path: str, font_size: int):
    """Loads font with fallback to system fonts."""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass

    sys_fallbacks = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "/Library/Fonts/Arial.ttf"
    ]
    for fb in sys_fallbacks:
        if os.path.exists(fb):
            try:
                return ImageFont.truetype(fb, font_size)
            except Exception:
                pass

    try:
        return ImageFont.load_default(size=font_size)
    except Exception:
        return ImageFont.load_default()


def get_extreme_coordinates(mask):
    indices = np.where(mask > 0)
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        return (0, 0), (mask.shape[1], mask.shape[0])
    topleft = (int(np.min(indices[1])), int(np.min(indices[0])))
    bottomright = (int(np.max(indices[1])), int(np.max(indices[0])))
    return topleft, bottomright


def preprocess_hand_image(hand_path, target_canvas_wd):
    if not os.path.exists(hand_path):
        alt = os.path.join(BASE_DIR, "assets", "whiteboard", "alpha-hand_2.png")
        if os.path.exists(alt):
            hand_path = alt
        else:
            raise FileNotFoundError(f"Hand image not found at '{hand_path}'")

    hand_raw = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    if hand_raw is None:
        raise ValueError(f"Failed to load hand image from {hand_path}")

    if hand_raw.ndim == 3 and hand_raw.shape[2] == 4:
        hand_bgr = hand_raw[:, :, :3]
        hand_alpha = (hand_raw[:, :, 3] > 15).astype(np.float32) * (hand_raw[:, :, 3] / 255.0)
    else:
        hand_bgr = hand_raw[:, :, :3]
        h_temp, w_temp = hand_bgr.shape[:2]
        ff_mask = np.zeros((h_temp + 2, w_temp + 2), np.uint8)
        gray_temp = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2GRAY)
        cv2.floodFill(gray_temp.copy(), ff_mask, (0, 0), 0, loDiff=15, upDiff=15)
        cv2.floodFill(gray_temp.copy(), ff_mask, (w_temp - 1, 0), 0, loDiff=15, upDiff=15)
        cv2.floodFill(gray_temp.copy(), ff_mask, (0, h_temp - 1), 0, loDiff=15, upDiff=15)
        cv2.floodFill(gray_temp.copy(), ff_mask, (w_temp - 1, h_temp - 1), 0, loDiff=15, upDiff=15)
        is_bg = (ff_mask[1:-1, 1:-1] == 1)
        hand_alpha = np.where(is_bg, 0.0, 1.0).astype(np.float32)

    binary_mask = (hand_alpha > 0.1).astype(np.uint8) * 255
    (tl_x, tl_y), (br_x, br_y) = get_extreme_coordinates(binary_mask)
    hand_bgr = hand_bgr[tl_y:br_y + 1, tl_x:br_x + 1]
    hand_alpha = hand_alpha[tl_y:br_y + 1, tl_x:br_x + 1]

    desired_hand_wd = max(150, int(target_canvas_wd * 0.35))
    scale = desired_hand_wd / float(hand_bgr.shape[1])
    new_wd = desired_hand_wd
    new_ht = int(hand_bgr.shape[0] * scale)

    hand_bgr = cv2.resize(hand_bgr, (new_wd, new_ht), interpolation=cv2.INTER_AREA)
    hand_alpha = cv2.resize(hand_alpha, (new_wd, new_ht), interpolation=cv2.INTER_AREA)

    y_indices, x_indices = np.where(hand_alpha > 0.5)
    if len(y_indices) == 0:
        y_indices, x_indices = np.where(hand_alpha > 0.1)

    diag_scores = x_indices.astype(np.float32) + y_indices.astype(np.float32)
    min_score = np.min(diag_scores)
    tip_zone = np.where(diag_scores <= min_score + 10)[0]
    cand_x = x_indices[tip_zone]
    cand_y = y_indices[tip_zone]

    tip_zone_bgr = hand_bgr[cand_y, cand_x]
    tip_zone_gray = cv2.cvtColor(tip_zone_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
    dark_pixels = tip_zone_gray < 110

    if np.sum(dark_pixels) > 2:
        tip_x = int(np.mean(cand_x[dark_pixels]))
        tip_y = int(np.mean(cand_y[dark_pixels]))
    else:
        tip_x = int(np.mean(cand_x))
        tip_y = int(np.mean(cand_y))

    return hand_bgr, hand_alpha, tip_x, tip_y


def draw_hand_with_extended_arm(canvas, hand_bgr, hand_alpha, tip_x, tip_y, target_x, target_y):
    canvas_ht, canvas_wd = canvas.shape[:2]
    origin_x = int(round(target_x)) - tip_x
    origin_y = int(round(target_y)) - tip_y

    current_hand_ht, _ = hand_bgr.shape[:2]
    arm_bottom_y = origin_y + current_hand_ht

    if arm_bottom_y < canvas_ht:
        extension_len = (canvas_ht - arm_bottom_y) + 80
        last_row_color = hand_bgr[-1:, :, :]
        last_row_alpha = hand_alpha[-1:, :]
        ext_color = np.tile(last_row_color, (extension_len, 1, 1))
        ext_alpha = np.tile(last_row_alpha, (extension_len, 1))
        full_hand_bgr = np.vstack([hand_bgr, ext_color])
        full_hand_alpha = np.vstack([hand_alpha, ext_alpha])
    else:
        full_hand_bgr = hand_bgr
        full_hand_alpha = hand_alpha

    h_ht, h_wd = full_hand_bgr.shape[:2]
    c_x1 = max(0, origin_x)
    c_y1 = max(0, origin_y)
    c_x2 = min(canvas_wd, origin_x + h_wd)
    c_y2 = min(canvas_ht, origin_y + h_ht)

    if c_x2 <= c_x1 or c_y2 <= c_y1:
        return canvas

    h_x1 = c_x1 - origin_x
    h_y1 = c_y1 - origin_y
    h_x2 = h_x1 + (c_x2 - c_x1)
    h_y2 = h_y1 + (c_y2 - c_y1)

    cropped_hand = full_hand_bgr[h_y1:h_y2, h_x1:h_x2]
    cropped_alpha = full_hand_alpha[h_y1:h_y2, h_x1:h_x2, np.newaxis]
    canvas_roi = canvas[c_y1:c_y2, c_x1:c_x2]
    canvas[c_y1:c_y2, c_x1:c_x2] = (canvas_roi * (1.0 - cropped_alpha) + cropped_hand * cropped_alpha).astype(np.uint8)
    return canvas


def zhang_suen_thinning(binary_mask):
    padded = np.pad((binary_mask > 0).astype(np.uint8), 1, mode="constant", constant_values=0)
    for _ in range(50):
        p1 = padded[1:-1, 1:-1]
        p2 = padded[0:-2, 1:-1]
        p3 = padded[0:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, 0:-2]
        p8 = padded[1:-1, 0:-2]
        p9 = padded[0:-2, 0:-2]

        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (((p2 == 0) & (p3 == 1)).astype(np.uint8) +
             ((p3 == 0) & (p4 == 1)).astype(np.uint8) +
             ((p4 == 0) & (p5 == 1)).astype(np.uint8) +
             ((p5 == 0) & (p6 == 1)).astype(np.uint8) +
             ((p6 == 0) & (p7 == 1)).astype(np.uint8) +
             ((p7 == 0) & (p8 == 1)).astype(np.uint8) +
             ((p8 == 0) & (p9 == 1)).astype(np.uint8) +
             ((p9 == 0) & (p2 == 1)).astype(np.uint8))
        m1 = (p1 == 1) & (b >= 2) & (b <= 6) & (a == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
        if np.any(m1):
            padded[1:-1, 1:-1][m1] = 0

        p1 = padded[1:-1, 1:-1]
        p2 = padded[0:-2, 1:-1]
        p3 = padded[0:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, 0:-2]
        p8 = padded[1:-1, 0:-2]
        p9 = padded[0:-2, 0:-2]

        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (((p2 == 0) & (p3 == 1)).astype(np.uint8) +
             ((p3 == 0) & (p4 == 1)).astype(np.uint8) +
             ((p4 == 0) & (p5 == 1)).astype(np.uint8) +
             ((p5 == 0) & (p6 == 1)).astype(np.uint8) +
             ((p6 == 0) & (p7 == 1)).astype(np.uint8) +
             ((p7 == 0) & (p8 == 1)).astype(np.uint8) +
             ((p8 == 0) & (p9 == 1)).astype(np.uint8) +
             ((p9 == 0) & (p2 == 1)).astype(np.uint8))
        m2 = (p1 == 1) & (b >= 2) & (b <= 6) & (a == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
        if np.any(m2):
            padded[1:-1, 1:-1][m2] = 0

        if not np.any(m1) and not np.any(m2):
            break

    return padded[1:-1, 1:-1]


def trace_glyph_skeleton_strokes(glyph_mask, offset_x, offset_y, step_px=2.0):
    gh, gw = glyph_mask.shape[:2]
    if gh == 0 or gw == 0 or not np.any(glyph_mask > 0):
        return []

    dist_map = cv2.distanceTransform(glyph_mask, cv2.DIST_L2, 3)
    skel = zhang_suen_thinning(glyph_mask)
    if np.sum(skel) == 0:
        skel = (glyph_mask > 0).astype(np.uint8)

    kernel_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_counts = cv2.filter2D(skel, -1, kernel_8, borderType=cv2.BORDER_CONSTANT)

    unvisited = set(zip(*np.where(skel > 0)))
    traced_points = []

    while unvisited:
        endpoints = [pt for pt in unvisited if neighbor_counts[pt[0], pt[1]] == 1]
        candidates = endpoints if endpoints else list(unvisited)
        start_pt = min(candidates, key=lambda p: p[0] + 0.45 * p[1])

        stroke = [start_pt]
        unvisited.remove(start_pt)
        curr = start_pt
        prev_dir = None

        while True:
            cy, cx = curr
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (ny, nx) in unvisited:
                        nbrs.append((ny, nx, dx, dy))

            if not nbrs:
                break

            if len(nbrs) == 1:
                ny, nx, dx, dy = nbrs[0]
            else:
                best_score = -999.0
                best_nbr = nbrs[0]
                for ny, nx, dx, dy in nbrs:
                    norm = math.hypot(dx, dy)
                    ux, uy = dx / norm, dy / norm
                    score = ux * prev_dir[0] + uy * prev_dir[1] if prev_dir else 0.7 * uy + 0.3 * ux
                    if score > best_score:
                        best_score = score
                        best_nbr = (ny, nx, dx, dy)
                ny, nx, dx, dy = best_nbr

            norm = math.hypot(dx, dy)
            ux, uy = dx / norm, dy / norm
            if prev_dir is None:
                prev_dir = (ux, uy)
            else:
                bx = 0.72 * ux + 0.28 * prev_dir[0]
                by = 0.72 * uy + 0.28 * prev_dir[1]
                bn = math.hypot(bx, by)
                if bn > 1e-5:
                    prev_dir = (bx / bn, by / bn)

            unvisited.remove((ny, nx))
            curr = (ny, nx)
            stroke.append(curr)

        accum_dist = 0.0
        for idx, (py, px) in enumerate(stroke):
            if idx > 0:
                prev_y, prev_x = stroke[idx - 1]
                accum_dist += math.hypot(px - prev_x, py - prev_y)

            if idx == 0 or idx == len(stroke) - 1 or accum_dist >= step_px:
                if 0 < idx < len(stroke) - 1:
                    accum_dist = 0.0
                
                # SAFE CLAMPED RADIUS TO PREVENT C OVERFLOW ERROR
                val = dist_map[py, px]
                calc_r = int(math.ceil(val)) + 2 if not np.isnan(val) and not np.isinf(val) else 4
                local_r = int(np.clip(calc_r, 2, 35))
                traced_points.append((int(offset_x + px), int(offset_y + py), local_r))

    return traced_points


def wrap_text_lines(text, font, max_width, lang="en"):
    has_raqm = features.check("raqm")
    raw_paragraphs = text.split("\n")
    wrapped_lines = []

    for para in raw_paragraphs:
        words = para.split(" ")
        if not words:
            wrapped_lines.append("")
            continue

        current_line = words[0]
        for word in words[1:]:
            test_line = f"{current_line} {word}"
            try:
                bbox = font.getbbox(test_line, language=lang) if has_raqm else font.getbbox(test_line)
            except TypeError:
                bbox = font.getbbox(test_line)
            line_w = bbox[2] - bbox[0]

            if line_w <= max_width:
                current_line = test_line
            else:
                wrapped_lines.append(current_line)
                current_line = word

        wrapped_lines.append(current_line)

    return wrapped_lines


def render_math_equation_rgba(latex_str, font_size=36, color="#111111", dpi=240):
    fig = plt.figure(figsize=(12, 3), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)

    expr = latex_str.strip()
    if not (expr.startswith("$") and expr.endswith("$")):
        expr = f"${expr}$"

    ax.text(0.02, 0.5, expr, fontsize=font_size, color=color, va="center", ha="left", transform=ax.transAxes)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    alpha = buf[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros((10, 10, 4), dtype=np.uint8)

    pad = 8
    y1 = max(0, int(np.min(ys)) - pad)
    y2 = min(buf.shape[0], int(np.max(ys)) + pad + 1)
    x1 = max(0, int(np.min(xs)) - pad)
    x2 = min(buf.shape[1], int(np.max(xs)) + pad + 1)

    return buf[y1:y2, x1:x2].copy()


def generate_line_stroke_points(x1, y, x2, step=2.5):
    if x2 <= x1:
        return [(int(x1), int(y))]
    n_pts = max(2, int(math.ceil((x2 - x1) / float(step))) + 1)
    xs = np.linspace(x1, x2, n_pts)
    return [(int(round(px)), int(y)) for px in xs]


def generate_box_perimeter_points(x1, y1, x2, y2, step=2.5):
    pts = []
    edges = [
        (x1, y1, x2, y1),
        (x2, y1, x2, y2),
        (x2, y2, x1, y2),
        (x1, y2, x1, y1)
    ]
    for sx, sy, ex, ey in edges:
        dist = math.hypot(ex - sx, ey - sy)
        n_pts = max(2, int(math.ceil(dist / float(step))) + 1)
        xs = np.linspace(sx, ex, n_pts)
        ys = np.linspace(sy, ey, n_pts)
        for idx in range(len(xs)):
            if idx == 0 and pts:
                continue
            pts.append((int(round(xs[idx])), int(round(ys[idx]))))
    return pts


def build_layout_and_stroke_tasks(content_blocks, canvas_wd, canvas_ht, en_font_path, ml_font_path, margin_x=90, margin_y=70):
    has_raqm = features.check("raqm")
    master_pil = Image.new("RGB", (canvas_wd, canvas_ht), (255, 255, 255))
    master_draw = ImageDraw.Draw(master_pil)

    max_text_width = canvas_wd - (2 * margin_x)
    cursor_y = margin_y
    stroke_tasks = []

    for block in content_blocks:
        b_type = block.get("type", "english").lower()
        align = block.get("align", "left").lower()
        space_after = block.get("space_after", 26)

        if b_type in ("english", "malayalam", "heading"):
            is_heading = (b_type == "heading")
            underline = block.get("underline", is_heading)
            underline_color = tuple(block.get("underline_color", (220, 38, 38)))
            underline_thickness = block.get("underline_thickness", 5)

            text = block.get("text", "")
            font_size = block.get("size", 62 if is_heading else 48)
            color = tuple(block.get("color", (15, 23, 42)))

            has_malayalam_chars = any('\u0d00' <= ch <= '\u0d7f' for ch in text)
            lang = "ml" if (b_type == "malayalam" or has_malayalam_chars) else "en"
            font_path = ml_font_path if (lang == "ml" and os.path.exists(ml_font_path)) else en_font_path

            font = _resolve_font(font_path, font_size)
            try:
                ascent, descent = font.getmetrics()
                line_step = int((ascent + descent) * 1.28)
            except Exception:
                line_step = int(font_size * 1.35)

            lines = wrap_text_lines(text, font, max_text_width, lang=lang)
            heading_min_x = canvas_wd
            heading_max_x = 0
            heading_last_y = cursor_y

            for line in lines:
                if not line.strip():
                    cursor_y += int(line_step * 0.6)
                    continue

                if cursor_y + line_step > canvas_ht - margin_y:
                    break

                try:
                    bbox = font.getbbox(line, language=lang) if has_raqm else font.getbbox(line)
                except TypeError:
                    bbox = font.getbbox(line)

                text_w = bbox[2] - bbox[0]
                if align == "center":
                    draw_x = max(margin_x, (canvas_wd - text_w) // 2)
                elif align == "right":
                    draw_x = max(margin_x, canvas_wd - margin_x - text_w)
                else:
                    draw_x = margin_x

                line_layer = Image.new("RGB", (canvas_wd, canvas_ht), (255, 255, 255))
                line_draw = ImageDraw.Draw(line_layer)

                try:
                    if has_raqm:
                        line_draw.text((draw_x, cursor_y), line, font=font, fill=color, language=lang)
                        master_draw.text((draw_x, cursor_y), line, font=font, fill=color, language=lang)
                    else:
                        line_draw.text((draw_x, cursor_y), line, font=font, fill=color)
                        master_draw.text((draw_x, cursor_y), line, font=font, fill=color)
                except TypeError:
                    line_draw.text((draw_x, cursor_y), line, font=font, fill=color)
                    master_draw.text((draw_x, cursor_y), line, font=font, fill=color)

                line_gray = cv2.cvtColor(np.array(line_layer), cv2.COLOR_RGB2GRAY)
                _, line_thresh = cv2.threshold(line_gray, 245, 255, cv2.THRESH_BINARY)

                if np.any(line_thresh < 255):
                    stroke_tasks.append({"kind": "text_line", "mask": line_thresh})
                    ys_ink, xs_ink = np.where(line_thresh < 255)
                    heading_min_x = min(heading_min_x, int(np.min(xs_ink)))
                    heading_max_x = max(heading_max_x, int(np.max(xs_ink)))
                    heading_last_y = max(heading_last_y, int(np.max(ys_ink)))

                cursor_y += line_step

            if underline and heading_max_x > heading_min_x:
                ul_y = min(canvas_ht - 12, heading_last_y + 12)
                ul_x1 = max(10, heading_min_x - 4)
                ul_x2 = min(canvas_wd - 10, heading_max_x + 6)

                ul_layer = Image.new("RGB", (canvas_wd, canvas_ht), (255, 255, 255))
                ul_draw = ImageDraw.Draw(ul_layer)

                master_draw.line([(ul_x1, ul_y), (ul_x2, ul_y)], fill=underline_color, width=underline_thickness)
                ul_draw.line([(ul_x1, ul_y), (ul_x2, ul_y)], fill=underline_color, width=underline_thickness)

                ul_gray = cv2.cvtColor(np.array(ul_layer), cv2.COLOR_RGB2GRAY)
                _, ul_thresh = cv2.threshold(ul_gray, 245, 255, cv2.THRESH_BINARY)

                stroke_tasks.append({
                    "kind": "vector_stroke",
                    "mask": ul_thresh,
                    "points": generate_line_stroke_points(ul_x1, ul_y, ul_x2, step=2.5),
                    "brush_r": max(8, underline_thickness + 4)
                })
                cursor_y = max(cursor_y, ul_y + 12)

            cursor_y += space_after

        elif b_type == "math":
            latex_str = block.get("latex", "")
            font_size = block.get("size", 34)
            color = block.get("color", "#111111")
            dpi = block.get("dpi", 240)

            draw_box = block.get("box", True)
            box_color = tuple(block.get("box_color", (220, 38, 38)))
            box_thickness = block.get("box_thickness", 4)
            box_pad_x = block.get("box_pad_x", 24) if draw_box else 0
            box_pad_y = block.get("box_pad_y", 16) if draw_box else 0

            eq_rgba = render_math_equation_rgba(latex_str, font_size=font_size, color=color, dpi=dpi)
            eq_h, eq_w = eq_rgba.shape[:2]

            avail_eq_w = max_text_width - (2 * box_pad_x)
            if eq_w > avail_eq_w:
                scale_f = avail_eq_w / float(eq_w)
                eq_w = avail_eq_w
                eq_h = max(10, int(eq_h * scale_f))
                eq_rgba = cv2.resize(eq_rgba, (eq_w, eq_h), interpolation=cv2.INTER_AREA)

            total_block_h = eq_h + (2 * box_pad_y)
            if cursor_y + total_block_h > canvas_ht - margin_y:
                break

            eq_y = cursor_y + box_pad_y
            if align == "center":
                eq_x = max(margin_x + box_pad_x, (canvas_wd - eq_w) // 2)
            elif align == "right":
                eq_x = max(margin_x + box_pad_x, canvas_wd - margin_x - box_pad_x - eq_w)
            else:
                eq_x = margin_x + box_pad_x

            eq_pil = Image.fromarray(eq_rgba, mode="RGBA")
            master_pil.paste(eq_pil, (eq_x, eq_y), eq_pil)

            eq_layer = Image.new("RGB", (canvas_wd, canvas_ht), (255, 255, 255))
            eq_layer.paste(eq_pil, (eq_x, eq_y), eq_pil)

            eq_gray = cv2.cvtColor(np.array(eq_layer), cv2.COLOR_RGB2GRAY)
            _, eq_thresh = cv2.threshold(eq_gray, 245, 255, cv2.THRESH_BINARY)

            if np.any(eq_thresh < 255):
                stroke_tasks.append({"kind": "math_line", "mask": eq_thresh})

            if draw_box and np.any(eq_thresh < 255):
                ys_eq, xs_eq = np.where(eq_thresh < 255)
                bx1 = max(12, int(np.min(xs_eq)) - box_pad_x)
                by1 = max(12, int(np.min(ys_eq)) - box_pad_y)
                bx2 = min(canvas_wd - 12, int(np.max(xs_eq)) + box_pad_x)
                by2 = min(canvas_ht - 12, int(np.max(ys_eq)) + box_pad_y)

                box_layer = Image.new("RGB", (canvas_wd, canvas_ht), (255, 255, 255))
                box_draw = ImageDraw.Draw(box_layer)

                master_draw.rectangle([(bx1, by1), (bx2, by2)], outline=box_color, width=box_thickness)
                box_draw.rectangle([(bx1, by1), (bx2, by2)], outline=box_color, width=box_thickness)

                box_gray = cv2.cvtColor(np.array(box_layer), cv2.COLOR_RGB2GRAY)
                _, box_thresh = cv2.threshold(box_gray, 245, 255, cv2.THRESH_BINARY)

                stroke_tasks.append({
                    "kind": "vector_stroke",
                    "mask": box_thresh,
                    "points": generate_box_perimeter_points(bx1, by1, bx2, by2, step=2.5),
                    "brush_r": max(9, box_thickness + 5)
                })
                cursor_y = by2 + space_after
            else:
                cursor_y += total_block_h + space_after

    master_bgr = cv2.cvtColor(np.array(master_pil), cv2.COLOR_RGB2BGR)
    return master_bgr, stroke_tasks


def extract_text_line_glyph_strokes(line_thresh):
    ink_binary = (line_thresh < 245).astype(np.uint8)
    if np.sum(ink_binary) == 0:
        return []

    kernel = np.ones((7, 2), np.uint8)
    grouped = cv2.dilate(ink_binary, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grouped)

    glyphs = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area == 0:
            continue
        sub_mask = ((labels[y:y+h, x:x+w] == lbl) & (ink_binary[y:y+h, x:x+w] == 1)).astype(np.uint8)
        if np.sum(sub_mask) == 0:
            continue
        glyphs.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "mask": sub_mask})

    glyphs.sort(key=lambda g: (g["x"], g["y"]))
    glyph_entries = []
    for g in glyphs:
        pts = trace_glyph_skeleton_strokes(g["mask"], g["x"], g["y"], step_px=2.0)
        if pts:
            glyph_entries.append({"bbox": (g["x"], g["y"], g["w"], g["h"]), "mask": g["mask"], "points": pts})
    return glyph_entries


def order_math_components_naturally(comps):
    if not comps:
        return []

    remaining = list(comps)
    fraction_bars = []
    for c in remaining:
        aspect = c["w"] / float(max(1, c["h"]))
        fill_ratio = c["area"] / float(max(1, c["w"] * c["h"]))
        if c["w"] >= 22 and c["h"] <= 9 and aspect >= 3.0 and fill_ratio >= 0.50:
            fraction_bars.append(c)

    fraction_bars.sort(key=lambda b: -b["w"])
    used_ids = set()
    structured_groups = []

    for bar in fraction_bars:
        if bar["id"] in used_ids:
            continue
        bx1, bx2 = bar["x"] - 4, bar["x"] + bar["w"] + 4
        by_mid = bar["y"] + bar["h"] * 0.5
        num_comps = []
        den_comps = []
        for c in remaining:
            if c["id"] == bar["id"] or c["id"] in used_ids:
                continue
            ccx = c["x"] + c["w"] * 0.5
            ccy = c["y"] + c["h"] * 0.5
            if bx1 <= ccx <= bx2:
                if 0 < (by_mid - ccy) <= 130:
                    num_comps.append(c)
                elif 0 < (ccy - by_mid) <= 130:
                    den_comps.append(c)

        if num_comps and den_comps:
            used_ids.add(bar["id"])
            for nc in num_comps:
                used_ids.add(nc["id"])
            for dc in den_comps:
                used_ids.add(dc["id"])
            group_seq = order_math_components_naturally(num_comps) + [bar] + order_math_components_naturally(den_comps)
            structured_groups.append({"x": bar["x"], "y": min(nc["y"] for nc in num_comps), "seq": group_seq})

    radicals = []
    for c in remaining:
        if c["id"] in used_ids:
            continue
        if c["w"] >= 28 and c["h"] >= 22:
            top_strip = c["mask"][:max(3, min(6, c["h"])), :]
            if np.mean(top_strip > 0) > 0.45:
                radicals.append(c)

    for rad in radicals:
        if rad["id"] in used_ids:
            continue
        rx1, rx2 = rad["x"] + 6, rad["x"] + rad["w"] + 2
        ry1, ry2 = rad["y"] + 2, rad["y"] + rad["h"] + 4
        inner_comps = []
        for c in remaining:
            if c["id"] == rad["id"] or c["id"] in used_ids:
                continue
            ccx = c["x"] + c["w"] * 0.5
            ccy = c["y"] + c["h"] * 0.5
            if rx1 <= ccx <= rx2 and ry1 <= ccy <= ry2:
                inner_comps.append(c)

        if inner_comps:
            used_ids.add(rad["id"])
            for ic in inner_comps:
                used_ids.add(ic["id"])
            group_seq = [rad] + order_math_components_naturally(inner_comps)
            structured_groups.append({"x": rad["x"], "y": rad["y"], "seq": group_seq})

    for c in remaining:
        if c["id"] not in used_ids:
            structured_groups.append({"x": c["x"], "y": c["y"], "seq": [c]})

    structured_groups.sort(key=lambda g: (g["x"] // 8, -g["y"]))
    ordered = []
    for g in structured_groups:
        ordered.extend(g["seq"])
    return ordered


def extract_math_line_glyph_strokes(eq_thresh):
    ink_binary = (eq_thresh < 245).astype(np.uint8)
    if np.sum(ink_binary) == 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_binary)
    comps = []
    for lbl in range(1, num_labels):
        x, y, w, h, area = stats[lbl]
        if area == 0:
            continue
        sub_mask = (labels[y:y+h, x:x+w] == lbl).astype(np.uint8)
        comps.append({"id": lbl, "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area), "mask": sub_mask})

    ordered_comps = order_math_components_naturally(comps)
    glyph_entries = []
    for c in ordered_comps:
        pts = trace_glyph_skeleton_strokes(c["mask"], c["x"], c["y"], step_px=2.0)
        if pts:
            glyph_entries.append({"bbox": (c["x"], c["y"], c["w"], c["h"]), "mask": c["mask"], "points": pts})
    return glyph_entries


def generate_text_math_whiteboard(
    content_blocks: list[dict],
    output_path: str,
    hand_path: str = DEFAULT_HAND_IMG,
    en_font_path: str = DEFAULT_EN_FONT,
    ml_font_path: str = DEFAULT_ML_FONT,
    canvas_wd: int = 1920,
    canvas_ht: int = 1080,
    total_duration_sec: float = 10.0,
    fps: int = 30,
    final_hold_sec: float = 1.0,
) -> str:
    """Generates a text/math whiteboard video with safe bound math."""
    total_duration_sec = max(2.0, float(total_duration_sec))
    master_bgr, stroke_tasks = build_layout_and_stroke_tasks(
        content_blocks=content_blocks,
        canvas_wd=canvas_wd,
        canvas_ht=canvas_ht,
        en_font_path=en_font_path,
        ml_font_path=ml_font_path,
    )

    if not stroke_tasks:
        raise ValueError("No visible text or math ink found.")

    total_target_frames = max(2, int(round(total_duration_sec * fps)))
    requested_hold_frames = max(0, int(round(final_hold_sec * fps)))
    final_hold_frames = min(requested_hold_frames, int(total_target_frames * 0.20))
    remaining_frames = max(1, total_target_frames - final_hold_frames)
    exit_frames = min(7, max(2, int(round(0.28 * fps)))) if remaining_frames >= 18 else 0
    active_draw_frames = max(1, remaining_frames - exit_frames)

    max_glide_steps = max(3, min(14, active_draw_frames // 22))
    timeline_steps = []
    last_nib_pos = None

    def append_pen_lift_glide(target_x, target_y):
        nonlocal last_nib_pos
        if last_nib_pos is None:
            return
        sx, sy = last_nib_pos
        dist = math.hypot(target_x - sx, target_y - sy)
        if dist > 90.0:
            n_steps = min(max_glide_steps, max(2, int(dist // 140)))
            lift_arc = min(22.0, dist * 0.025)
            for step_i in range(1, n_steps + 1):
                t = step_i / float(n_steps + 1)
                gx = sx + t * (target_x - sx)
                gy = sy + t * (target_y - sy) - math.sin(math.pi * t) * lift_arc
                timeline_steps.append(("glide", int(gx), int(gy), 0, None))

    for task in stroke_tasks:
        if task["kind"] in ("text_line", "math_line"):
            glyph_entries = extract_text_line_glyph_strokes(task["mask"]) if task["kind"] == "text_line" else extract_math_line_glyph_strokes(task["mask"])
            for g_entry in glyph_entries:
                pts = g_entry["points"]
                if not pts:
                    continue
                append_pen_lift_glide(pts[0][0], pts[0][1])
                for px, py, radius in pts:
                    timeline_steps.append(("draw_glyph", int(px), int(py), int(radius), g_entry))
                    last_nib_pos = (px, py)
                end_px, end_py, _ = pts[-1]
                timeline_steps.append(("snap_glyph", int(end_px), int(end_py), 0, g_entry))

        elif task["kind"] == "vector_stroke":
            pts = task["points"]
            v_r = int(task["brush_r"])
            if not pts:
                continue
            append_pen_lift_glide(pts[0][0], pts[0][1])
            task_mask_bool = (task["mask"] < 245)
            for px, py in pts:
                timeline_steps.append(("draw_vector", int(px), int(py), v_r, task_mask_bool))
                last_nib_pos = (px, py)
            end_px, end_py = pts[-1]
            timeline_steps.append(("snap_vector", int(end_px), int(end_py), 0, task_mask_bool))

    num_steps = len(timeline_steps)
    hand_bgr, hand_alpha, tip_x, tip_y = preprocess_hand_image(hand_path, canvas_wd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_wd, canvas_ht))

    drawn_canvas = np.full((canvas_ht, canvas_wd, 3), 255, dtype=np.uint8)
    prev_step_idx = 0
    active_nib_pos = (float(timeline_steps[0][1]), float(timeline_steps[0][2]))

    for frame_i in range(active_draw_frames):
        exact_pos = ((frame_i + 1) / float(active_draw_frames)) * num_steps
        target_step_idx = min(num_steps, max(1, int(math.floor(exact_pos))))

        while prev_step_idx < target_step_idx:
            kind, px, py, radius, payload = timeline_steps[prev_step_idx]
            
            # SAFE NUMERICAL BOUNDS TO PREVENT C-LONG OVERFLOW
            px = int(np.clip(px, 0, canvas_wd - 1))
            py = int(np.clip(py, 0, canvas_ht - 1))
            radius = int(np.clip(radius, 1, 35))

            if kind == "draw_glyph":
                gx, gy, gw, gh = payload["bbox"]
                g_mask = payload["mask"] > 0
                y1 = int(max(gy, py - radius))
                y2 = int(min(gy + gh, py + radius + 1))
                x1 = int(max(gx, px - radius))
                x2 = int(min(gx + gw, px + radius + 1))

                if y2 > y1 and x2 > x1:
                    cy, cx = np.ogrid[y1 - py : y2 - py, x1 - px : x2 - px]
                    local_g = g_mask[y1 - gy : y2 - gy, x1 - gx : x2 - gx]
                    stamp = ((cx * cx + cy * cy) <= (radius * radius)) & local_g
                    drawn_canvas[y1:y2, x1:x2][stamp] = master_bgr[y1:y2, x1:x2][stamp]

            elif kind == "snap_glyph":
                gx, gy, gw, gh = payload["bbox"]
                g_mask = payload["mask"] > 0
                drawn_canvas[gy:gy+gh, gx:gx+gw][g_mask] = master_bgr[gy:gy+gh, gx:gx+gw][g_mask]

            elif kind == "draw_vector":
                y1 = int(max(0, py - radius))
                y2 = int(min(canvas_ht, py + radius + 1))
                x1 = int(max(0, px - radius))
                x2 = int(min(canvas_wd, px + radius + 1))

                if y2 > y1 and x2 > x1:
                    cy, cx = np.ogrid[y1 - py : y2 - py, x1 - px : x2 - px]
                    stamp = ((cx * cx + cy * cy) <= (radius * radius)) & payload[y1:y2, x1:x2]
                    drawn_canvas[y1:y2, x1:x2][stamp] = master_bgr[y1:y2, x1:x2][stamp]

            elif kind == "snap_vector":
                drawn_canvas[payload] = master_bgr[payload]

            active_nib_pos = (float(px), float(py))
            prev_step_idx += 1

        frac = exact_pos - target_step_idx
        if frac > 0.0 and 0 < target_step_idx < num_steps:
            p0 = timeline_steps[target_step_idx - 1]
            p1 = timeline_steps[target_step_idx]
            render_nib_pos = (float(p0[1] + frac * (p1[1] - p0[1])), float(p0[2] + frac * (p1[2] - p0[2])))
        else:
            render_nib_pos = active_nib_pos

        frame_view = drawn_canvas.copy()
        composite = draw_hand_with_extended_arm(
            frame_view, hand_bgr, hand_alpha, tip_x, tip_y,
            min(max(0.0, render_nib_pos[0]), float(canvas_wd - 1)),
            min(max(0.0, render_nib_pos[1]), float(canvas_ht - 1))
        )
        video_writer.write(composite)

    drawn_canvas[:] = master_bgr[:]

    if exit_frames > 0:
        start_x, start_y = active_nib_pos
        end_y = float(canvas_ht + tip_y + 20)
        for e_i in range(1, exit_frames + 1):
            t = e_i / float(exit_frames)
            ease_t = t * t
            cur_y = start_y + ease_t * (end_y - start_y)
            frame_view = drawn_canvas.copy()
            composite = draw_hand_with_extended_arm(
                frame_view, hand_bgr, hand_alpha, tip_x, tip_y,
                min(max(0.0, start_x), float(canvas_wd - 1)), cur_y
            )
            video_writer.write(composite)

    for _ in range(final_hold_frames):
        video_writer.write(drawn_canvas)

    video_writer.release()
    return output_path
