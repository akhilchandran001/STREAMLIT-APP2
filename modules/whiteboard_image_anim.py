import json
import math
import os
import sys
import time
import cv2
import numpy as np

BASE_DIR = os.getcwd()
DEFAULT_HAND_IMG = os.path.join(BASE_DIR, "assets", "whiteboard", "alpha-hand.png")

# Border & Canvas Configuration
BORDER_RED_BGR = (20, 20, 220)      # High-intensity Red
BORDER_THICKNESS = 4                # Clean crisp single-line border
BORDER_MARGIN_X = 14                # 14px from left/right edges
BORDER_MARGIN_Y = 12                # 12px from top/bottom edges (touches boundaries without overflow)


def euc_dist_sq(arr1, point):
    return np.sum((arr1 - point) ** 2, axis=1)


def get_extreme_coordinates(mask):
    indices = np.where(mask > 0)
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        return (0, 0), (mask.shape[1], mask.shape[0])
    topleft = (int(np.min(indices[1])), int(np.min(indices[0])))
    bottomright = (int(np.max(indices[1])), int(np.max(indices[0])))
    return topleft, bottomright


def preprocess_image_fullscreen(img_path, target_wd=1920, target_ht=1080):
    """
    Crops whitespace margins from the generated image, scales it up to maximize
    full-screen height with minimum gap top and bottom, and centers it on the canvas.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Input image not found: {img_path}")
    src = cv2.imread(img_path)
    if src is None:
        raise ValueError(f"Failed to read image from {img_path}")

    # 1. Auto-crop empty outer white margins so the artwork expands to maximum scale
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ink_ys, ink_xs = np.where(gray_src < 235)
    if len(ink_ys) > 0 and len(ink_xs) > 0:
        pad = 8
        x1 = max(0, int(np.min(ink_xs)) - pad)
        y1 = max(0, int(np.min(ink_ys)) - pad)
        x2 = min(src.shape[1], int(np.max(ink_xs)) + pad + 1)
        y2 = min(src.shape[0], int(np.max(ink_ys)) + pad + 1)
        content_crop = src[y1:y2, x1:x2]
    else:
        content_crop = src

    sh, sw = content_crop.shape[:2]

    # 2. Maximum dimensions inside the red border (leaving 8px inner breathing room)
    bx1 = BORDER_MARGIN_X
    by1 = BORDER_MARGIN_Y
    bx2 = target_wd - BORDER_MARGIN_X
    by2 = target_ht - BORDER_MARGIN_Y

    inner_pad = 8
    avail_w = (bx2 - bx1) - 2 * inner_pad   # ~1876px
    avail_h = (by2 - by1) - 2 * inner_pad   # ~1040px (fills 97% of 1080 height!)

    scale = min(avail_w / float(sw), avail_h / float(sh))
    nw = max(10, int(sw * scale))
    nh = max(10, int(sh * scale))
    resized_art = cv2.resize(content_crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # 3. Center inside master canvas without border yet (Border is drawn last by the hand)
    canvas = np.full((target_ht, target_wd, 3), 255, dtype=np.uint8)
    ox = (target_wd - nw) // 2
    oy = (target_ht - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = resized_art

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)
    return canvas, img_thresh, target_ht, target_wd, (bx1, by1, bx2, by2)


def extract_single_hand_rgba(hand_path, target_canvas_wd):
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
        hand_alpha = (hand_raw[:, :, 3] > 20).astype(np.float32) * (hand_raw[:, :, 3] / 255.0)
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

    binary_mask = (hand_alpha > 0.15).astype(np.uint8) * 255
    (tl_x, tl_y), (br_x, br_y) = get_extreme_coordinates(binary_mask)
    hand_bgr = hand_bgr[tl_y:br_y + 1, tl_x:br_x + 1]
    hand_alpha = hand_alpha[tl_y:br_y + 1, tl_x:br_x + 1]

    desired_hand_wd = max(160, int(target_canvas_wd * 0.38))
    scale = desired_hand_wd / float(hand_bgr.shape[1])
    new_wd = desired_hand_wd
    new_ht = int(hand_bgr.shape[0] * scale)

    hand_bgr = cv2.resize(hand_bgr, (new_wd, new_ht), interpolation=cv2.INTER_AREA)
    hand_alpha = cv2.resize(hand_alpha, (new_wd, new_ht), interpolation=cv2.INTER_AREA)
    hand_alpha = np.clip((hand_alpha - 0.12) / 0.88, 0.0, 1.0)
    hand_alpha = cv2.GaussianBlur(hand_alpha, (3, 3), 0.6)

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

    sample_h = min(50, new_ht // 4)
    row_top = (hand_alpha[new_ht - sample_h, :] > 0.3)
    row_bot = (hand_alpha[new_ht - 2, :] > 0.3)
    if np.any(row_top) and np.any(row_bot):
        cx_top = np.mean(np.where(row_top)[0])
        cx_bot = np.mean(np.where(row_bot)[0])
        arm_slope_x = float(cx_bot - cx_top) / float(max(1, sample_h - 2))
    else:
        arm_slope_x = 0.14

    ext_ht = int(target_canvas_wd * 0.65)
    pad_left = int(max(100, new_wd * 0.35))
    pad_right = int(max(160, new_wd * 0.45 + abs(arm_slope_x) * ext_ht))
    pad_top = int(max(60, new_ht * 0.15))

    full_ht = pad_top + new_ht + ext_ht
    full_wd = pad_left + new_wd + pad_right

    canvas_bgr = np.zeros((full_ht, full_wd, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((full_ht, full_wd), dtype=np.float32)

    canvas_bgr[pad_top:pad_top + new_ht, pad_left:pad_left + new_wd] = hand_bgr
    canvas_alpha[pad_top:pad_top + new_ht, pad_left:pad_left + new_wd] = hand_alpha

    src_y = pad_top + new_ht - 3
    strip_bgr = np.tile(canvas_bgr[src_y:src_y + 1, :, :], (ext_ht + 3, 1, 1))
    strip_alpha = np.tile(canvas_alpha[src_y:src_y + 1, :], (ext_ht + 3, 1))

    M_shear = np.float32([[1.0, arm_slope_x, 0.0], [0.0, 1.0, 0.0]])
    strip_bgr_sheared = cv2.warpAffine(strip_bgr, M_shear, (full_wd, ext_ht + 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    strip_alpha_sheared = cv2.warpAffine(strip_alpha, M_shear, (full_wd, ext_ht + 3), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    canvas_bgr[src_y:src_y + ext_ht + 3, :] = strip_bgr_sheared
    canvas_alpha[src_y:src_y + ext_ht + 3, :] = strip_alpha_sheared

    tip_x = int(tip_x + pad_left)
    tip_y = int(tip_y + pad_top)
    return canvas_bgr, canvas_alpha, tip_x, tip_y


class FastLivingHandEngine:
    def __init__(self, hand_path, orig_wd, orig_ht):
        self.orig_wd = orig_wd
        self.orig_ht = orig_ht
        base_bgr, base_alpha, self.tip_x, self.tip_y = extract_single_hand_rgba(hand_path, orig_wd)
        self.sprite_ht, self.sprite_wd = base_bgr.shape[:2]
        nib_center = (float(self.tip_x), float(self.tip_y))

        angles = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
        self.poses = []
        for ang in angles:
            M_rot = cv2.getRotationMatrix2D(nib_center, ang, 1.0)
            rot_bgr = cv2.warpAffine(base_bgr, M_rot, (self.sprite_wd, self.sprite_ht), flags=cv2.INTER_LINEAR)
            rot_alpha = cv2.warpAffine(base_alpha, M_rot, (self.sprite_wd, self.sprite_ht), flags=cv2.INTER_LINEAR)
            shadow_alpha = self._build_natural_shadow(rot_alpha, lifted=False)
            self.poses.append(self._pack_pose(rot_bgr, rot_alpha, shadow_alpha))

        M_lift = cv2.getRotationMatrix2D(nib_center, -2.5, 1.02)
        lift_bgr = cv2.warpAffine(base_bgr, M_lift, (self.sprite_wd, self.sprite_ht), flags=cv2.INTER_LINEAR)
        lift_alpha = cv2.warpAffine(base_alpha, M_lift, (self.sprite_wd, self.sprite_ht), flags=cv2.INTER_LINEAR)
        lift_shadow = self._build_natural_shadow(lift_alpha, lifted=True)
        self.lifted_pose = self._pack_pose(lift_bgr, lift_alpha, lift_shadow)

        self.curr_x = None
        self.curr_y = None
        self.prev_x = None
        self.prev_y = None
        self.smooth_idx = 4.0
        self.step_count = 0

    def _build_natural_shadow(self, alpha_mask, lifted=False):
        ds = 4
        small_w = max(16, self.sprite_wd // ds)
        small_h = max(16, self.sprite_ht // ds)
        small_alpha = cv2.resize(alpha_mask, (small_w, small_h), interpolation=cv2.INTER_AREA)

        ty_s = self.tip_y / float(ds)
        shear_x = -0.11 if not lifted else -0.15
        scale_y = 1.04 if not lifted else 1.06
        base_dx = (-1.0 if not lifted else -3.5) - shear_x * ty_s
        base_dy = (1.5 if not lifted else 4.5) + (1.0 - scale_y) * ty_s

        M_shadow = np.float32([[1.0, shear_x, base_dx], [0.0, scale_y, base_dy]])
        warped_s = cv2.warpAffine(small_alpha, M_shadow, (small_w, small_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        sigma_contact = 2.2 if not lifted else 3.8
        sigma_ambient = 9.5 if not lifted else 14.0
        blur_tight = cv2.GaussianBlur(warped_s, (0, 0), sigmaX=sigma_contact)
        blur_wide = cv2.GaussianBlur(warped_s, (0, 0), sigmaX=sigma_ambient)

        small_shadow = blur_tight * (0.08 if not lifted else 0.05) + blur_wide * (0.13 if not lifted else 0.11)
        full_shadow = cv2.resize(small_shadow, (self.sprite_wd, self.sprite_ht), interpolation=cv2.INTER_LINEAR)
        return np.clip(full_shadow, 0.0, 0.22).astype(np.float32)

    def _pack_pose(self, bgr, alpha, shadow):
        active_mask = ((alpha > 0.01) | (shadow > 0.005)).astype(np.uint8)
        (x1, y1), (x2, y2) = get_extreme_coordinates(active_mask)

        c_bgr = bgr[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
        c_alpha = alpha[y1:y2 + 1, x1:x2 + 1, np.newaxis]
        c_shadow = shadow[y1:y2 + 1, x1:x2 + 1, np.newaxis]

        premul_bgr = np.clip(c_bgr * c_alpha, 0, 255).astype(np.uint8)
        inv_trans = np.round((1.0 - c_alpha) * (1.0 - c_shadow) * 256.0).astype(np.uint16)

        return {
            "premul": premul_bgr,
            "inv_t": inv_trans,
            "ox": self.tip_x - x1,
            "oy": self.tip_y - y1,
            "w": x2 - x1 + 1,
            "h": y2 - y1 + 1,
        }

    def render_hand_in_place(self, canvas_view, target_x, target_y, is_lifted=False):
        if self.curr_x is None:
            self.curr_x, self.curr_y = float(target_x), float(target_y)
            self.prev_x, self.prev_y = float(target_x), float(target_y)

        spring = 0.60 if is_lifted else 0.82
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
        self.curr_x += spring * (target_x - self.curr_x)
        self.curr_y += spring * (target_y - self.curr_y)

        vx = self.curr_x - self.prev_x
        vy = self.curr_y - self.prev_y
        self.step_count += 1

        if is_lifted:
            pose = self.lifted_pose
            draw_x = int(round(self.curr_x))
            draw_y = int(round(self.curr_y - 8))
        else:
            norm_x = (self.curr_x / float(max(1, self.orig_wd))) - 0.5
            norm_y = (self.curr_y / float(max(1, self.orig_ht))) - 0.5
            elbow_shift = norm_x * 2.5 - norm_y * 1.5
            stroke_shift = float(np.clip((vy * 0.18 - vx * 0.12), -2.2, 2.2))
            micro_osc = 0.45 * math.sin(self.step_count * 0.55)

            target_idx = float(np.clip(4.0 + elbow_shift + stroke_shift + micro_osc, 0.0, 8.0))
            self.smooth_idx = 0.72 * self.smooth_idx + 0.28 * target_idx
            pose = self.poses[int(round(self.smooth_idx))]
            draw_x = int(round(self.curr_x))
            draw_y = int(round(self.curr_y))

        origin_x = draw_x - pose["ox"]
        origin_y = draw_y - pose["oy"]

        c_x1 = max(0, origin_x)
        c_y1 = max(0, origin_y)
        c_x2 = min(self.orig_wd, origin_x + pose["w"])
        c_y2 = min(self.orig_ht, origin_y + pose["h"])

        if c_x2 <= c_x1 or c_y2 <= c_y1:
            return None

        h_x1 = c_x1 - origin_x
        h_y1 = c_y1 - origin_y
        h_x2 = h_x1 + (c_x2 - c_x1)
        h_y2 = h_y1 + (c_y2 - c_y1)

        roi_slice = canvas_view[c_y1:c_y2, c_x1:c_x2]
        backup_patch = roi_slice.copy()
        crop_premul = pose["premul"][h_y1:h_y2, h_x1:h_x2]
        crop_inv_t = pose["inv_t"][h_y1:h_y2, h_x1:h_x2]

        canvas_view[c_y1:c_y2, c_x1:c_x2] = ((roi_slice.astype(np.uint16) * crop_inv_t) >> 8).astype(np.uint8) + crop_premul
        return (c_y1, c_y2, c_x1, c_x2, backup_patch)


def generate_box_perimeter_steps(x1, y1, x2, y2, step_px=3.0):
    """Generates continuous clockwise perimeter coordinates (Top -> Right -> Bottom -> Left)."""
    edges = [
        (x1, y1, x2, y1),  # Top: Left -> Right
        (x2, y1, x2, y2),  # Right: Top -> Bottom
        (x2, y2, x1, y2),  # Bottom: Right -> Left
        (x1, y2, x1, y1),  # Left: Bottom -> Top
    ]
    pts = []
    for sx, sy, ex, ey in edges:
        dist = math.hypot(ex - sx, ey - sy)
        n_pts = max(2, int(math.ceil(dist / float(step_px))) + 1)
        xs = np.linspace(sx, ex, n_pts)
        ys = np.linspace(sy, ey, n_pts)
        for idx in range(len(xs)):
            if idx == 0 and pts:
                continue
            pts.append((int(round(xs[idx])), int(round(ys[idx]))))
    return pts


def render_whiteboard_sequence(
    drawn_canvas,
    orig_bgr_padded,
    grid_of_cuts,
    cut_black_indices,
    split_len,
    orig_ht,
    orig_wd,
    hand_engine,
    video_writer,
    target_drawing_frames,
    current_frame_count,
):
    total_blocks = len(cut_black_indices)
    if total_blocks == 0:
        return current_frame_count, (orig_wd // 2, orig_ht // 2)

    selected_ind = 0
    blocks_drawn = 0
    prev_hand_pos = None
    lift_distance_thresh = max(75.0, orig_wd * 0.07)
    canvas_view = drawn_canvas[:orig_ht, :orig_wd]

    while len(cut_black_indices) > 0:
        selected_coord = cut_black_indices[selected_ind].copy()
        row_idx, col_idx = selected_coord[0], selected_coord[1]

        r_v_start = row_idx * split_len
        r_v_end = r_v_start + split_len
        r_h_start = col_idx * split_len
        r_h_end = r_h_start + split_len

        drawn_canvas[r_v_start:r_v_end, r_h_start:r_h_end] = orig_bgr_padded[r_v_start:r_v_end, r_h_start:r_h_end]

        hand_cx = min(r_h_start + (split_len // 2), orig_wd - 1)
        hand_cy = min(r_v_start + (split_len // 2), orig_ht - 1)
        curr_hand_pos = (hand_cx, hand_cy)

        if prev_hand_pos is None:
            prev_hand_pos = curr_hand_pos

        cut_black_indices[selected_ind] = cut_black_indices[-1]
        cut_black_indices = cut_black_indices[:-1]

        if len(cut_black_indices) > 0:
            distances = euc_dist_sq(cut_black_indices, selected_coord)
            selected_ind = int(np.argmin(distances))

        blocks_drawn += 1
        expected_frames = int((blocks_drawn / float(total_blocks)) * target_drawing_frames)
        frames_to_emit = min(expected_frames, target_drawing_frames) - current_frame_count

        if frames_to_emit > 0:
            jump_dist = math.hypot(curr_hand_pos[0] - prev_hand_pos[0], curr_hand_pos[1] - prev_hand_pos[1])
            is_long_jump = (jump_dist > lift_distance_thresh)

            for f_step in range(frames_to_emit):
                t = (f_step + 1) / float(frames_to_emit)
                interp_x = int(round(prev_hand_pos[0] + t * (curr_hand_pos[0] - prev_hand_pos[0])))
                interp_y = int(round(prev_hand_pos[1] + t * (curr_hand_pos[1] - prev_hand_pos[1])))
                lifted_now = is_long_jump and (f_step < frames_to_emit - 1 or frames_to_emit == 1)

                restore_info = hand_engine.render_hand_in_place(canvas_view, interp_x, interp_y, is_lifted=lifted_now)
                video_writer.write(canvas_view)
                if restore_info is not None:
                    cy1, cy2, cx1, cx2, patch = restore_info
                    canvas_view[cy1:cy2, cx1:cx2] = patch

                current_frame_count += 1

            prev_hand_pos = curr_hand_pos

    last_pos = prev_hand_pos if prev_hand_pos is not None else (orig_wd // 2, orig_ht // 2)
    return current_frame_count, last_pos


def generate_native_whiteboard_animation(
    img_path: str,
    output_path: str,
    hand_path: str = DEFAULT_HAND_IMG,
    total_duration_sec: float = 10.0,
    fps: int = 30,
    split_len: int = 10,
    black_pixel_threshold: int = 10,
    final_hold_sec: float = 1.0,
) -> str:
    """
    Renders whiteboard drawing:
    1. Full-screen auto-cropped illustration drawn first.
    2. Single-line red bounding box framing the screen drawn last.
    3. Hand exits smoothly off-screen into final hold.
    """
    total_duration_sec = max(2.0, float(total_duration_sec))
    orig_bgr, img_thresh, orig_ht, orig_wd, (bx1, by1, bx2, by2) = preprocess_image_fullscreen(img_path)

    pad_h = (split_len - (orig_ht % split_len)) % split_len
    pad_w = (split_len - (orig_wd % split_len)) % split_len

    if pad_h > 0 or pad_w > 0:
        thresh_padded = np.pad(img_thresh, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
        orig_bgr_padded = np.pad(orig_bgr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=255)
    else:
        thresh_padded = img_thresh
        orig_bgr_padded = orig_bgr

    padded_ht, padded_wd = thresh_padded.shape[:2]
    n_cuts_v = padded_ht // split_len
    n_cuts_h = padded_wd // split_len

    hand_engine = FastLivingHandEngine(hand_path, orig_wd, orig_ht)

    # Timing Budget
    total_target_frames = int(round(total_duration_sec * fps))
    final_hold_frames = min(int(round(final_hold_sec * fps)), int(total_target_frames * 0.16))
    remaining_frames = max(2, total_target_frames - final_hold_frames)
    exit_frames = min(8, max(3, int(round(0.26 * fps)))) if remaining_frames >= 20 else 0
    active_draw_frames = max(2, remaining_frames - exit_frames)

    # Allocate 84% time for drawing the illustration and 16% time for drawing the final red border
    box_draw_frames = max(14, min(int(active_draw_frames * 0.16), int(round(1.1 * fps))))
    if active_draw_frames - box_draw_frames < 20:
        box_draw_frames = max(8, int(active_draw_frames * 0.14))
    art_draw_frames = active_draw_frames - box_draw_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_wd, orig_ht))

    drawn_canvas = np.full((padded_ht, padded_wd, 3), 255, dtype=np.uint8)
    grid_cols = np.array(np.split(thresh_padded, n_cuts_h, axis=-1))
    grid_of_cuts = np.array(np.split(grid_cols, n_cuts_v, axis=-2))

    has_black = np.sum(np.sum((grid_of_cuts < black_pixel_threshold) * 1, axis=-1), axis=-1)
    black_coords = np.array(np.where(has_black > 0)).T

    current_frames = 0
    canvas_view = drawn_canvas[:orig_ht, :orig_wd]

    # =============================================================
    # STAGE 1: DRAW THE FULL-SCREEN ILLUSTRATION (STARS, GALAXY, PLANET)
    # =============================================================
    current_frames, last_nib = render_whiteboard_sequence(
        drawn_canvas, orig_bgr_padded, grid_of_cuts, black_coords,
        split_len, orig_ht, orig_wd, hand_engine, video_writer,
        art_draw_frames, current_frames
    )

    # Ensure all illustration ink is 100% crisp before border starts
    drawn_canvas[:orig_ht, :orig_wd] = orig_bgr[:orig_ht, :orig_wd]

    # =============================================================
    # STAGE 2: DRAW THE SINGLE-LINE RED BORDER AT THE VERY END
    # =============================================================
    box_pts = generate_box_perimeter_steps(bx1, by1, bx2, by2, step_px=3.0)
    total_box_steps = len(box_pts)

    # Glide from last illustration position to the top-left start of the red border
    start_bx, start_by = box_pts[0]
    glide_steps = 4
    for g_i in range(glide_steps):
        t = (g_i + 1) / float(glide_steps)
        gx = int(round(last_nib[0] + t * (start_bx - last_nib[0])))
        gy = int(round(last_nib[1] + t * (start_by - last_nib[1])))
        restore_info = hand_engine.render_hand_in_place(canvas_view, gx, gy, is_lifted=True)
        video_writer.write(canvas_view)
        if restore_info is not None:
            cy1, cy2, cx1, cx2, patch = restore_info
            canvas_view[cy1:cy2, cx1:cx2] = patch

    # Draw the red box clockwise
    prev_pt = (start_bx, start_by)
    prev_box_idx = 0

    for b_frame in range(box_draw_frames):
        prog = (b_frame + 1) / float(box_draw_frames)
        target_idx = min(total_box_steps, max(1, int(math.floor(prog * total_box_steps))))

        while prev_box_idx < target_idx:
            curr_pt = box_pts[prev_box_idx]
            cv2.line(canvas_view, prev_pt, curr_pt, BORDER_RED_BGR, BORDER_THICKNESS, cv2.LINE_AA)
            prev_pt = curr_pt
            prev_box_idx += 1

        nib_pos = box_pts[min(target_idx, total_box_steps - 1)]
        restore_info = hand_engine.render_hand_in_place(canvas_view, nib_pos[0], nib_pos[1], is_lifted=False)
        video_writer.write(canvas_view)
        if restore_info is not None:
            cy1, cy2, cx1, cx2, patch = restore_info
            canvas_view[cy1:cy2, cx1:cx2] = patch
        current_frames += 1

    # Close any sub-pixel seam on the red rectangle perfectly
    cv2.rectangle(canvas_view, (bx1, by1), (bx2, by2), BORDER_RED_BGR, BORDER_THICKNESS, cv2.LINE_AA)

    # =============================================================
    # STAGE 3: SMOOTH HAND EXIT & FINAL HOLD
    # =============================================================
    if exit_frames > 0:
        start_x, start_y = box_pts[0]
        end_y = float(orig_ht + 140)
        for e_i in range(1, exit_frames + 1):
            t = e_i / float(exit_frames)
            ease_t = t * t
            cur_y = start_y + ease_t * (end_y - start_y)
            restore_info = hand_engine.render_hand_in_place(canvas_view, start_x, int(cur_y), is_lifted=True)
            video_writer.write(canvas_view)
            if restore_info is not None:
                cy1, cy2, cx1, cx2, patch = restore_info
                canvas_view[cy1:cy2, cx1:cx2] = patch

    for _ in range(final_hold_frames):
        video_writer.write(canvas_view)

    video_writer.release()
    return output_path
