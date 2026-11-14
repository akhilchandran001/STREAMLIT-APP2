import json
import math
import os
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np

# Resolve repository root and bundled hand assets (assets/whiteboard/)
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MODULE_DIR)
DEFAULT_HAND_PATH = os.path.join(
    REPO_ROOT, "assets", "whiteboard", "alpha-hand.png"
)
FALLBACK_HAND_PATH = os.path.join(
    REPO_ROOT, "assets", "whiteboard", "alpha-hand_2.png"
)


def euc_dist_sq(arr1, point):
  """Squared Euclidean distance for fast monotonic nearest-neighbor search."""
  return np.sum((arr1 - point) ** 2, axis=1)


def get_extreme_coordinates(mask):
  """Locate tight bounding box around foreground mask."""
  indices = np.where(mask > 0)
  if len(indices[0]) == 0 or len(indices[1]) == 0:
    return (0, 0), (mask.shape[1] - 1, mask.shape[0] - 1)
  topleft = (int(np.min(indices[1])), int(np.min(indices[0])))
  bottomright = (int(np.max(indices[1])), int(np.max(indices[0])))
  return topleft, bottomright


def preprocess_image_fullscreen(img_path, target_wd=1920, target_ht=1080):
  """Legacy helper preserved for repo backwards-compatibility:

  Crops whitespace margins from an illustration and fits it onto a white canvas.
  """
  if not os.path.exists(img_path):
    raise FileNotFoundError(f"Input image not found: {img_path}")

  img = cv2.imread(img_path)
  if img is None:
    raise ValueError(f"Failed to read image from {img_path}")

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
  (x1, y1), (x2, y2) = get_extreme_coordinates(thresh)
  cropped = img[y1 : y2 + 1, x1 : x2 + 1]

  ch, cw = cropped.shape[:2]
  scale = min((target_wd * 0.90) / max(1, cw), (target_ht * 0.90) / max(1, ch))
  nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
  resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)

  canvas = np.full((target_ht, target_wd, 3), 255, dtype=np.uint8)
  ox = (target_wd - nw) // 2
  oy = (target_ht - nh) // 2
  canvas[oy : oy + nh, ox : ox + nw] = resized

  canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
  _, canvas_thresh = cv2.threshold(canvas_gray, 220, 255, cv2.THRESH_BINARY)
  return canvas, canvas_thresh


def generate_box_perimeter_steps(x1, y1, x2, y2, step_px=12):
  """Legacy helper preserved for repo backwards-compatibility:

  Generates continuous clockwise perimeter coordinates (Top -> Right -> Bottom ->
  Left).
  """
  pts = []
  for x in range(x1, x2 + 1, step_px):
    pts.append((x, y1))
  for y in range(y1, y2 + 1, step_px):
    pts.append((x2, y))
  for x in range(x2, x1 - 1, -step_px):
    pts.append((x, y2))
  for y in range(y2, y1 - 1, -step_px):
    pts.append((x1, y))
  return pts


def resolve_hand_asset_path(hand_path=None):
  """Resolves the hand sprite path from repo assets (assets/whiteboard/)."""
  candidates = []
  if hand_path:
    candidates.append(hand_path)
    if not os.path.isabs(hand_path):
      candidates.append(os.path.join(REPO_ROOT, hand_path))
  candidates.extend([DEFAULT_HAND_PATH, FALLBACK_HAND_PATH])

  for path in candidates:
    if path and os.path.exists(path):
      return path

  raise FileNotFoundError(f"Hand sprite not found. Checked: {candidates}")


def extract_masks_from_bgr(img_bgr):
  """Extracts fg_mask, ink_mask, and fill_mask (including pale highlights)."""
  ht, wd = img_bgr.shape[:2]

  border_pixels = np.concatenate(
      [
          img_bgr[0, :, :],
          img_bgr[-1, :, :],
          img_bgr[:, 0, :],
          img_bgr[:, -1, :],
      ],
      axis=0,
  )
  bg_bgr = np.median(border_pixels, axis=0).astype(np.uint8)
  img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
  bg_lab = cv2.cvtColor(bg_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2LAB).astype(
      np.float32
  )[0, 0]

  delta_e = np.linalg.norm(img_lab - bg_lab, axis=2)
  # Lowered to 7.5 so subtle metallic/glass tints are also shaded cleanly
  fg_mask = (delta_e > 7.5).astype(np.uint8) * 255
  fg_mask = cv2.morphologyEx(
      fg_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
  )

  dist_map = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
  max_line_half_w = max(4.0, min(wd, ht) * 0.007)

  gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
  sat = hsv[:, :, 1]

  dark_ink = ((gray < 95) & (sat < 85)) | (gray < 50)

  bh_size = max(9, (int(min(wd, ht) * 0.015) | 1))
  kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_size, bh_size))
  blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
  thin_colored_lines = (
      (dist_map <= max_line_half_w) & (blackhat > 30) & (gray < 175)
  )

  ink_mask = ((dark_ink | thin_colored_lines) & (fg_mask > 0)).astype(
      np.uint8
  ) * 255
  ink_mask = cv2.morphologyEx(
      ink_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
  )

  erode_r = max(3, int(min(wd, ht) * 0.006))
  erode_k = cv2.getStructuringElement(
      cv2.MORPH_ELLIPSE, (erode_r * 2 + 1, erode_r * 2 + 1)
  )
  solid_black_cores = cv2.erode(
      (dark_ink & (fg_mask > 0)).astype(np.uint8) * 255, erode_k, iterations=1
  )

  non_ink_fg = cv2.bitwise_and(fg_mask, cv2.bitwise_not(ink_mask))
  raw_fills = cv2.bitwise_or(non_ink_fg, solid_black_cores)

  if np.any(raw_fills > 0):
    fill_mask = cv2.dilate(raw_fills, np.ones((5, 5), np.uint8), iterations=1)
    fill_mask = cv2.bitwise_and(fill_mask, fg_mask)
  else:
    fill_mask = np.zeros_like(fg_mask)

  if np.any(solid_black_cores > 0):
    ink_mask = cv2.bitwise_and(ink_mask, cv2.bitwise_not(solid_black_cores))

  return fg_mask, ink_mask, fill_mask


def decompose_artwork_layers(img_path, max_work_dim=960):
  """Loads native image for crisp rendering, plus a fast working-resolution

  copy (~960px) so path planning finishes in <0.3s even on 4K images.
  """
  if not os.path.exists(img_path):
    raise FileNotFoundError(f"Input image not found: {img_path}")

  orig_bgr = cv2.imread(img_path)
  if orig_bgr is None:
    raise ValueError(f"Failed to read image from {img_path}")

  orig_ht, orig_wd = orig_bgr.shape[:2]
  work_scale = min(1.0, float(max_work_dim) / float(max(orig_wd, orig_ht)))

  if work_scale < 0.99:
    work_wd = max(64, int(round(orig_wd * work_scale)))
    work_ht = max(64, int(round(orig_ht * work_scale)))
    work_bgr = cv2.resize(
        orig_bgr, (work_wd, work_ht), interpolation=cv2.INTER_AREA
    )
    native_fg, native_ink, _ = extract_masks_from_bgr(orig_bgr)
    work_fg, work_ink, work_fill = extract_masks_from_bgr(work_bgr)
  else:
    work_scale = 1.0
    work_wd, work_ht = orig_wd, orig_ht
    work_bgr = orig_bgr
    native_fg, native_ink, work_fill = extract_masks_from_bgr(orig_bgr)
    work_fg, work_ink = native_fg, native_ink

  return (
      orig_bgr,
      native_fg,
      native_ink,
      orig_ht,
      orig_wd,
      work_bgr,
      work_fg,
      work_ink,
      work_fill,
      work_ht,
      work_wd,
      work_scale,
  )


def extract_single_hand_rgba(hand_path, target_canvas_wd, target_canvas_ht):
  """Loads the hand image at full original scale (36% canvas width),

  isolates alpha, finds the exact marker nib, and extends the forearm.
  """
  resolved_hand = resolve_hand_asset_path(hand_path)

  hand_raw = cv2.imread(resolved_hand, cv2.IMREAD_UNCHANGED)
  if hand_raw is None:
    raise ValueError(f"Failed to load hand image from {resolved_hand}")

  if hand_raw.ndim == 3 and hand_raw.shape[2] == 4:
    hand_bgr = hand_raw[:, :, :3]
    hand_alpha = (hand_raw[:, :, 3] > 20).astype(np.float32) * (
        hand_raw[:, :, 3] / 255.0
    )
  else:
    hand_bgr = hand_raw[:, :, :3]
    h_temp, w_temp = hand_bgr.shape[:2]
    ff_mask = np.zeros((h_temp + 2, w_temp + 2), np.uint8)
    gray_temp = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2GRAY)
    for seed in [
        (0, 0),
        (w_temp - 1, 0),
        (0, h_temp - 1),
        (w_temp - 1, h_temp - 1),
    ]:
      cv2.floodFill(gray_temp.copy(), ff_mask, seed, 0, loDiff=15, upDiff=15)
    is_bg = ff_mask[1:-1, 1:-1] == 1
    hand_alpha = np.where(is_bg, 0.0, 1.0).astype(np.float32)

  binary_mask = (hand_alpha > 0.15).astype(np.uint8) * 255
  (tl_x, tl_y), (br_x, br_y) = get_extreme_coordinates(binary_mask)
  hand_bgr = hand_bgr[tl_y : br_y + 1, tl_x : br_x + 1]
  hand_alpha = hand_alpha[tl_y : br_y + 1, tl_x : br_x + 1]

  # Original 36% hand width preserved
  desired_hand_wd = max(160, int(target_canvas_wd * 0.36))
  scale = desired_hand_wd / float(hand_bgr.shape[1])
  new_wd = desired_hand_wd
  new_ht = int(hand_bgr.shape[0] * scale)

  hand_bgr = cv2.resize(
      hand_bgr, (new_wd, new_ht), interpolation=cv2.INTER_AREA
  )
  hand_alpha = cv2.resize(
      hand_alpha, (new_wd, new_ht), interpolation=cv2.INTER_AREA
  )

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
  tip_zone_gray = cv2.cvtColor(
      tip_zone_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY
  ).flatten()
  dark_pixels = tip_zone_gray < 110

  if np.sum(dark_pixels) > 2:
    tip_x = int(np.mean(cand_x[dark_pixels]))
    tip_y = int(np.mean(cand_y[dark_pixels]))
  else:
    tip_x = int(np.mean(cand_x))
    tip_y = int(np.mean(cand_y))

  sample_h = min(50, new_ht // 4)
  row_top = hand_alpha[new_ht - sample_h, :] > 0.3
  row_bot = hand_alpha[new_ht - 2, :] > 0.3
  if np.any(row_top) and np.any(row_bot):
    cx_top = np.mean(np.where(row_top)[0])
    cx_bot = np.mean(np.where(row_bot)[0])
    arm_slope_x = float(cx_bot - cx_top) / float(max(1, sample_h - 2))
  else:
    arm_slope_x = 0.14

  ext_ht = max(60, int(target_canvas_ht * 1.05) - new_ht)
  pad_left = int(max(80, new_wd * 0.25))
  pad_right = int(max(120, new_wd * 0.32 + abs(arm_slope_x) * ext_ht))
  pad_top = int(max(40, new_ht * 0.10))

  full_ht = pad_top + new_ht + ext_ht
  full_wd = pad_left + new_wd + pad_right

  canvas_bgr = np.zeros((full_ht, full_wd, 3), dtype=np.uint8)
  canvas_alpha = np.zeros((full_ht, full_wd), dtype=np.float32)

  canvas_bgr[pad_top : pad_top + new_ht, pad_left : pad_left + new_wd] = (
      hand_bgr
  )
  canvas_alpha[pad_top : pad_top + new_ht, pad_left : pad_left + new_wd] = (
      hand_alpha
  )

  src_y = pad_top + new_ht - 3
  strip_bgr = np.tile(canvas_bgr[src_y : src_y + 1, :, :], (ext_ht + 3, 1, 1))
  strip_alpha = np.tile(canvas_alpha[src_y : src_y + 1, :], (ext_ht + 3, 1))

  M_shear = np.float32([[1.0, arm_slope_x, 0.0], [0.0, 1.0, 0.0]])
  strip_bgr_sheared = cv2.warpAffine(
      strip_bgr,
      M_shear,
      (full_wd, ext_ht + 3),
      flags=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_REPLICATE,
  )
  strip_alpha_sheared = cv2.warpAffine(
      strip_alpha,
      M_shear,
      (full_wd, ext_ht + 3),
      flags=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_CONSTANT,
      borderValue=0.0,
  )

  canvas_bgr[src_y : src_y + ext_ht + 3, :] = strip_bgr_sheared
  canvas_alpha[src_y : src_y + ext_ht + 3, :] = strip_alpha_sheared

  tip_x = int(tip_x + pad_left)
  tip_y = int(tip_y + pad_top)

  return canvas_bgr, canvas_alpha, tip_x, tip_y


class FastLivingHandEngine:
  """Pre-baked wrist-pivot angles + dynamic parabolic lift shadow engine."""

  def __init__(self, hand_path, orig_wd, orig_ht):
    self.orig_wd = orig_wd
    self.orig_ht = orig_ht

    base_bgr, base_alpha, self.tip_x, self.tip_y = extract_single_hand_rgba(
        hand_path, orig_wd, orig_ht
    )
    self.sprite_ht, self.sprite_wd = base_bgr.shape[:2]
    nib_center = (float(self.tip_x), float(self.tip_y))

    angles = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    self.poses = []

    for ang in angles:
      M_rot = cv2.getRotationMatrix2D(nib_center, ang, 1.0)
      rot_bgr = cv2.warpAffine(
          base_bgr,
          M_rot,
          (self.sprite_wd, self.sprite_ht),
          flags=cv2.INTER_LINEAR,
      )
      rot_alpha = cv2.warpAffine(
          base_alpha,
          M_rot,
          (self.sprite_wd, self.sprite_ht),
          flags=cv2.INTER_LINEAR,
      )
      shadow_alpha = self._build_natural_shadow(rot_alpha, lifted=False)
      self.poses.append(self._pack_pose(rot_bgr, rot_alpha, shadow_alpha))

    M_lift = cv2.getRotationMatrix2D(nib_center, -3.0, 1.02)
    lift_bgr = cv2.warpAffine(
        base_bgr,
        M_lift,
        (self.sprite_wd, self.sprite_ht),
        flags=cv2.INTER_LINEAR,
    )
    lift_alpha = cv2.warpAffine(
        base_alpha,
        M_lift,
        (self.sprite_wd, self.sprite_ht),
        flags=cv2.INTER_LINEAR,
    )
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
    small_alpha = cv2.resize(
        alpha_mask, (small_w, small_h), interpolation=cv2.INTER_AREA
    )

    ty_s = self.tip_y / float(ds)
    shear_x = -0.11 if not lifted else -0.16
    scale_y = 1.04 if not lifted else 1.07
    base_dx = (-1.0 if not lifted else -4.5) - shear_x * ty_s
    base_dy = (1.5 if not lifted else 5.5) + (1.0 - scale_y) * ty_s

    M_shadow = np.float32([[1.0, shear_x, base_dx], [0.0, scale_y, base_dy]])
    warped_s = cv2.warpAffine(
        small_alpha,
        M_shadow,
        (small_w, small_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    sigma_contact = 2.2 if not lifted else 4.2
    sigma_ambient = 9.5 if not lifted else 15.0
    blur_tight = cv2.GaussianBlur(warped_s, (0, 0), sigmaX=sigma_contact)
    blur_wide = cv2.GaussianBlur(warped_s, (0, 0), sigmaX=sigma_ambient)

    small_shadow = blur_tight * (0.08 if not lifted else 0.04) + blur_wide * (
        0.13 if not lifted else 0.11
    )
    full_shadow = cv2.resize(
        small_shadow,
        (self.sprite_wd, self.sprite_ht),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.clip(full_shadow, 0.0, 0.22).astype(np.float32)

  def _pack_pose(self, bgr, alpha, shadow):
    active_mask = ((alpha > 0.01) | (shadow > 0.005)).astype(np.uint8)
    (x1, y1), (x2, y2) = get_extreme_coordinates(active_mask)

    c_bgr = bgr[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32)
    c_alpha = alpha[y1 : y2 + 1, x1 : x2 + 1, np.newaxis]
    c_shadow = shadow[y1 : y2 + 1, x1 : x2 + 1, np.newaxis]

    premul_bgr = np.clip(c_bgr * c_alpha, 0, 255).astype(np.uint8)
    inv_trans = np.round((1.0 - c_alpha) * (1.0 - c_shadow) * 256.0).astype(
        np.uint16
    )

    return {
        "premul": premul_bgr,
        "inv_t": inv_trans,
        "ox": self.tip_x - x1,
        "oy": self.tip_y - y1,
        "w": x2 - x1 + 1,
        "h": y2 - y1 + 1,
    }

  def render_hand_in_place(
      self, canvas_view, target_x, target_y, lift_offset_y=0.0
  ):
    if self.curr_x is None:
      self.curr_x, self.curr_y = float(target_x), float(target_y)
      self.prev_x, self.prev_y = float(target_x), float(target_y)

    is_lifted = lift_offset_y > 1.5
    spring = 0.65 if is_lifted else 0.90
    self.prev_x, self.prev_y = self.curr_x, self.curr_y
    self.curr_x += spring * (target_x - self.curr_x)
    self.curr_y += spring * (target_y - self.curr_y)

    vx = self.curr_x - self.prev_x
    vy = self.curr_y - self.prev_y
    self.step_count += 1

    if is_lifted:
      pose = self.lifted_pose
      draw_x = int(round(self.curr_x))
      draw_y = int(round(self.curr_y - lift_offset_y))
    else:
      norm_x = (self.curr_x / float(max(1, self.orig_wd))) - 0.5
      norm_y = (self.curr_y / float(max(1, self.orig_ht))) - 0.5
      elbow_shift = norm_x * 2.5 - norm_y * 1.5
      stroke_shift = float(np.clip((vy * 0.18 - vx * 0.12), -2.2, 2.2))
      micro_osc = 0.45 * math.sin(self.step_count * 0.55)

      target_idx = float(
          np.clip(4.0 + elbow_shift + stroke_shift + micro_osc, 0.0, 8.0)
      )
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

    canvas_view[c_y1:c_y2, c_x1:c_x2] = (
        (roi_slice.astype(np.uint16) * crop_inv_t) >> 8
    ).astype(np.uint8) + crop_premul

    return (c_y1, c_y2, c_x1, c_x2, backup_patch)


def segment_scene_objects(work_fg, work_wd, work_ht):
  """Groups connected details on the working grid and sorts Left-to-Right."""
  clean_fg = cv2.morphologyEx(
      work_fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
  )
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
      clean_fg, connectivity=8
  )

  min_major_area = max(80, int(work_wd * work_ht * 0.0005))
  major_objs = []
  tiny_bits = []

  for lbl in range(1, num_labels):
    area = stats[lbl, cv2.CC_STAT_AREA]
    if area < 4:
      continue
    cx, cy = centroids[lbl]
    mask = (labels == lbl).astype(np.uint8) * 255
    if area >= min_major_area:
      major_objs.append({"cx": cx, "cy": cy, "area": area, "mask": mask})
    else:
      tiny_bits.append({"cx": cx, "cy": cy, "mask": mask})

  if len(major_objs) == 0:
    return [work_fg]

  for bit in tiny_bits:
    best_idx = 0
    best_dist = float("inf")
    for idx, obj in enumerate(major_objs):
      d = math.hypot(bit["cx"] - obj["cx"], bit["cy"] - obj["cy"])
      if d < best_dist:
        best_dist = d
        best_idx = idx
    major_objs[best_idx]["mask"] = cv2.bitwise_or(
        major_objs[best_idx]["mask"], bit["mask"]
    )

  col_band_w = max(50, int(work_wd * 0.22))
  major_objs.sort(
      key=lambda o: (int(o["cx"]) // col_band_w, int(o["cy"]), -o["area"])
  )

  zone_masks = []
  claimed = np.zeros_like(work_fg)
  for obj in major_objs:
    dil = cv2.dilate(obj["mask"], np.ones((7, 7), np.uint8), iterations=1)
    dil = cv2.bitwise_and(dil, cv2.bitwise_not(claimed))
    claimed = cv2.bitwise_or(claimed, dil)
    zone_masks.append(dil)

  return zone_masks


def build_contour_outline_strokes(work_ink, work_wd, work_ht, inv_scale):
  """Extracts smooth vector contours ordered via Greedy Nearest-Neighbor chaining

  from top-left to bottom-right so:
  - Main structural outlines draw calmly at uniform pixel speed.
  - Drawn ink emerges above and to the left of the hand (never occluded).
  - Micro-stipples never cause erratic cross-canvas teleporting.
  """
  if not np.any(work_ink > 0):
    return []

  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
      work_ink, connectivity=8
  )

  work_r = max(4, int(min(work_wd, work_ht) * 0.0105))
  step_px = max(3, int(work_r * 0.90))
  native_r = max(7, int(round(work_r * inv_scale)))

  raw_strokes = []
  for lbl in range(1, num_labels):
    area = stats[lbl, cv2.CC_STAT_AREA]
    if area < 4:
      continue
    comp_mask = (labels == lbl).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        comp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
      pts = cnt.reshape(-1, 2)
      if len(pts) == 0:
        continue

      # Rotate contour loop to start at its top-leftmost point (min x + y)
      # so drawing flows downward/rightward, keeping fresh ink uncovered!
      tl_idx = int(
          np.argmin(pts[:, 0].astype(np.int32) + pts[:, 1].astype(np.int32))
      )
      pts = np.roll(pts, -tl_idx, axis=0)

      sampled = pts[::step_px]
      if len(sampled) == 0:
        sampled = pts[:1]

      native_pts = [
          (int(round(px * inv_scale)), int(round(py * inv_scale)))
          for px, py in sampled
      ]
      if len(pts) > step_px * 2:
        sx, sy = pts[0]
        native_pts.append(
            (int(round(sx * inv_scale)), int(round(sy * inv_scale)))
        )

      # Calculate true physical arc length in native pixels
      arc_len = 0.0
      for i in range(1, len(native_pts)):
        arc_len += math.hypot(
            native_pts[i][0] - native_pts[i - 1][0],
            native_pts[i][1] - native_pts[i - 1][1],
        )

      raw_strokes.append(
          {
              "pts": native_pts,
              "start": native_pts[0],
              "end": native_pts[-1],
              "arc_len": max(4.0, arc_len),
              "is_major": area >= 110,
          }
      )

  if len(raw_strokes) == 0:
    return []

  # Greedy Nearest-Neighbor chaining: draw major structural outlines first in a
  # calm top-left -> bottom-right flow, then nearby texture/grain details.
  major_strokes = [s for s in raw_strokes if s["is_major"]]
  detail_strokes = [s for s in raw_strokes if not s["is_major"]]

  def chain_strokes(stroke_pool, start_pos):
    ordered = []
    remaining = list(stroke_pool)
    cur_x, cur_y = start_pos
    while remaining:
      best_i = 0
      best_cost = float("inf")
      for idx, st in enumerate(remaining):
        sx, sy = st["start"]
        # Euclidean proximity + gentle top-to-bottom preference
        cost = math.hypot(sx - cur_x, sy - cur_y) + 0.25 * sy
        if cost < best_cost:
          best_cost = cost
          best_i = idx
      chosen = remaining.pop(best_i)
      ordered.append(chosen)
      cur_x, cur_y = chosen["end"]
    return ordered, (cur_x, cur_y)

  ordered_all = []
  anchor = (0, 0)
  if major_strokes:
    chained_major, anchor = chain_strokes(major_strokes, anchor)
    ordered_all.extend(chained_major)
  if detail_strokes:
    chained_detail, _ = chain_strokes(detail_strokes, anchor)
    ordered_all.extend(chained_detail)

  # Convert chained strokes into arc-length-weighted waypoints
  waypoints = []
  prev_pt = None
  for st in ordered_all:
    pts = st["pts"]
    is_detail = not st["is_major"]
    for idx, (nx, ny) in enumerate(pts):
      is_new = idx == 0
      is_end = idx == len(pts) - 1
      if is_new:
        jump_d = (
            math.hypot(nx - prev_pt[0], ny - prev_pt[1])
            if prev_pt is not None
            else 10.0
        )
        # Keep jumps crisp so travel time doesn't steal from actual drawing
        seg_w = min(22.0, max(3.0, jump_d * 0.08))
      else:
        seg_d = math.hypot(nx - prev_pt[0], ny - prev_pt[1])
        # Tiny stipple dots get compact weight so they never stall the hand
        seg_w = max(2.0, seg_d * (0.45 if is_detail else 1.0))

      waypoints.append((nx, ny, native_r, is_new, is_end, seg_w))
      prev_pt = (nx, ny)

  return waypoints


def build_smooth_color_patches(
    work_fill,
    work_bgr,
    work_wd,
    work_ht,
    native_fg,
    orig_wd,
    orig_ht,
    inv_scale,
):
  """Chains color regions via Greedy Nearest-Neighbor from top-left to bottom-right

  so color shading flows calmly down the object without jumping back and forth.
  """
  if not np.any(work_fill > 0):
    return []

  smoothed = cv2.bilateralFilter(work_bgr, 5, 35, 35)
  img_lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)

  q_l = (img_lab[:, :, 0] // 45).astype(np.int32)
  q_a = (img_lab[:, :, 1] // 32).astype(np.int32)
  q_b = (img_lab[:, :, 2] // 32).astype(np.int32)
  color_code = q_l * 100 + q_a * 10 + q_b

  fg_ys, fg_xs = np.where(work_fill > 0)
  unique_codes = np.unique(color_code[fg_ys, fg_xs])

  min_patch_area = max(35, int(work_wd * work_ht * 0.00018))
  raw_patches = []
  sliver_mask = np.zeros_like(work_fill)

  for code in unique_codes:
    c_mask = ((work_fill > 0) & (color_code == code)).astype(np.uint8) * 255
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(
        c_mask, connectivity=8
    )
    for lbl in range(1, num_l):
      area = stats[lbl, cv2.CC_STAT_AREA]
      if area < min_patch_area:
        sliver_mask[labels == lbl] = 255
      else:
        x, y, w, h, _ = stats[lbl]
        tight_sub = (labels[y : y + h, x : x + w] == lbl).astype(np.uint8) * 255
        raw_patches.append(
            {"area": area, "rx": x, "ry": y, "rw": w, "rh": h, "sub": tight_sub}
        )

  raw_patches.sort(key=lambda p: (-p["area"], p["ry"], p["rx"]))
  if len(raw_patches) > 65:
    for extra in raw_patches[65:]:
      rx, ry, rw, rh = extra["rx"], extra["ry"], extra["rw"], extra["rh"]
      sliver_mask[ry : ry + rh, rx : rx + rw][extra["sub"] > 0] = 255
    raw_patches = raw_patches[:65]

  id_map = np.zeros((work_ht, work_wd), dtype=np.uint16)
  for idx, p in enumerate(raw_patches, start=1):
    rx, ry, rw, rh = p["rx"], p["ry"], p["rw"], p["rh"]
    roi = id_map[ry : ry + rh, rx : rx + rw]
    roi[p["sub"] > 0] = idx

  if len(raw_patches) > 0 and np.any(sliver_mask > 0):
    dil_ids = cv2.dilate(id_map, np.ones((9, 9), np.uint8), iterations=1)
    absorb = (sliver_mask > 0) & (id_map == 0) & (dil_ids > 0)
    id_map[absorb] = dil_ids[absorb]
    sliver_mask[absorb] = 0

  if np.any(sliver_mask > 0):
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(
        sliver_mask, connectivity=8
    )
    next_id = len(raw_patches) + 1
    for lbl in range(1, num_l):
      if stats[lbl, cv2.CC_STAT_AREA] < 4:
        continue
      id_map[labels == lbl] = next_id
      next_id += 1

  max_id = int(np.max(id_map))
  if max_id == 0:
    return []

  merged_patches = []
  for pid in range(1, max_id + 1):
    ys, xs = np.where(id_map == pid)
    if len(ys) == 0:
      continue
    x1, y1 = int(np.min(xs)), int(np.min(ys))
    x2, y2 = int(np.max(xs)), int(np.max(ys))
    sub = (id_map[y1 : y2 + 1, x1 : x2 + 1] == pid).astype(np.uint8) * 255
    merged_patches.append(
        {
            "area": len(ys),
            "rx": x1,
            "ry": y1,
            "rw": x2 - x1 + 1,
            "rh": y2 - y1 + 1,
            "cx": float(np.mean(xs)),
            "cy": float(np.mean(ys)),
            "sub": sub,
        }
    )

  # Chain color patches by Greedy Nearest-Neighbor from top-left to bottom-right
  # so coloring flows naturally down the object without cross-screen hops!
  ordered_patches = []
  remaining = list(merged_patches)
  cur_x, cur_y = 0.0, 0.0
  while remaining:
    best_i = 0
    best_cost = float("inf")
    for idx, p in enumerate(remaining):
      cost = math.hypot(p["cx"] - cur_x, p["cy"] - cur_y) + 0.35 * p["cy"]
      if cost < best_cost:
        best_cost = cost
        best_i = idx
    chosen = remaining.pop(best_i)
    ordered_patches.append(chosen)
    cur_x, cur_y = chosen["cx"], chosen["cy"]

  native_stripe_w = max(16, int(min(orig_wd, orig_ht) * 0.022))
  stripes = []

  for patch in ordered_patches:
    nrx = max(0, int(math.floor(patch["rx"] * inv_scale)))
    nry = max(0, int(math.floor(patch["ry"] * inv_scale)))
    nrx2 = min(orig_wd, int(math.ceil((patch["rx"] + patch["rw"]) * inv_scale)))
    nry2 = min(orig_ht, int(math.ceil((patch["ry"] + patch["rh"]) * inv_scale)))
    nrw = nrx2 - nrx
    nrh = nry2 - nry
    if nrw <= 0 or nrh <= 0:
      continue

    up_sub = cv2.resize(
        patch["sub"], (nrw, nrh), interpolation=cv2.INTER_LINEAR
    )
    nat_sub = ((up_sub > 110) & (native_fg[nry:nry2, nrx:nrx2] > 0)).astype(
        np.uint8
    )

    ys, xs = np.where(nat_sub > 0)
    if len(ys) == 0:
      continue

    diag_coord = xs + ys
    band_ids = diag_coord // native_stripe_w
    unique_bands = np.unique(band_ids)

    for b_idx, b_id in enumerate(unique_bands):
      in_band = np.where(band_ids == b_id)[0]
      if len(in_band) == 0:
        continue

      bx = xs[in_band]
      by = ys[in_band]
      u = bx - by
      u_min = int(np.min(u))
      u_max = int(np.max(u))

      reverse_dir = (b_idx % 2) == 1
      if u_max == u_min:
        prog = np.zeros(len(in_band), dtype=np.float32)
        start_pt = (int(nrx + bx[0]), int(nry + by[0]))
        end_pt = start_pt
      else:
        if not reverse_dir:
          prog = (u - u_min).astype(np.float32) / float(u_max - u_min)
          i_start = int(np.argmin(u))
          i_end = int(np.argmax(u))
        else:
          prog = (u_max - u).astype(np.float32) / float(u_max - u_min)
          i_start = int(np.argmax(u))
          i_end = int(np.argmin(u))
        start_pt = (int(nrx + bx[i_start]), int(nry + by[i_start]))
        end_pt = (int(nrx + bx[i_end]), int(nry + by[i_end]))

      stripes.append(
          {
              "rx": nrx,
              "ry": nry,
              "bx": bx,
              "by": by,
              "prog": prog,
              "start_pt": start_pt,
              "end_pt": end_pt,
              "is_new_region": b_idx == 0,
              "pixel_count": len(in_band),
          }
      )

  return stripes


def update_canvas_from_mask_roi(
    canvas_view, reveal_mask, orig_bgr, x1, y1, x2, y2
):
  """Updates only the modified bounding box ROI on the whiteboard canvas."""
  if x2 <= x1 or y2 <= y1:
    return
  alpha = reveal_mask[y1:y2, x1:x2, np.newaxis].astype(np.uint16)
  orig_roi = orig_bgr[y1:y2, x1:x2].astype(np.uint16)
  canvas_view[y1:y2, x1:x2] = (
      (orig_roi * alpha + 255 * (255 - alpha)) >> 8
  ).astype(np.uint8)


def stamp_outline_segment(
    canvas_view, reveal_mask, ink_mask, orig_bgr, p1, p2, radius
):
  """Reveals crisp outline pixels along a contour segment strictly within ink_mask."""
  h, w = orig_bgr.shape[:2]
  pad = radius + 2
  x1 = max(0, min(p1[0], p2[0]) - pad)
  y1 = max(0, min(p1[1], p2[1]) - pad)
  x2 = min(w, max(p1[0], p2[0]) + pad + 1)
  y2 = min(h, max(p1[1], p2[1]) + pad + 1)

  if x2 <= x1 or y2 <= y1:
    return

  stroke_buf = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
  cv2.line(
      stroke_buf,
      (p1[0] - x1, p1[1] - y1),
      (p2[0] - x1, p2[1] - y1),
      255,
      thickness=radius * 2,
      lineType=cv2.LINE_AA,
  )

  gated = cv2.bitwise_and(stroke_buf, ink_mask[y1:y2, x1:x2])
  np.maximum(reveal_mask[y1:y2, x1:x2], gated, out=reveal_mask[y1:y2, x1:x2])
  update_canvas_from_mask_roi(canvas_view, reveal_mask, orig_bgr, x1, y1, x2, y2)


class FrameSink:
  """Pipes raw BGR frames directly into FFmpeg libx264 for razor-sharp edges."""

  def __init__(self, output_path, fps, width, height, total_frames):
    self.proc = None
    self.writer = None
    self.written = 0
    self.total_frames = total_frames
    ffmpeg_bin = shutil.which("ffmpeg")

    if ffmpeg_bin is not None:
      cmd = [
          ffmpeg_bin,
          "-y",
          "-f",
          "rawvideo",
          "-vcodec",
          "rawvideo",
          "-s",
          f"{width}x{height}",
          "-pix_fmt",
          "bgr24",
          "-r",
          str(fps),
          "-i",
          "-",
          "-vf",
          "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
          "-c:v",
          "libx264",
          "-preset",
          "veryfast",
          "-crf",
          "18",
          "-pix_fmt",
          "yuv420p",
          output_path,
      ]
      try:
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
      except Exception:
        self.proc = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not self.writer.isOpened():
      raise RuntimeError(f"Failed to open video writer for {output_path}")

  def write(self, frame_bgr):
    if self.proc is not None and self.proc.stdin is not None:
      self.proc.stdin.write(frame_bgr.tobytes())
    elif self.writer is not None:
      self.writer.write(frame_bgr)
    self.written += 1
    if self.written % 25 == 0 or self.written == self.total_frames:
      sys.stdout.write(
          f"\r[*] Rendering frames: {self.written}/{self.total_frames}"
      )
      sys.stdout.flush()

  def close(self):
    print()
    if self.proc is not None:
      if self.proc.stdin is not None:
        self.proc.stdin.close()
      self.proc.wait()
    if self.writer is not None:
      self.writer.release()


def emit_hold_frames(
    canvas_view,
    hand_engine,
    frame_sink,
    hold_x,
    hold_y,
    num_frames,
    current_frame_count,
    max_frame_limit,
):
  """Emits natural wrist-dwell frames at landings and stroke endpoints."""
  for _ in range(num_frames):
    if current_frame_count >= max_frame_limit:
      break
    restore_info = hand_engine.render_hand_in_place(
        canvas_view, hold_x, hold_y, lift_offset_y=0.0
    )
    frame_sink.write(canvas_view)
    if restore_info is not None:
      cy1, cy2, cx1, cx2, patch = restore_info
      canvas_view[cy1:cy2, cx1:cx2] = patch
    current_frame_count += 1
  return current_frame_count


def render_outline_pass(
    waypoints,
    canvas_view,
    reveal_mask,
    ink_mask,
    orig_bgr,
    hand_engine,
    frame_sink,
    pass_target_frames,
    current_frame_count,
):
  """Renders arc-length-paced contour outline tracing so nib speed stays

  calm and uniform across long curves, starting from an empty board at 0.0s.
  """
  if len(waypoints) == 0 or pass_target_frames <= current_frame_count:
    return current_frame_count

  # Guarantee Frame 0 shows the hand landing on a clean board before ink appears
  if current_frame_count == 0:
    first_x, first_y = waypoints[0][0], waypoints[0][1]
    current_frame_count = emit_hold_frames(
        canvas_view,
        hand_engine,
        frame_sink,
        first_x,
        first_y,
        1,
        current_frame_count,
        pass_target_frames,
    )

  start_frame = current_frame_count
  frames_budget = pass_target_frames - start_frame
  if frames_budget <= 0:
    return current_frame_count

  total_arc_w = sum(wp[5] for wp in waypoints)
  accum_arc_w = 0.0

  prev_pos = None
  orig_ht, orig_wd = orig_bgr.shape[:2]
  lift_dist_thresh = max(65.0, orig_wd * 0.055)
  max_lift_px = max(18.0, orig_ht * 0.028)
  pending_segments = []

  for wx, wy, radius, is_new_stroke, is_end, seg_w in waypoints:
    curr_pos = (wx, wy)
    if prev_pos is None:
      prev_pos = curr_pos

    jump_dist = math.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
    is_jump = is_new_stroke or (jump_dist > lift_dist_thresh)

    if not is_jump:
      pending_segments.append((prev_pos, curr_pos, radius))
    else:
      pending_segments.append((curr_pos, curr_pos, radius))

    accum_arc_w += seg_w
    expected_frame = start_frame + int(
        round((accum_arc_w / max(1.0, total_arc_w)) * frames_budget)
    )
    frames_to_emit = min(expected_frame, pass_target_frames) - current_frame_count

    if frames_to_emit > 0:
      if len(pending_segments) > 1:
        for seg_p1, seg_p2, seg_r in pending_segments[:-1]:
          stamp_outline_segment(
              canvas_view,
              reveal_mask,
              ink_mask,
              orig_bgr,
              seg_p1,
              seg_p2,
              seg_r,
          )
      last_p1, last_p2, last_r = pending_segments[-1]
      pending_segments.clear()

      sub_prev = prev_pos
      for f_step in range(frames_to_emit):
        t = (f_step + 1) / float(frames_to_emit)
        interp_x = int(round(prev_pos[0] + t * (curr_pos[0] - prev_pos[0])))
        interp_y = int(round(prev_pos[1] + t * (curr_pos[1] - prev_pos[1])))
        sub_curr = (interp_x, interp_y)

        if is_jump and jump_dist > lift_dist_thresh:
          lift_y = math.sin(t * math.pi) * max_lift_px
          if f_step == frames_to_emit - 1:
            stamp_outline_segment(
                canvas_view,
                reveal_mask,
                ink_mask,
                orig_bgr,
                last_p1,
                last_p2,
                last_r,
            )
        else:
          lift_y = 0.0
          stamp_outline_segment(
              canvas_view,
              reveal_mask,
              ink_mask,
              orig_bgr,
              sub_prev,
              sub_curr,
              last_r,
          )

        restore_info = hand_engine.render_hand_in_place(
            canvas_view, interp_x, interp_y, lift_offset_y=lift_y
        )
        frame_sink.write(canvas_view)
        if restore_info is not None:
          cy1, cy2, cx1, cx2, patch = restore_info
          canvas_view[cy1:cy2, cx1:cx2] = patch

        sub_prev = sub_curr
        current_frame_count += 1

      if is_end and (pass_target_frames - current_frame_count) > 2:
        current_frame_count = emit_hold_frames(
            canvas_view,
            hand_engine,
            frame_sink,
            curr_pos[0],
            curr_pos[1],
            1,
            current_frame_count,
            pass_target_frames,
        )

    prev_pos = curr_pos

  for seg_p1, seg_p2, seg_r in pending_segments:
    stamp_outline_segment(
        canvas_view, reveal_mask, ink_mask, orig_bgr, seg_p1, seg_p2, seg_r
    )

  return current_frame_count


def render_sweep_fill_pass(
    stripes,
    canvas_view,
    reveal_mask,
    orig_bgr,
    hand_engine,
    frame_sink,
    pass_target_frames,
    current_frame_count,
):
  """Shades chained color patches smoothly from top-left to bottom-right."""
  if len(stripes) == 0 or pass_target_frames <= current_frame_count:
    return current_frame_count

  start_frame = current_frame_count
  frames_budget = pass_target_frames - start_frame
  total_weights = sum(math.sqrt(s["pixel_count"]) + 1.5 for s in stripes)
  accum_w = 0.0

  prev_hand_pos = None
  orig_ht, orig_wd = orig_bgr.shape[:2]
  lift_dist_thresh = max(65.0, orig_wd * 0.055)
  max_lift_px = max(18.0, orig_ht * 0.028)

  for stripe in stripes:
    s_weight = math.sqrt(stripe["pixel_count"]) + 1.5
    accum_w += s_weight
    stripe_end_frame = start_frame + int(
        round((accum_w / max(1.0, total_weights)) * frames_budget)
    )
    stripe_end_frame = min(pass_target_frames, stripe_end_frame)

    s_pt = stripe["start_pt"]
    e_pt = stripe["end_pt"]
    rx, ry = stripe["rx"], stripe["ry"]
    bx, by, prog = stripe["bx"], stripe["by"], stripe["prog"]

    if prev_hand_pos is None:
      prev_hand_pos = s_pt

    jump_dist = math.hypot(
        s_pt[0] - prev_hand_pos[0], s_pt[1] - prev_hand_pos[1]
    )
    if stripe["is_new_region"] and jump_dist > lift_dist_thresh:
      jump_frames = min(
          4, max(2, int(round(jump_dist / max(45.0, orig_wd * 0.06))))
      )
      if current_frame_count + jump_frames < stripe_end_frame:
        for jf in range(jump_frames):
          jt = (jf + 1) / float(jump_frames)
          jx = int(round(prev_hand_pos[0] + jt * (s_pt[0] - prev_hand_pos[0])))
          jy = int(round(prev_hand_pos[1] + jt * (s_pt[1] - prev_hand_pos[1])))
          lift_y = math.sin(jt * math.pi) * max_lift_px

          restore_info = hand_engine.render_hand_in_place(
              canvas_view, jx, jy, lift_offset_y=lift_y
          )
          frame_sink.write(canvas_view)
          if restore_info is not None:
            cy1, cy2, cx1, cx2, patch = restore_info
            canvas_view[cy1:cy2, cx1:cx2] = patch
          current_frame_count += 1

        current_frame_count = emit_hold_frames(
            canvas_view,
            hand_engine,
            frame_sink,
            s_pt[0],
            s_pt[1],
            1,
            current_frame_count,
            stripe_end_frame,
        )

    sweep_frames = stripe_end_frame - current_frame_count
    x1_roi = max(0, rx + int(np.min(bx)))
    y1_roi = max(0, ry + int(np.min(by)))
    x2_roi = min(orig_wd, rx + int(np.max(bx)) + 1)
    y2_roi = min(orig_ht, ry + int(np.max(by)) + 1)

    if sweep_frames > 0:
      for f_step in range(sweep_frames):
        t = (f_step + 1) / float(sweep_frames)
        active = prog <= (t + 1e-4)
        if np.any(active):
          reveal_mask[ry + by[active], rx + bx[active]] = 255
          update_canvas_from_mask_roi(
              canvas_view, reveal_mask, orig_bgr, x1_roi, y1_roi, x2_roi, y2_roi
          )

        hx = int(round(s_pt[0] + t * (e_pt[0] - s_pt[0])))
        hy = int(round(s_pt[1] + t * (e_pt[1] - s_pt[1])))
        restore_info = hand_engine.render_hand_in_place(
            canvas_view, hx, hy, lift_offset_y=0.0
        )
        frame_sink.write(canvas_view)
        if restore_info is not None:
          cy1, cy2, cx1, cx2, patch = restore_info
          canvas_view[cy1:cy2, cx1:cx2] = patch

        current_frame_count += 1
    else:
      reveal_mask[ry + by, rx + bx] = 255
      update_canvas_from_mask_roi(
          canvas_view, reveal_mask, orig_bgr, x1_roi, y1_roi, x2_roi, y2_roi
      )

    prev_hand_pos = e_pt

  return current_frame_count


def generate_native_whiteboard_animation(
    img_path,
    hand_path=None,
    output_path=None,
    json_mask_path=None,
    total_duration_sec=12.0,
    fps=25,
    final_hold_sec=2.0,
    color_time_ratio=0.40,
    **kwargs,
):
  """Generates a Doodly/VideoScribe-grade whiteboard animation aligned frame-for-frame

  with the TTS audio duration passed from modules/script_parser.py.
  """
  start_time = time.time()

  if output_path is None:
    base_no_ext = os.path.splitext(img_path)[0]
    output_path = f"{base_no_ext}_whiteboard.mp4"

  resolved_hand_path = resolve_hand_asset_path(hand_path)

  # Support any duration keyword alias passed by script_parser.py
  for alias in (
      "duration",
      "target_duration",
      "target_duration_sec",
      "audio_duration",
  ):
    if alias in kwargs and kwargs[alias] is not None:
      total_duration_sec = float(kwargs[alias])
      break

  (
      orig_bgr,
      native_fg,
      native_ink,
      orig_ht,
      orig_wd,
      work_bgr,
      work_fg,
      work_ink,
      work_fill,
      work_ht,
      work_wd,
      work_scale,
  ) = decompose_artwork_layers(img_path)

  inv_scale = 1.0 / work_scale
  print(f"[*] Native Image Dimensions: {orig_wd}x{orig_ht}")
  print(f"[*] Target Video Duration: {total_duration_sec:.2f}s @ {fps} FPS")
  print(
      f"[*] Analyzing layers on {work_wd}x{work_ht} grid (rendering at"
      f" {orig_wd}x{orig_ht})..."
  )

  hand_engine = FastLivingHandEngine(resolved_hand_path, orig_wd, orig_ht)

  # Keep full 2.0s hold on normal clips, while protecting active drawing time
  # on short (<3.5s) TTS voiceover segments.
  effective_hold_sec = (
      final_hold_sec
      if total_duration_sec >= (final_hold_sec + 1.5)
      else min(final_hold_sec, max(0.3, total_duration_sec * 0.22))
  )

  total_target_frames = max(1, int(round(total_duration_sec * fps)))
  final_hold_frames = int(round(effective_hold_sec * fps))
  drawing_target_frames = max(1, total_target_frames - final_hold_frames)

  out_dir = os.path.dirname(output_path)
  if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

  frame_sink = FrameSink(
      output_path, fps, orig_wd, orig_ht, total_target_frames
  )
  canvas_view = np.full((orig_ht, orig_wd, 3), 255, dtype=np.uint8)
  reveal_mask = np.zeros((orig_ht, orig_wd), dtype=np.uint8)

  if json_mask_path is not None and os.path.exists(json_mask_path):
    print(f"[*] Loading custom object sequence from: {json_mask_path}")
    with open(json_mask_path, "r") as f:
      mask_data = json.load(f)
    zone_masks = []
    bg_zone = np.full((work_ht, work_wd), 255, dtype=np.uint8)
    for shape in mask_data.get("shapes", []):
      poly_mask = np.zeros((work_ht, work_wd), dtype=np.uint8)
      pts = np.round(
          np.array(shape.get("points", []), dtype=np.float32) * work_scale
      ).astype(np.int32)
      if len(pts) > 2:
        cv2.fillPoly(poly_mask, [pts], 255)
        zone_masks.append(poly_mask)
        bg_zone[poly_mask == 255] = 0
    zone_masks.append(bg_zone)
  else:
    zone_masks = segment_scene_objects(work_fg, work_wd, work_ht)

  object_plans = []
  raw_weights = []

  for z_mask in zone_masks:
    z_ink = cv2.bitwise_and(work_ink, z_mask)
    z_fill = cv2.bitwise_and(work_fill, z_mask)

    outline_pts = build_contour_outline_strokes(
        z_ink, work_wd, work_ht, inv_scale
    )
    fill_stripes = build_smooth_color_patches(
        z_fill,
        work_bgr,
        work_wd,
        work_ht,
        native_fg,
        orig_wd,
        orig_ht,
        inv_scale,
    )

    if len(outline_pts) == 0 and len(fill_stripes) == 0:
      continue

    ink_px = float(np.count_nonzero(z_ink))
    fill_px = float(np.count_nonzero(z_fill))
    w = math.sqrt(ink_px + fill_px * 0.65) + 4.0
    object_plans.append((outline_pts, fill_stripes))
    raw_weights.append(w)

  n_objs = len(object_plans)
  if n_objs > 0:
    total_w = sum(raw_weights)
    min_share = min(0.20, 0.85 / float(n_objs))
    norm_shares = [max(min_share, w / total_w) for w in raw_weights]
    share_sum = sum(norm_shares)
    norm_shares = [s / share_sum for s in norm_shares]
  else:
    norm_shares = []

  print(
      f"[*] Planned {n_objs} left-to-right object groups in"
      f" {time.time() - start_time:.2f}s"
  )

  current_frames = 0
  accum_share = 0.0

  for obj_idx, ((outline_pts, fill_stripes), share) in enumerate(
      zip(object_plans, norm_shares)
  ):
    accum_share += share
    obj_end_frame = int(round(accum_share * drawing_target_frames))
    min_needed = 6 if (len(outline_pts) > 0 and len(fill_stripes) > 0) else 2
    obj_end_frame = min(
        drawing_target_frames, max(current_frames + min_needed, obj_end_frame)
    )
    obj_frames = obj_end_frame - current_frames

    dwell_after_obj = 3 if (obj_idx < n_objs - 1 and obj_frames > 14) else 0
    active_obj_end = obj_end_frame - dwell_after_obj
    active_frames = max(1, active_obj_end - current_frames)

    if len(outline_pts) > 0 and len(fill_stripes) > 0:
      outline_frames = max(
          2, int(round(active_frames * (1.0 - color_time_ratio)))
      )
      outline_end = min(active_obj_end - 2, current_frames + outline_frames)
    elif len(outline_pts) > 0:
      outline_end = active_obj_end
    else:
      outline_end = current_frames

    if len(outline_pts) > 0 and outline_end > current_frames:
      current_frames = render_outline_pass(
          outline_pts,
          canvas_view,
          reveal_mask,
          native_ink,
          orig_bgr,
          hand_engine,
          frame_sink,
          outline_end,
          current_frames,
      )

    if len(fill_stripes) > 0 and active_obj_end > current_frames:
      current_frames = render_sweep_fill_pass(
          fill_stripes,
          canvas_view,
          reveal_mask,
          orig_bgr,
          hand_engine,
          frame_sink,
          active_obj_end,
          current_frames,
      )

    if dwell_after_obj > 0 and hand_engine.curr_x is not None:
      current_frames = emit_hold_frames(
          canvas_view,
          hand_engine,
          frame_sink,
          int(round(hand_engine.curr_x)),
          int(round(hand_engine.curr_y)),
          dwell_after_obj,
          current_frame_count=current_frames,
          max_frame_limit=obj_end_frame,
      )

  while current_frames < drawing_target_frames:
    frame_sink.write(orig_bgr)
    current_frames += 1

  for _ in range(final_hold_frames):
    frame_sink.write(orig_bgr)
    current_frames += 1

  frame_sink.close()
  total_time = time.time() - start_time
  print(f"[✓] Render completed: {output_path}")
  print(f"[✓] Output Resolution: {orig_wd}x{orig_ht} (100% Native)")
  print(
      f"[✓] Frames Written: {current_frames} ({current_frames / fps:.2f}s) in"
      f" {total_time:.2f}s"
  )
  return output_path
