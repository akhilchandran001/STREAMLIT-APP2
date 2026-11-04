import os
import matplotlib
matplotlib.use("Agg")  # Headless backend for Streamlit / server containers
import matplotlib.pyplot as plt
import numpy as np
import cv2

CANVAS_W = 1920
CANVAS_H = 1080


def render_cinematic_equation_card(
    latex_equation: str,
    output_path: str,
    formula_name: str = "",
    width: int = 1760,
    height: int = 870
) -> str:
    """
    Renders an exact LaTeX formula onto a dark, frosted-glass cinematic card.
    Uses Matplotlib's native mathtext engine (Computer Modern font).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Prepare LaTeX string
    clean_expr = latex_equation.strip()
    if not clean_expr.startswith("$"):
        clean_expr = f"${clean_expr}$"

    # 2. Render LaTeX equation to memory using Matplotlib
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.patch.set_alpha(0.0)

    # Main formula in high-contrast brilliant white
    ax.text(
        0.5, 0.48, clean_expr,
        fontsize=46,
        color="#FFFFFF",
        ha="center", va="center",
        usetex=False  # Uses built-in Mathtext parser; works on any machine
    )

    # Optional formula label (e.g., "Navier–Stokes Momentum Equation")
    if formula_name:
        ax.text(
            0.5, 0.18, formula_name.upper(),
            fontsize=20,
            color="#DC2644",  # Crimson accent
            fontweight="bold",
            ha="center", va="center",
            letterspacing=2
        )

    fig.canvas.draw()
    rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    formula_bgr = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGBA2BGR)
    formula_alpha = rgba_buffer[:, :, 3] / 255.0

    # 3. Create a cinematic dark obsidian blackboard background
    base_card = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Subtle dark blue-gray radial gradient
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width * 0.5, height * 0.5
    r = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / np.sqrt(cx ** 2 + cy ** 2)
    gradient = np.clip(1.0 - 0.55 * r, 0.0, 1.0)[:, :, np.newaxis]
    
    dark_tone = np.array([28, 22, 18], dtype=np.float32)  # Deep obsidian
    ambient_tone = np.array([12, 10, 8], dtype=np.float32)
    card_surface = (ambient_tone + (dark_tone - ambient_tone) * gradient).astype(np.uint8)

    # Subtle scientific blueprint grid lines
    card_surface[::60, :, :] = np.clip(card_surface[::60, :, :].astype(np.int16) + 12, 0, 255).astype(np.uint8)
    card_surface[:, ::60, :] = np.clip(card_surface[:, ::60, :].astype(np.int16) + 12, 0, 255).astype(np.uint8)

    # 4. Composite the crisp formula on top with subtle bloom glow
    alpha_3ch = formula_alpha[:, :, np.newaxis]
    final_card = (card_surface * (1.0 - alpha_3ch) + formula_bgr * alpha_3ch).astype(np.uint8)

    # Subtle luminous glow around symbols
    glow_mask = (formula_alpha * 255).astype(np.uint8)
    glow_blurred = cv2.GaussianBlur(glow_mask, (31, 31), 0)
    glow_layer = (glow_blurred[:, :, np.newaxis] / 255.0) * np.array([120, 40, 20], dtype=np.float32)  # Cyan/blue aura
    final_card = np.clip(final_card.astype(np.float32) + glow_layer, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, final_card)
    print(f"[EquationRenderer] Exact math card generated at: {output_path}")
    return output_path
