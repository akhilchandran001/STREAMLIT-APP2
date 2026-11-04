import os
import sys

# Import the 6 procedural kinetic animation generators in exact priority order
from animations.velocity_speed_ramp import generate_velocity_speed_ramp
from animations.bullet_time_orbit_freeze import generate_bullet_time_orbit_freeze
from animations.duct_tape_vinyl_sticker_slam import generate_duct_tape_vinyl_sticker_slam
from animations.isometric_tilt_shift import generate_isometric_tilt_shift_miniature
from animations.dolly_zoom_vertigo import generate_dolly_zoom_vertigo
from animations.inertial_drift_brake import generate_inertial_drift_snap_brake

# Priority registry mapping index to runner functions
ANIMATION_REGISTRY = [
    {
        "id": "velocity_speed_ramp",
        "name": "Velocity Speed Ramp",
        "func": generate_velocity_speed_ramp,
    },
    {
        "id": "bullet_time_orbit_freeze",
        "name": "Bullet Time Orbit Freeze",
        "func": generate_bullet_time_orbit_freeze,
    },
    {
        "id": "duct_tape_vinyl_sticker_slam",
        "name": "Duct Tape & Vinyl Sticker Slam",
        "func": generate_duct_tape_vinyl_sticker_slam,
    },
    {
        "id": "isometric_tilt_shift",
        "name": "Isometric Tilt Shift Miniature",
        "func": generate_isometric_tilt_shift_miniature,
    },
    {
        "id": "dolly_zoom_vertigo",
        "name": "Hitchcock Dolly Zoom Vertigo Shift",
        "func": generate_dolly_zoom_vertigo,
    },
    {
        "id": "inertial_drift_brake",
        "name": "Inertial Drift & Snap Brake",
        "func": generate_inertial_drift_snap_brake,
    },
]


def render_animation_by_index(
    image_path: str = None,
    output_path: str = None,
    duration_sec: float = None,
    cycle_index: int = 0,
    fps: int = 30,
    *,
    index: int = None,
    duration: float = None,
    **kwargs
) -> str:
    """
    Rounds through the 6 animations sequentially based on the cycle counter,
    ensuring each animated sentence receives a fresh, non-repeating procedural style
    calibrated to the exact duration of the TTS audio.
    Accepts both (cycle_index, duration_sec) and (index, duration) to prevent keyword argument errors.
    """
    if index is not None:
        cycle_index = index
    if duration is not None:
        duration_sec = duration

    if duration_sec is None:
        duration_sec = 6.0

    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at '{image_path}'")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Determine which animation to run using modulo on the registered count
    chosen_idx = cycle_index % len(ANIMATION_REGISTRY)
    selected_spec = ANIMATION_REGISTRY[chosen_idx]

    print(f"\n[Animation Router] Cycle #{cycle_index} -> Executing '{selected_spec['name']}' ({duration_sec:.2f}s)...")

    try:
        rendered_file = selected_spec["func"](
            input_path=image_path,
            output_path=output_path,
            duration_sec=duration_sec,
            fps=fps
        )
        return rendered_file
    except Exception as e:
        print(f"[Animation Router] Warning: '{selected_spec['name']}' failed with error: {e}")
        print("[Animation Router] Engaging Velocity Speed Ramp as automatic failsafe...")
        return generate_velocity_speed_ramp(
            input_path=image_path,
            output_path=output_path,
            duration_sec=duration_sec,
            fps=fps
        )


def render_animation_by_name(
    image_path: str,
    output_path: str,
    duration_sec: float,
    animation_id: str,
    fps: int = 30
) -> str:
    """
    Allows executing a specific animation requested by ID in the scene JSON script.
    """
    normalized_id = animation_id.strip().lower()

    for idx, spec in enumerate(ANIMATION_REGISTRY):
        if spec["id"] == normalized_id or spec["name"].lower() == normalized_id:
            return render_animation_by_index(
                image_path=image_path,
                output_path=output_path,
                duration_sec=duration_sec,
                cycle_index=idx,
                fps=fps
            )

    print(f"[Animation Router] Unknown animation ID '{animation_id}', defaulting to cycle index 0.")
    return render_animation_by_index(
        image_path=image_path,
        output_path=output_path,
        duration_sec=duration_sec,
        cycle_index=0,
        fps=fps
    )


if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else "1.jpg"
    test_out = sys.argv[2] if len(sys.argv) > 2 else "router_test.mp4"
    test_dur = float(sys.argv[3]) if len(sys.argv) > 3 else 6.5
    test_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    render_animation_by_index(
        image_path=test_img,
        output_path=test_out,
        duration_sec=test_dur,
        cycle_index=test_idx
    )
