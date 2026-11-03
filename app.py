import asyncio
import json
import os
import shutil
import glob
import streamlit as st

# Engine and Subsystem Imports
from bot.bot_runner import start_bot, stop_bot, is_bot_active
from modules.cartesia_tts import VOICE_LIST, DEFAULT_VOICE_NAME
from modules.script_parser import process_video_pipeline, get_available_backgrounds, BG_DIR, OUTPUT_DIR
from animations.animation_router import ANIMATION_REGISTRY

# Page configuration
st.set_page_config(
    page_title="Video Engine Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize storage paths
os.makedirs(BG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------------
# BACKGROUND BOT SUPERVISOR
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_telegram_bot_daemon():
    """Starts the Pyrofork daemon thread once per server lifecycle."""
    success = start_bot()
    return success

# Trigger daemon startup
bot_started = initialize_telegram_bot_daemon()


# -------------------------------------------------------------
# SIDEBAR CONTROLS & ENVIRONMENT SETTINGS
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Engine Settings")

    # Bot Status Panel
    st.subheader("Telegram Bot Supervisor")
    active = is_bot_active()
    if active:
        st.success("🟢 Bot Status: Online (Daemon Active)")
    else:
        st.error("🔴 Bot Status: Offline")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Start Bot", use_container_width=True, disabled=active):
            start_bot()
            st.rerun()
    with col_b2:
        if st.button("Stop Bot", use_container_width=True, disabled=not active):
            stop_bot()
            st.rerun()

    st.divider()

    # Voice Configuration
    st.subheader("🎙️ TTS Voice")
    voice_names = [v["name"] for v in VOICE_LIST]
    default_idx = voice_names.index(DEFAULT_VOICE_NAME) if DEFAULT_VOICE_NAME in voice_names else 0
    selected_voice = st.selectbox(
        "Select Malayalam Voice Profile",
        options=voice_names,
        index=default_idx
    )

    st.divider()

    # Credentials & Session Cookie Overrides
    st.subheader("🔑 Authentication")
    secrets_cookie = ""
    try:
        secrets_cookie = st.secrets.get("BING_COOKIE_U", "")
    except Exception:
        pass

    bing_cookie = st.text_input(
        "Bing `_U` Cookie",
        value=secrets_cookie,
        type="password",
        help="Session cookie for Microsoft Designer / Bing Image Creator MAI1."
    )

    st.divider()

    # Background Video Assets Management
    st.subheader("🎞️ Canvas Backgrounds")
    available_bgs = get_available_backgrounds()
    st.write(f"Loaded: `{len(available_bgs)}` video loops")

    uploaded_bgs = st.file_uploader(
        "Upload Background Videos (.mp4)",
        type=["mp4"],
        accept_multiple_files=True
    )
    if uploaded_bgs:
        for uploaded_file in uploaded_bgs:
            save_path = os.path.join(BG_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_bgs)} file(s).")
        st.rerun()


# -------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# -------------------------------------------------------------
st.title("🎬 Multi-Scene Automated Video Engine")
st.caption("Synchronized Cartesia TTS • Pixabay Footage • Bing AI Graphics • Procedural OpenCV Animations")

tabs = st.tabs(["🚀 Run Pipeline", "📋 Script Schema & Sample", "🎨 Registered Animations & Backgrounds"])

# -------------------------------------------------------------
# TAB 1: RUN PIPELINE
# -------------------------------------------------------------
with tabs[0]:
    input_method = st.radio("Input Source", ["Paste JSON Script", "Upload .json File"], horizontal=True)
    raw_script_content = ""

    if input_method == "Paste JSON Script":
        sample_json_default = json.dumps([
            {
                "sentence": "നമസ്കാരം, ഇത് ആധുനിക വീഡിയോ നിർമ്മാണ സംവിധാനത്തിന്റെ ആദ്യ രംഗമാണ്.",
                "query": "cinematic technology abstract landscape"
            },
            {
                "sentence": "രണ്ടാമത്തെ രംഗത്തിൽ പശ്ചാത്തല വീഡിയോയുടെ മുകളിൽ ഇമേജ് കൃത്യമായ അനുപാതത്തിൽ വരുന്നു.",
                "prompt": "futuristic neon glass citadel floating in sunset clouds",
                "source": "bing"
            },
            {
                "sentence": "മൂന്നാമത്തെ രംഗത്തിൽ ചലനാത്മകമായ അനിമേഷൻ സംവിധാനം പ്രവർത്തിക്കുന്നു.",
                "query": "cyberpunk digital matrix hologram",
                "source": "pixabay"
            }
        ], ensure_ascii=False, indent=2)

        raw_script_content = st.text_area(
            "Script JSON Payload",
            value=sample_json_default,
            height=260
        )
    else:
        uploaded_json = st.file_uploader("Upload JSON Script", type=["json"])
        if uploaded_json:
            raw_script_content = uploaded_json.read().decode("utf-8")

    start_render = st.button("🚀 Render Master Video", type="primary", use_container_width=True)

    if start_render:
        if not raw_script_content.strip():
            st.error("Please supply a valid JSON script payload.")
        else:
            try:
                parsed_script = json.loads(raw_script_content)
                if not isinstance(parsed_script, list) or len(parsed_script) == 0:
                    st.error("JSON payload must be a non-empty array of scene objects.")
                    st.stop()
            except json.JSONDecodeError as err:
                st.error(f"Invalid JSON Syntax: {err}")
                st.stop()

            # Progress UI Elements
            prog_bar = st.progress(0.0)
            status_box = st.empty()

            def ui_progress_callback(idx: int, total: int, step_desc: str):
                fraction = float(idx / max(total, 1))
                prog_bar.progress(min(fraction, 1.0))
                status_box.info(f"**Step [{idx}/{total}]:** {step_desc}")

            try:
                with st.spinner("Processing video engine pipeline..."):
                    # Execute synchronous pipeline wrapper across asyncio event loop
                    output_file = asyncio.run(
                        process_video_pipeline(
                            script_data=parsed_script,
                            bing_cookie_u=bing_cookie,
                            voice_name=selected_voice,
                            progress_callback=ui_progress_callback
                        )
                    )

                prog_bar.progress(1.0)
                status_box.success("🎉 Video generation completed successfully!")

                # Video Preview and Download Section
                if os.path.exists(output_file):
                    col_v1, col_v2 = st.columns([2, 1])
                    with col_v1:
                        st.subheader("Master Video Output (1080p)")
                        st.video(output_file)
                    with col_v2:
                        st.subheader("File Details")
                        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                        st.write(f"• **Filename:** `{os.path.basename(output_file)}`")
                        st.write(f"• **Size:** `{file_size_mb:.2f} MB`")
                        st.write(f"• **Resolution:** `1920x1080` (30 FPS)")
                        st.write(f"• **Voice Profile:** `{selected_voice}`")

                        with open(output_file, "rb") as vf:
                            st.download_button(
                                label="⬇️ Download Video (MP4)",
                                data=vf,
                                file_name=os.path.basename(output_file),
                                mime="video/mp4",
                                type="primary",
                                use_container_width=True
                            )

            except Exception as e:
                status_box.error(f"❌ Execution Failure: {e}")


# -------------------------------------------------------------
# TAB 2: SCHEMA & SAMPLE
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Pipeline Scene Lifecycle Specification")
    st.markdown("""
    | Sentence Sequence | Visual Pipeline Logic | Transition Behavior |
    | :--- | :--- | :--- |
    | **Sentence 1** | Scrapes & downloads unique landscape video from Pixabay matching query. | Exact TTS length + 0.4s fade-out. |
    | **Sentence 2, 4, 6...** | Cycles background loop (`BG1`, `BG2`...) + Centered Bing/Pixabay image with equal margins. | Exact TTS length + 0.4s fade-out. |
    | **Sentence 3, 5, 7...** | Procedural OpenCV animation dynamically rendered to TTS duration. | Exact TTS length + 0.4s fade-out. |
    """)

    st.subheader("Standard JSON Schema")
    st.code("""[
  {
    "sentence": "വാചകം ഇവിടെ ചേർക്കുക...",
    "query": "pixabay search query",
    "prompt": "bing image creator prompt",
    "source": "bing"  // or "pixabay"
  }
]""", language="json")


# -------------------------------------------------------------
# TAB 3: ASSET & ANIMATION REGISTRY
# -------------------------------------------------------------
with tabs[2]:
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.subheader("Loaded Kinetic Animation Engines")
        for idx, anim in enumerate(ANIMATION_REGISTRY, start=1):
            st.markdown(f"**{idx}. {anim['name']}** (`{anim['id']}`)")

    with col_a2:
        st.subheader("Current Background Library")
        bgs = get_available_backgrounds()
        if bgs:
            for b in bgs:
                st.write(f"• `{os.path.basename(b)}`")
        else:
            st.warning("No `.mp4` background files found in `assets/backgrounds/`.")
