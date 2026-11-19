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
from modules.text_parser import parse_input_script
from modules.newspaper_anim import (
    NEWSPAPER_ASSETS_DIR,
    LOGOS_DIR,
    FONTS_DIR,
    resolve_table_path,
    resolve_logo_path,
)
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
os.makedirs(NEWSPAPER_ASSETS_DIR, exist_ok=True)
os.makedirs(LOGOS_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)

if "_saved_uploads" not in st.session_state:
    st.session_state["_saved_uploads"] = set()


def _save_uploaded_once(uploaded_file, dest_path: str) -> bool:
    """Saves an uploaded Streamlit file once per (name, size) to prevent rerun loops."""
    sig = (dest_path, uploaded_file.name, uploaded_file.size)
    if sig in st.session_state["_saved_uploads"]:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["_saved_uploads"].add(sig)
    return True


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

    # Newspaper Engine Settings & Assets
    st.subheader("📰 Newspaper Engine & SFX")
    newspaper_edition = st.selectbox(
        "Newspaper Edition",
        ["Alternating (Manorama ↔ Mathrubhumi)", "Malayala Manorama", "Mathrubhumi"],
        index=0
    )
    edition_key = (
        "alternating" if "Alternating" in newspaper_edition
        else ("mathrubhumi" if "Mathrubhumi" in newspaper_edition else "manorama")
    )

    news_audio_mode = st.radio(
        "Newspaper Audio Mode",
        ["TTS Voiceover + Paper/Marker SFX", "Newspaper SFX Only (No TTS)"],
        index=0
    )
    news_use_tts = "No TTS" not in news_audio_mode
    news_enable_sfx = st.checkbox("🔊 Enable Drop & Highlighter SFX", value=True)
    news_sfx_vol = st.slider(
        "SFX Mix Volume",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        disabled=not news_enable_sfx,
        help="Volume of paper drop and yellow marker sound effects relative to TTS."
    )
    news_sfx_dur = st.number_input(
        "Duration (SFX-Only Mode, sec)",
        min_value=3.0,
        max_value=30.0,
        value=8.0,
        step=0.5,
        disabled=news_use_tts
    )

    news_date_line = st.text_input("Date Line (Optional)", value="2026 നവംബർ 19, വ്യാഴം")
    news_edition_info = st.text_input("Edition Info (Optional)", value="തിരുവനന്തപുരം • പേജ് 1 • വില ₹8.00")

    has_table = bool(resolve_table_path())
    has_manorama = bool(resolve_logo_path("manorama"))
    has_mathrubhumi = bool(resolve_logo_path("mathrubhumi"))
    st.caption(
        f"Table BG: {'✅ Loaded' if has_table else '⚠️ Wood Fallback'} | "
        f"Manorama: {'✅' if has_manorama else '⚠️ Text'} | "
        f"Mathrubhumi: {'✅' if has_mathrubhumi else '⚠️ Text'}"
    )

    up_table = st.file_uploader(
        "Upload Wooden Table (`table.png`)",
        type=["png", "jpg", "jpeg"],
        key="up_news_table"
    )
    if up_table:
        ext = os.path.splitext(up_table.name)[1].lower() or ".png"
        if _save_uploaded_once(up_table, os.path.join(NEWSPAPER_ASSETS_DIR, f"table{ext}")):
            st.success(f"Saved table background (`table{ext}`).")
            st.rerun()

    up_logos = st.file_uploader(
        "Upload Newspaper Logos (`Malayala Manorama.png`, `Mathrubhumi.png`)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="up_news_logos"
    )
    if up_logos:
        saved_any = False
        for lf in up_logos:
            if _save_uploaded_once(lf, os.path.join(LOGOS_DIR, lf.name)):
                saved_any = True
        if saved_any:
            st.success(f"Saved {len(up_logos)} logo(s).")
            st.rerun()

    up_fonts = st.file_uploader(
        "Upload Malayalam Fonts (`.ttf` / `.otf`)",
        type=["ttf", "otf"],
        accept_multiple_files=True,
        key="up_news_fonts"
    )
    if up_fonts:
        saved_any = False
        for ff in up_fonts:
            if _save_uploaded_once(ff, os.path.join(FONTS_DIR, ff.name)):
                saved_any = True
        if saved_any:
            st.success(f"Saved {len(up_fonts)} font(s).")
            st.rerun()

    st.divider()

    # Background Video Assets Management
    st.subheader("🎞️ Canvas Backgrounds (Default Mode)")
    available_bgs = get_available_backgrounds()
    st.write(f"Loaded: `{len(available_bgs)}` video loops")

    uploaded_bgs = st.file_uploader(
        "Upload Background Videos (.mp4)",
        type=["mp4"],
        accept_multiple_files=True
    )
    if uploaded_bgs:
        saved_any = False
        for uploaded_file in uploaded_bgs:
            save_path = os.path.join(BG_DIR, uploaded_file.name)
            if _save_uploaded_once(uploaded_file, save_path):
                saved_any = True
        if saved_any:
            st.success(f"Saved {len(uploaded_bgs)} file(s).")
            st.rerun()


# -------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# -------------------------------------------------------------
st.title("🎬 Multi-Scene Automated Video Engine")
st.caption("Synchronized Cartesia TTS • Newspaper Drop & Highlight • Whiteboard Living Hand • Bing AI Graphics • Kinetic Animations")

tabs = st.tabs(["🚀 Run Pipeline", "📋 Script Schema & Sample", "🎨 Registered Engines"])

# -------------------------------------------------------------
# TAB 1: RUN PIPELINE
# -------------------------------------------------------------
with tabs[0]:
    col_mode, col_src = st.columns([1, 1])

    with col_mode:
        selected_mode_label = st.radio(
            "🎬 Pipeline Rendering Mode",
            [
                "🎬 Default (3-Beat Video/Stamp/Animation)",
                "✍️ Whiteboard (Living Hand Drawing & Math)",
                "📰 Newspaper (Drop & Highlight Edition)"
            ],
            index=0
        )
        if "Whiteboard" in selected_mode_label:
            pipeline_mode = "whiteboard"
        elif "Newspaper" in selected_mode_label:
            pipeline_mode = "newspaper"
        else:
            pipeline_mode = "default"

    with col_src:
        input_method = st.radio(
            "Input Source",
            ["Paste Script (.json / .txt)", "Upload File (.json / .txt)"],
            horizontal=True
        )

    raw_script_content = ""

    if input_method == "Paste Script (.json / .txt)":
        if pipeline_mode == "whiteboard":
            sample_content_default = (
                "നമസ്കാരം, ഇത് വൈറ്റ്ബോർഡ് ആനിമേഷൻ സംവിധാനത്തിന്റെ ആദ്യ രംഗമാണ്. | futuristic robot science\n"
                "ഷ്രോഡിംഗർ സമവാക്യം ക്വാണ്ടം ഭൗതികത്തിലെ അടിസ്ഥാന സമവാക്യമാണ്. | i\\hbar \\frac{\\partial}{\\partial t}\\Psi = \\hat{H}\\Psi | Schrodinger Equation\n"
                "മൂന്നാമത്തെ രംഗത്തിൽ ശാസ്ത്ര ചിത്രം കൃത്യമായി വരയ്ക്കുന്നു. | molecular crystal structure"
            )
        elif pipeline_mode == "newspaper":
            sample_content_default = json.dumps([
                {
                    "sentence": "കേരളത്തിൽ വൻ വികസന കുതിപ്പ്. മെട്രോ രണ്ടാം ഘട്ടവും നവീകരിച്ച ദേശീയപാതയും നാടിന് സമർപ്പിച്ചു.",
                    "headline": "കേരളത്തിൽ വൻ വികസന കുതിപ്പ്:\nവൻപദ്ധതികൾ നാടിന് സമർപ്പിച്ചു",
                    "subheadline": "മെട്രോ രണ്ടാം ഘട്ടവും നവീകരിച്ച ദേശീയപാതയും ഉദ്ഘാടനം ചെയ്തു; കേരളത്തിന് പുതിയ ചരിത്ര നേട്ടം.",
                    "yellow_highlights": [
                        "കേരളത്തിന് ചരിത്ര നേട്ടം",
                        "₹12,000 കോടി രൂപയുടെ പദ്ധതികൾ",
                        "മെട്രോ രണ്ടാം ഘട്ടത്തിന് അനുമതി"
                    ],
                    "story_heading": "വിശദാംശങ്ങൾ",
                    "body_paragraphs": [
                        "കേരളത്തിൽ വൻ വികസനക്കുതിപ്പ്: വൻപദ്ധതികൾ നാടിന് സമർപ്പിച്ചു. സംസ്ഥാനത്തിന്റെ അടിസ്ഥാന സൗകര്യ വികസനത്തിൽ പുതിയ അധ്യായം കുറിച്ചുകൊണ്ട് മെട്രോ രണ്ടാം ഘട്ടവും ആറുവരി ദേശീയപാതയും ജനങ്ങൾക്കായി തുറന്നുകൊടുത്തു.",
                        "ദേശീയപാത വികസനം പൂർത്തിയാകുന്നതോടെ യാത്രാസമയം പകുതിയായി കുറയും. തുറമുഖ നഗരങ്ങളെയും ഐടി ഇടനാഴികളെയും ബന്ധിപ്പിക്കുന്ന പുതിയ പാതകൾ വിനോദസഞ്ചാര മേഖലയ്ക്കും വലിയ ഉണർവ് നൽകുമെന്ന് അധികൃതർ വ്യക്തമാക്കി."
                    ]
                }
            ], ensure_ascii=False, indent=2)
        else:
            sample_content_default = json.dumps([
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
            "Script Content (JSON Array or Plain Text Lines)",
            value=sample_content_default,
            height=260,
            key=f"script_editor_{pipeline_mode}"
        )
    else:
        uploaded_file = st.file_uploader("Upload Script File (.json or .txt)", type=["json", "txt"])
        if uploaded_file:
            raw_script_content = uploaded_file.read().decode("utf-8", errors="ignore")

    start_render = st.button("🚀 Render Master Video", type="primary", use_container_width=True)

    if start_render:
        if not raw_script_content.strip():
            st.error("Please supply a valid script payload.")
        else:
            try:
                parsed_script = parse_input_script(raw_script_content)
                if not isinstance(parsed_script, list) or len(parsed_script) == 0:
                    st.error("Failed to extract any scenes from the input.")
                    st.stop()
            except Exception as err:
                st.error(f"Invalid Script Syntax: {err}")
                st.stop()

            # Progress UI Elements
            prog_bar = st.progress(0.0)
            status_box = st.empty()

            def ui_progress_callback(step_desc: str, *args, **kwargs):
                status_box.info(f"⏳ {step_desc}")

            newspaper_options = {
                "edition": edition_key,
                "enable_sfx": news_enable_sfx,
                "sfx_volume": news_sfx_vol,
                "use_tts": news_use_tts,
                "sfx_only_duration": float(news_sfx_dur),
                "date_line": news_date_line,
                "edition_info": news_edition_info,
                "table_path": resolve_table_path(),
            }

            try:
                with st.spinner(f"Processing {pipeline_mode.upper()} video engine pipeline..."):
                    output_file = process_video_pipeline(
                        script_data=parsed_script,
                        bing_cookie_u=bing_cookie,
                        voice_name=selected_voice,
                        pipeline_mode=pipeline_mode,
                        progress_callback=ui_progress_callback,
                        newspaper_options=newspaper_options
                    )

                prog_bar.progress(1.0)
                status_box.success(f"🎉 Master video rendered successfully in [{pipeline_mode.upper()}] mode!")

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
                        st.write(f"• **Mode:** `{pipeline_mode.upper()}`")
                        st.write(f"• **Size:** `{file_size_mb:.2f} MB`")
                        st.write("• **Resolution:** `1920x1080` (30 FPS)")
                        if pipeline_mode == "newspaper":
                            audio_lbl = f"{selected_voice} + SFX ({int(news_sfx_vol * 100)}%)" if news_use_tts else "Newspaper SFX Only"
                            st.write(f"• **Audio:** `{audio_lbl}`")
                            st.write(f"• **Edition:** `{newspaper_edition}`")
                        else:
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
    st.subheader("Pipeline Modes Overview")
    st.markdown("""
    | Mode | Visual Logic | Audio & Timing |
    | :--- | :--- | :--- |
    | **Default 3-Beat Pipeline** | Beat 1: Mixkit/Wikimedia stock b-roll video.<br>Beat 2: Looping background + die-cut stamp card.<br>Beat 3: Procedural OpenCV kinetic animation. | Cartesia TTS duration + 0.4s fade-out per scene. |
    | **Whiteboard Animation Mode** | **Math / Text:** Living hand draws Malayalam text, English headers, and boxed LaTeX formulas.<br>**Visuals:** Hand sketches vector line art on pure white canvas via Bing AI / Wikimedia. | Visuals are drawn at the exact speed of the Cartesia TTS voiceover. |
    | **Newspaper Animation Mode** | Symmetrical paper drop onto wooden table (`table.png`), adaptive 2/3-column Malayalam/English typography, red underlines, yellow highlighter marker sweeps & camera pan/zoom. | Cartesia TTS mixed with stereo paper drop + marker SFX (or SFX-only mode). |
    """)

    st.subheader("Plain Text (.txt) Format Sample (Works for All Modes)")
    st.code("""Sentence in Malayalam or English | Highlights (comma-separated) or Prompt/LaTeX | Headline | Subheadline
കേരളത്തിൽ വൻ വികസന കുതിപ്പ് | ചരിത്ര നേട്ടം, ₹12,000 കോടി പദ്ധതികൾ, മെട്രോ രണ്ടാം ഘട്ടം | കേരളത്തിൽ വൻ വികസന കുതിപ്പ് | മെട്രോ രണ്ടാം ഘട്ടവും ദേശീയപാതയും ഉദ്ഘാടനം ചെയ്തു
ഐൻസ്റ്റീന്റെ ഊർജ്ജ സമവാക്യം | E = mc^2 | Mass Energy Equivalence""", language="text")

    st.subheader("Newspaper JSON Schema (Optional Rich Fields)")
    st.code("""[
  {
    "sentence": "വാചകം ഇവിടെ ചേർക്കുക...",
    "newspaper": "manorama",
    "headline": "പ്രധാന തലക്കെട്ട്",
    "subheadline": "ഉപതലക്കെട്ട്...",
    "yellow_highlights": ["പ്രധാന പോയിന്റ് 1", "പ്രധാന പോയിന്റ് 2"],
    "story_heading": "വിശദാംശങ്ങൾ",
    "body_paragraphs": ["ഖണ്ഡിക 1...", "ഖണ്ഡിക 2..."]
  }
]""", language="json")


# -------------------------------------------------------------
# TAB 3: REGISTERED ENGINES
# -------------------------------------------------------------
with tabs[2]:
    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.subheader("Loaded Kinetic Animation Engines")
        for idx, anim in enumerate(ANIMATION_REGISTRY, start=1):
            st.markdown(f"**{idx}. {anim['name']}** (`{anim['id']}`)")

    with col_a2:
        st.subheader("Whiteboard Subsystems")
        st.write("• **FastLivingHandEngine:** 9 wrist-pivot poses + diffused studio shadows.")
        st.write("• **Text & Math Engine:** Zhang-Suen morphological thinning + STIX / Computer Modern LaTeX.")
        st.write("• **Vector Grid Ink Engine:** Continuous nearest-neighbor line revelation.")

    with col_a3:
        st.subheader("Newspaper Subsystems")
        st.write("• **Any-Text Normalizer:** Auto-extracts headlines, highlights & balanced columns from any text or JSON.")
        st.write("• **Shaped Typography:** `libraqm` HarfBuzz Malayalam shaping + auto-scaling `13px–36px`.")
        st.write("• **Synced Stereo SFX:** Paper whoosh/thud + felt-tip marker sweeps mixed via FFmpeg.")
