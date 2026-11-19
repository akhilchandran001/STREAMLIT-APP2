import asyncio
import json
import os
import shutil
import glob
import streamlit as st

from bot.bot_runner import start_bot, stop_bot, is_bot_active
from modules.cartesia_tts import VOICE_LIST, DEFAULT_VOICE_NAME
from modules.script_parser import process_video_pipeline, get_available_backgrounds, BG_DIR, OUTPUT_DIR
from modules.text_parser import parse_input_script
from modules.bing_fetcher import parse_cookie_pool
from modules.newspaper_anim import (
    NEWSPAPER_ASSETS_DIR,
    LOGOS_DIR,
    FONTS_DIR,
    resolve_table_path,
    resolve_logo_path,
)
from animations.animation_router import ANIMATION_REGISTRY

st.set_page_config(
    page_title="Video Engine Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs(BG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(NEWSPAPER_ASSETS_DIR, exist_ok=True)
os.makedirs(LOGOS_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)

if "_saved_uploads" not in st.session_state:
    st.session_state["_saved_uploads"] = set()


def _save_uploaded_once(uploaded_file, dest_path: str) -> bool:
    sig = (dest_path, uploaded_file.name, uploaded_file.size)
    if sig in st.session_state["_saved_uploads"]:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["_saved_uploads"].add(sig)
    return True


@st.cache_resource(show_spinner=False)
def initialize_telegram_bot_daemon():
    return start_bot()


bot_started = initialize_telegram_bot_daemon()


with st.sidebar:
    st.title("⚙️ Engine Settings")

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

    st.subheader("🎙️ TTS Voice")
    voice_names = [v["name"] for v in VOICE_LIST]
    default_idx = voice_names.index(DEFAULT_VOICE_NAME) if DEFAULT_VOICE_NAME in voice_names else 0
    selected_voice = st.selectbox("Select Malayalam Voice Profile", options=voice_names, index=default_idx)

    st.divider()

    st.subheader("🔑 Bing Authentication")
    secrets_cookies = ""
    try:
        if "BING_COOKIES" in st.secrets:
            secrets_cookies = "\n".join(list(st.secrets["BING_COOKIES"]))
        elif "BING_COOKIE_U" in st.secrets:
            secrets_cookies = st.secrets.get("BING_COOKIE_U", "")
    except Exception:
        pass

    bing_cookie = st.text_area(
        "Bing `_U` Cookie(s) (comma or newline separated)",
        value=secrets_cookies,
        height=80,
        help="Session cookie(s) for Microsoft Designer / Bing Image Creator MAI1."
    )
    loaded_cookie_count = len(parse_cookie_pool(bing_cookie))
    if loaded_cookie_count > 0:
        st.success(f"🟢 Bing Cookies Loaded: {loaded_cookie_count}")
    else:
        st.warning("🟡 No Bing `_U` cookies loaded — will fall back to Wikimedia")

    st.divider()

    # Common Newspaper Settings (Applies whenever JSON triggers a Newspaper scene in either mode)
    with st.expander("📰 Newspaper Scene Settings & Assets", expanded=False):
        newspaper_edition = st.selectbox(
            "Default Newspaper Edition",
            ["Alternating (Manorama ↔ Mathrubhumi)", "Malayala Manorama", "Mathrubhumi"],
            index=0
        )
        edition_key = (
            "alternating" if "Alternating" in newspaper_edition
            else ("mathrubhumi" if "Mathrubhumi" in newspaper_edition else "manorama")
        )
        news_enable_sfx = st.checkbox("🔊 Enable Drop & Highlighter SFX", value=True)
        news_sfx_vol = st.slider("SFX Mix Volume (Ducked under TTS)", 0.0, 1.0, 0.45, 0.05, disabled=not news_enable_sfx)
        news_date_line = st.text_input("Default Date Line", value="2026 നവംബർ 19, വ്യാഴം")
        news_edition_info = st.text_input("Default Edition Info", value="തിരുവനന്തപുരം • പേജ് 1 • വില ₹8.00")

        st.caption(
            f"Table: {'✅' if resolve_table_path() else '⚠️ Fallback'} | "
            f"Manorama: {'✅' if resolve_logo_path('manorama') else '⚠️ Text'} | "
            f"Mathrubhumi: {'✅' if resolve_logo_path('mathrubhumi') else '⚠️ Text'}"
        )

        up_table = st.file_uploader("Upload `table.png`", type=["png", "jpg", "jpeg"], key="up_news_table")
        if up_table:
            ext = os.path.splitext(up_table.name)[1].lower() or ".png"
            if _save_uploaded_once(up_table, os.path.join(NEWSPAPER_ASSETS_DIR, f"table{ext}")):
                st.success(f"Saved table background (`table{ext}`).")
                st.rerun()

        up_logos = st.file_uploader("Upload Newspaper Logos", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="up_news_logos")
        if up_logos:
            saved_any = False
            for lf in up_logos:
                if _save_uploaded_once(lf, os.path.join(LOGOS_DIR, lf.name)):
                    saved_any = True
            if saved_any:
                st.success(f"Saved {len(up_logos)} logo(s).")
                st.rerun()

        up_fonts = st.file_uploader("Upload Malayalam Fonts (`.ttf`)", type=["ttf", "otf"], accept_multiple_files=True, key="up_news_fonts")
        if up_fonts:
            saved_any = False
            for ff in up_fonts:
                if _save_uploaded_once(ff, os.path.join(FONTS_DIR, ff.name)):
                    saved_any = True
            if saved_any:
                st.success(f"Saved {len(up_fonts)} font(s).")
                st.rerun()

    st.divider()

    st.subheader("🎞️ Canvas Backgrounds (Default Mode)")
    available_bgs = get_available_backgrounds()
    st.write(f"Loaded: `{len(available_bgs)}` video loops")

    uploaded_bgs = st.file_uploader("Upload Background Videos (.mp4)", type=["mp4"], accept_multiple_files=True)
    if uploaded_bgs:
        saved_any = False
        for uploaded_file in uploaded_bgs:
            if _save_uploaded_once(uploaded_file, os.path.join(BG_DIR, uploaded_file.name)):
                saved_any = True
        if saved_any:
            st.success(f"Saved {len(uploaded_bgs)} file(s).")
            st.rerun()


st.title("🎬 Multi-Scene Automated Video Engine")
st.caption("Synchronized Cartesia TTS • Whiteboard Stick-Figure & Math • Per-Scene Newspaper • Bing AI • Kinetic Animations")

tabs = st.tabs(["🚀 Run Pipeline", "📋 Script Schema & Sample", "🎨 Registered Engines"])

with tabs[0]:
    col_mode, col_src = st.columns([1, 1])

    with col_mode:
        selected_mode_label = st.radio(
            "🎬 Pipeline Rendering Mode",
            [
                "🎬 Default (3-Beat Video/Stamp/Animation)",
                "✍️ Whiteboard (Living Hand Drawing & Math)"
            ],
            index=0
        )
        pipeline_mode = "whiteboard" if "Whiteboard" in selected_mode_label else "default"

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
                "നമസ്കാരം, ഇത് വൈറ്റ്ബോർഡ് ആനിമേഷൻ സംവിധാനത്തിന്റെ ആദ്യ രംഗമാണ്. | intricate stick figure character standing beside giant lightbulb and laptop, pure white background\n"
                "രണ്ടാമത്തെ വാചകം ആദ്യ ചിത്രത്തോടൊപ്പം ചേരുന്നു. | stick figure laptop\n"
                "ഷ്രോഡിംഗർ സമവാക്യം ക്വാണ്ടം ഭൗതികത്തിലെ അടിസ്ഥാന സമവാക്യമാണ്. | i\\hbar \\frac{\\partial}{\\partial t}\\Psi = \\hat{H}\\Psi | Schrodinger Equation"
            )
        else:
            sample_content_default = json.dumps([
                {
                    "sentence": "നമസ്കാരം, ഇത് ആധുനിക വീഡിയോ നിർമ്മാണ സംവിധാനത്തിന്റെ ആദ്യ രംഗമാണ്.",
                    "query": "cinematic technology abstract landscape"
                },
                {
                    "sentence": "രണ്ടാമത്തെ രംഗത്തിൽ പശ്ചാത്തല വീഡിയോയുടെ മുകളിൽ ഇമേജ് കൃത്യമായ അനുപാതത്തിൽ വരുന്നു.",
                    "prompt": "intricate stick figure illustration with gears, servers, and rockets, pure white background",
                    "source": "bing"
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

            prog_bar = st.progress(0.0)
            status_box = st.empty()
            log_box = st.empty()
            live_logs = []

            def ui_progress_callback(step_desc: str, *args, **kwargs):
                status_box.info(f"⏳ {step_desc}")
                live_logs.append(step_desc)
                log_box.code("\n".join(live_logs[-8:]), language="text")

            newspaper_options = {
                "edition": edition_key,
                "enable_sfx": news_enable_sfx,
                "sfx_volume": news_sfx_vol,
                "use_tts": True,
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


with tabs[1]:
    st.subheader("Pipeline Modes & Per-Scene Newspaper Trigger")
    st.markdown("""
    | Mode | Visual Logic | Audio & Timing |
    | :--- | :--- | :--- |
    | **Default (Normal)** | Beat 1: Stock video.<br>Beat 2: Looping BG + stamp card.<br>Beat 3: Kinetic animation.<br>**If scene has `"type": "newspaper"`:** Renders Newspaper animation instead. | Cartesia TTS (+ ducked Paper/Marker SFX on Newspaper scenes). |
    | **Whiteboard Mode** | Pairs 2 sentences per visual (`1+2`, `3+4`, `5+6`).<br>**Odd sentence (`1, 3, 5...`) controls the pair:**<br>• `"type": "newspaper"` → 1 Newspaper animation.<br>• `"equation"` → Math/Text whiteboard.<br>• Otherwise → 1 Bing AI stick-figure drawing. | Visuals & SFX synced to combined 2-sentence Cartesia TTS. |
    """)

    st.subheader("Newspaper Scene JSON Fields (Works inside BOTH Modes)")
    st.code("""{
  "sentence": "കേരളത്തിൽ വൻ വികസന കുതിപ്പ്...",
  "type": "newspaper",
  "newspaper": "manorama",
  "headline": "പ്രധാന തലക്കെട്ട്",
  "subheadline": "ഉപതലക്കെട്ട്...",
  "yellow_highlights": ["പ്രധാന പോയിന്റ് 1", "പ്രധാന പോയിന്റ് 2"],
  "story_heading": "വിശദാംശങ്ങൾ",
  "body_paragraphs": ["ഖണ്ഡിക 1...", "ഖണ്ഡിക 2..."]
}""", language="json")


with tabs[2]:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.subheader("Loaded Kinetic Animation Engines")
        for idx, anim in enumerate(ANIMATION_REGISTRY, start=1):
            st.markdown(f"**{idx}. {anim['name']}** (`{anim['id']}`)")
    with col_a2:
        st.subheader("Whiteboard & Newspaper Subsystems")
        st.write("• **FastLivingHandEngine:** 9 wrist-pivot poses + diffused studio shadows.")
        st.write("• **Text & Math Engine:** Zhang-Suen morphological thinning + STIX LaTeX.")
        st.write("• **Newspaper Engine (Per-Scene):** Symmetrical table drop, adaptive columns, marker highlights & ducked stereo SFX.")
