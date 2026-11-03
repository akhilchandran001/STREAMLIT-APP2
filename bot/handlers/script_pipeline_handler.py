import asyncio
import json
import os
import time
from pyrogram import Client, filters
from pyrogram.types import Message
from pyrogram.errors import FloodWait, RPCError

from modules.script_parser import process_video_pipeline

BASE_DIR = os.getcwd()
TEMP_DOWNLOAD_DIR = os.path.join(BASE_DIR, "storage", "temp", "telegram_uploads")


def get_bing_cookie() -> str:
    """Retrieves Bing cookie from Streamlit secrets or environment variables."""
    try:
        import streamlit as st
        cookie = st.secrets.get("BING_COOKIE_U")
        if cookie and cookie != "your_cookie_here":
            return cookie
    except Exception:
        pass
    return os.getenv("BING_COOKIE_U", "")


async def safe_edit_message(msg: Message, text: str) -> None:
    """Edits status messages safely while handling Telegram rate limits (FloodWait)."""
    try:
        await msg.edit_text(text)
    except FloodWait as e:
        await asyncio.sleep(e.value)
        try:
            await msg.edit_text(text)
        except Exception:
            pass
    except RPCError:
        pass
    except Exception:
        pass


@Client.on_message(filters.command(["start", "help"]) & filters.private)
async def handle_start_help(client: Client, message: Message):
    """Explains engine operation, scene rules, and JSON input schema."""
    help_text = (
        "🎬 **Welcome to the Automated Video Engine Bot!**\n\n"
        "Send me a `.json` script file or paste your raw JSON text to trigger video generation.\n\n"
        "📋 **Scene Lifecycle:**\n"
        "• **Sentence 1**: Unique landscape video from Pixabay\n"
        "• **Sentence 2, 4, 6...**: Background video (`BG1`, `BG2`...) + Centered Bing/Pixabay image\n"
        "• **Sentence 3, 5, 7...**: Procedural kinetic animations (Velocity Ramp, Orbit Freeze, Tape Slam...)\n"
        "• **Every Scene**: Standardized with a 0.4s smooth fade-out\n\n"
        "📝 **Expected JSON Schema:**\n"
        "```json\n"
        "[\n"
        "  {\n"
        "    \"sentence\": \"നമസ്കാരം, ഇത് ആദ്യത്തെ രംഗമാണ്.\",\n"
        "    \"query\": \"futuristic technology nature\"\n"
        "  },\n"
        "  {\n"
        "    \"sentence\": \"രണ്ടാമത്തെ രംഗത്തിൽ ഇമേജ് പശ്ചാത്തലത്തിൽ വരുന്നു.\",\n"
        "    \"prompt\": \"ancient temple carved into floating mountain in clouds\",\n"
        "    \"source\": \"bing\"\n"
        "  }\n"
        "]\n"
        "```"
    )
    await message.reply_text(help_text, disable_web_page_preview=True)


async def execute_pipeline_for_user(client: Client, message: Message, script_obj: list[dict]):
    """Orchestrates pipeline execution, progress reporting, and final delivery."""
    bing_cookie = get_bing_cookie()
    status_msg = await message.reply_text("⏳ **Initializing Video Engine Pipeline...**")

    last_edit_time = 0.0

    def progress_callback(idx: int, total: int, step_desc: str):
        nonlocal last_edit_time
        now = time.time()
        # Rate limit Telegram message updates to avoid flood restrictions
        if (now - last_edit_time >= 1.5) or (idx == total):
            last_edit_time = now
            percent = int((idx / max(total, 1)) * 100)
            status_text = (
                f"🎬 **Rendering Video Engine**\n\n"
                f"**Progress:** {percent}% [{idx}/{total} Scenes]\n"
                f"**Current Task:** {step_desc}"
            )
            asyncio.create_task(safe_edit_message(status_msg, status_text))

    try:
        # Run core rendering pipeline
        master_video_path = await process_video_pipeline(
            script_data=script_obj,
            bing_cookie_u=bing_cookie,
            voice_name="Shreeraj_HindiProfile",
            progress_callback=progress_callback
        )

        if not os.path.exists(master_video_path):
            raise FileNotFoundError("Master output MP4 was not created.")

        await safe_edit_message(status_msg, "🚀 **Uploading finalized 1080p video to Telegram...**")

        caption = (
            f"✅ **Render Complete!**\n"
            f"• **Scenes:** {len(script_obj)}\n"
            f"• **Resolution:** 1920x1080 (30 FPS)\n"
            f"• **TTS Voice:** Shreeraj_HindiProfile\n"
            f"• **Fade Transitions:** 0.4s applied per scene"
        )

        await message.reply_video(
            video=master_video_path,
            caption=caption,
            supports_streaming=True
        )
        await status_msg.delete()

    except Exception as e:
        error_msg = f"❌ **Pipeline Execution Failed:**\n`{str(e)}`"
        await safe_edit_message(status_msg, error_msg)
        print(f"[Telegram Handler] Pipeline Error: {e}")


@Client.on_message(filters.document & filters.private)
async def handle_json_document(client: Client, message: Message):
    """Processes uploaded .json script files."""
    doc = message.document
    if not (doc.file_name.lower().endswith(".json") or doc.mime_type == "application/json"):
        await message.reply_text("⚠️ Please send a valid `.json` file containing your scene script.")
        return

    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    temp_json_path = os.path.join(TEMP_DOWNLOAD_DIR, f"script_{message.id}_{int(time.time())}.json")

    download_msg = await message.reply_text("📥 **Downloading script document...**")
    try:
        await message.download(file_name=temp_json_path)
        with open(temp_json_path, "r", encoding="utf-8") as f:
            script_obj = json.load(f)

        if not isinstance(script_obj, list) or len(script_obj) == 0:
            await download_msg.edit_text("❌ JSON must contain a non-empty array `[...]` of scene objects.")
            return

        await download_msg.delete()
        await execute_pipeline_for_user(client, message, script_obj)

    except json.JSONDecodeError as err:
        await download_msg.edit_text(f"❌ **Invalid JSON Syntax:**\n`{err}`")
    except Exception as err:
        await download_msg.edit_text(f"❌ **Failed processing document:**\n`{err}`")
    finally:
        if os.path.exists(temp_json_path):
            try:
                os.remove(temp_json_path)
            except Exception:
                pass


@Client.on_message(filters.text & filters.private & ~filters.command(["start", "help"]))
async def handle_json_text(client: Client, message: Message):
    """Processes raw JSON arrays pasted directly into chat."""
    raw_text = message.text.strip()

    # Fast verification for JSON array syntax
    if not (raw_text.startswith("[") and raw_text.endswith("]")):
        return

    try:
        script_obj = json.loads(raw_text)
        if not isinstance(script_obj, list) or len(script_obj) == 0:
            return

        await execute_pipeline_for_user(client, message, script_obj)

    except json.JSONDecodeError:
        await message.reply_text("⚠️ Text resembles JSON but failed to parse. Please check your syntax.")
    except Exception as e:
        await message.reply_text(f"❌ **Error:** `{e}`")
