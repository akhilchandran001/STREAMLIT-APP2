import asyncio
import json
import logging
import os
import time
from pyrogram import Client, filters
from pyrogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from pyrogram.errors import FloodWait

from modules.script_parser import process_video_pipeline
from modules.text_parser import parse_input_script

logger = logging.getLogger("VideoEngine.TelegramHandler")

# Temporary cache mapping user_id -> parsed script data
PENDING_JOBS = {}

START_MESSAGE = (
    "🎬 **Welcome to the Automated Video Engine Studio!**\n\n"
    "Send me a `.json` or `.txt` file, or paste your script text.\n\n"
    "⚙️ **Two Generation Modes:**\n"
    "1️⃣ **Default 3-Beat Rhythm:** Stock landscape video + stamp card + procedural kinetic animation.\n"
    "2️⃣ **Whiteboard Animation:** Pure hand-drawn animation with living hand physics, math LaTeX equations, and vector sketching (driven by Cartesia TTS durations and Bing image generation).\n"
)

def get_bing_cookies():
    try:
        import streamlit as st
        if "BING_COOKIES" in st.secrets:
            return list(st.secrets["BING_COOKIES"])
        if "BING_COOKIE_U" in st.secrets:
            return st.secrets["BING_COOKIE_U"]
    except Exception:
        pass
    return os.getenv("BING_COOKIES") or os.getenv("BING_COOKIE_U") or ""

async def safe_edit_message(message: Message, text: str):
    for attempt in range(3):
        try:
            await message.edit_text(text)
            return
        except FloodWait as e:
            await asyncio.sleep(e.value)
        except Exception:
            break

async def prompt_user_mode_selection(client: Client, message: Message, script_data: list):
    """Presents the two interactive buttons to the user before starting."""
    user_id = message.from_user.id
    job_key = f"{user_id}_{int(time.time())}"
    PENDING_JOBS[job_key] = script_data

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🎬 Default Pipeline (Normal)", callback_data=f"mode:default:{job_key}"),
            InlineKeyboardButton("✍️ Whiteboard Animation", callback_data=f"mode:whiteboard:{job_key}")
        ]
    ])

    await message.reply_text(
        f"📋 **Script Loaded Successfully!** ({len(script_data)} scene(s) detected)\n\n"
        "Please choose your preferred animation engine:",
        reply_markup=keyboard
    )

async def execute_pipeline_for_user(client: Client, message: Message, script_data: list, mode: str):
    chat_id = message.chat.id
    status_msg = await message.reply_text(f"🚀 Initializing [{mode.upper()}] Pipeline...")

    bing_cookies = get_bing_cookies()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    final_output_video = os.path.join(output_dir, f"video_{mode}_{int(time.time())}.mp4")

    loop = asyncio.get_running_loop()

    def progress_callback(status_text: str):
        asyncio.run_coroutine_threadsafe(
            safe_edit_message(status_msg, f"⏳ {status_text}"),
            loop
        )

    def artifact_callback(artifact_type: str, file_path: str, caption: str):
        if not os.path.exists(file_path):
            return

        async def _upload():
            for attempt in range(3):
                try:
                    if artifact_type == "audio":
                        await client.send_audio(chat_id, audio=file_path, caption=caption)
                    elif artifact_type == "image":
                        await client.send_photo(chat_id, photo=file_path, caption=caption)
                    elif artifact_type in ["video", "final_video"]:
                        await client.send_video(
                            chat_id,
                            video=file_path,
                            caption=caption,
                            supports_streaming=True
                        )
                    return
                except FloodWait as fw:
                    await asyncio.sleep(fw.value)
                except Exception as err:
                    if attempt == 2:
                        raise err
                    await asyncio.sleep(2)

        future = asyncio.run_coroutine_threadsafe(_upload(), loop)
        try:
            future.result(timeout=300)
        except Exception as e:
            logger.error(f"Upload failed: {e}")

    try:
        await loop.run_in_executor(
            None,
            lambda: process_video_pipeline(
                script_data=script_data,
                output_video_path=final_output_video,
                bing_cookie=bing_cookies,
                pipeline_mode=mode,
                progress_callback=progress_callback,
                artifact_callback=artifact_callback
            )
        )
        completion_text = (
            f"✅ **Render Complete! [{mode.upper()} MODE]**\n"
            f"• Scenes: {len(script_data)}\n"
            f"• Resolution: 1920x1080 (30 FPS)"
        )
        await safe_edit_message(status_msg, completion_text)

    except Exception as e:
        logger.exception("Pipeline execution failed")
        await safe_edit_message(status_msg, f"❌ Execution Failed:\n`{str(e)}`")

# =====================================================================
# PYROGRAM HANDLERS
# =====================================================================

@Client.on_message(filters.command("start") & filters.private)
async def handle_start_command(client: Client, message: Message):
    await message.reply_text(START_MESSAGE, disable_web_page_preview=True)

@Client.on_message(filters.document & filters.private)
async def handle_document_upload(client: Client, message: Message):
    fname = message.document.file_name.lower()
    if not (fname.endswith(".json") or fname.endswith(".txt")):
        await message.reply_text("⚠️ Please upload a valid `.json` or `.txt` file.")
        return

    temp_path = await message.download()
    try:
        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        script_data = parse_input_script(raw_text)
        await prompt_user_mode_selection(client, message, script_data)
    except Exception as err:
        await message.reply_text(f"❌ Could not parse file:\n`{str(err)}`")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@Client.on_message(filters.text & filters.private)
async def handle_raw_text(client: Client, message: Message):
    text = message.text.strip()
    if text.startswith("/"):
        return

    try:
        script_data = parse_input_script(text)
        await prompt_user_mode_selection(client, message, script_data)
    except Exception as err:
        await message.reply_text(f"❌ Error parsing script text:\n`{str(err)}`")

@Client.on_callback_query(filters.regex(r"^mode:(default|whiteboard):(.+)$"))
async def handle_mode_callback(client: Client, callback: CallbackQuery):
    mode = callback.matches[0].group(1)
    job_key = callback.matches[0].group(2)

    script_data = PENDING_JOBS.get(job_key)
    if not script_data:
        await callback.answer("⚠️ Session expired. Please re-send your script.", show_alert=True)
        return

    await callback.answer(f"Selected: {mode.title()} Mode")
    await callback.message.edit_reply_markup(reply_markup=None)
    await execute_pipeline_for_user(client, callback.message, script_data, mode=mode)
