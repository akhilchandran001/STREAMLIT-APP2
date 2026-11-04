import asyncio
import json
import logging
import os
import time
from pyrogram import Client, filters
from pyrogram.types import Message
from pyrogram.errors import FloodWait

from modules.script_parser import process_video_pipeline

logger = logging.getLogger("VideoEngine.TelegramHandler")


def get_bing_cookie() -> str:
    """Retrieves Bing cookie from Streamlit secrets or environment."""
    try:
        import streamlit as st
        cookie = st.secrets.get("BING_COOKIE_U", "")
        if cookie and cookie.strip():
            return cookie.strip()
    except Exception:
        pass
    return os.getenv("BING_COOKIE_U", "").strip()


async def safe_edit_message(message: Message, text: str):
    """Safely updates Telegram progress messages with FloodWait protection."""
    for attempt in range(3):
        try:
            await message.edit_text(text)
            return
        except FloodWait as e:
            await asyncio.sleep(e.value)
        except Exception:
            break


async def execute_pipeline_for_user(client: Client, message: Message, script_data: list):
    """Orchestrates pipeline execution, status progress, and real-time artifact delivery."""
    chat_id = message.chat.id
    status_msg = await message.reply_text("🚀 Pipeline Initialized! Processing script...")

    bing_cookie = get_bing_cookie()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    final_output_video = os.path.join(output_dir, f"video_{int(time.time())}.mp4")

    loop = asyncio.get_running_loop()

    def progress_callback(status_text: str):
        asyncio.run_coroutine_threadsafe(
            safe_edit_message(status_msg, f"⏳ {status_text}"),
            loop
        )

    def artifact_callback(artifact_type: str, file_path: str, caption: str):
        if not os.path.exists(file_path):
            logger.error(f"Cannot upload missing artifact: {file_path}")
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
                    logger.warning(f"FloodWait on {artifact_type}. Waiting {fw.value}s...")
                    await asyncio.sleep(fw.value)
                except Exception as err:
                    logger.error(f"Upload attempt {attempt + 1} failed: {err}")
                    if attempt == 2:
                        raise err
                    await asyncio.sleep(2)

        future = asyncio.run_coroutine_threadsafe(_upload(), loop)
        try:
            future.result(timeout=300)
        except Exception as e:
            logger.error(f"Upload failed or timed out for {file_path}: {e}")

    try:
        await loop.run_in_executor(
            None,
            lambda: process_video_pipeline(
                script_data=script_data,
                output_video_path=final_output_video,
                bing_cookie=bing_cookie,
                progress_callback=progress_callback,
                artifact_callback=artifact_callback
            )
        )
        completion_text = (
            f"✅ **Render Complete!**\n"
            f"• Scenes: {len(script_data)}\n"
            f"• Resolution: 1920x1080 (30 FPS)\n"
            f"• Fade Transitions: 0.4s applied per scene"
        )
        await safe_edit_message(status_msg, completion_text)

    except Exception as e:
        logger.exception("Pipeline execution failed")
        await safe_edit_message(status_msg, f"❌ Pipeline Execution Failed:\n`{str(e)}`")


# =====================================================================
# TELEGRAM MESSAGE HANDLERS
# =====================================================================

@Client.on_message(filters.command("start") & filters.private)
async def handle_start_command(client: Client, message: Message):
    """Answers /start immediately."""
    await message.reply_text(
        "👋 **Video Engine Bot is Online!**\n\n"
        "Send me a `.json` script file or paste a JSON array directly to generate your video."
    )


@Client.on_message(filters.document & filters.private)
async def handle_json_document(client: Client, message: Message):
    """Handles uploaded .json script files."""
    if not message.document.file_name.endswith(".json"):
        await message.reply_text("⚠️ Please upload a valid `.json` script file.")
        return

    temp_path = await message.download()
    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        if not isinstance(script_data, list):
            await message.reply_text("❌ Invalid JSON schema: Root must be an array `[...]` of scene objects.")
            return

        await execute_pipeline_for_user(client, message, script_data)
    except json.JSONDecodeError as err:
        await message.reply_text(f"❌ Invalid JSON file syntax:\n`{str(err)}`")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@Client.on_message(filters.text & filters.private)
async def handle_json_text(client: Client, message: Message):
    """Handles raw pasted JSON text or displays a helpful usage tip."""
    text = message.text.strip()

    if text.startswith("/"):
        return

    if text.startswith("[") and text.endswith("]"):
        try:
            script_data = json.loads(text)
            if not isinstance(script_data, list):
                await message.reply_text("❌ JSON root must be an array of scene objects.")
                return
            await execute_pipeline_for_user(client, message, script_data)
        except json.JSONDecodeError as err:
            await message.reply_text(f"❌ JSON Syntax Error:\n`{str(err)}`")
    else:
        await message.reply_text(
            "⚠️ Please send a valid JSON script array (starting with `[` and ending with `]`) "
            "or upload a `.json` file."
        )
