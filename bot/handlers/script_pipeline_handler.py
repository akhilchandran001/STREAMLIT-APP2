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
    """Retrieves Bing cookie from secrets or environment."""
    try:
        import streamlit as st
        cookie = st.secrets.get("BING_COOKIE_U", "")
        if cookie and cookie.strip():
            return cookie.strip()
    except Exception:
        pass
    return os.getenv("BING_COOKIE_U", "").strip()


async def safe_edit_message(message: Message, text: str):
    """Safely updates Telegram progress message with FloodWait protection."""
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

    # Thread-safe progress callback
    def progress_callback(status_text: str):
        asyncio.run_coroutine_threadsafe(
            safe_edit_message(status_msg, f"⏳ {status_text}"),
            loop
        )

    # Thread-safe artifact dispatcher
    def artifact_callback(artifact_type: str, file_path: str, caption: str):
        async def _upload():
            try:
                if artifact_type == "audio":
                    await client.send_audio(chat_id, audio=file_path, caption=caption)
                elif artifact_type == "image":
                    await client.send_photo(chat_id, photo=file_path, caption=caption)
                elif artifact_type in ["video", "final_video"]:
                    await client.send_video(chat_id, video=file_path, caption=caption, supports_streaming=True)
            except Exception as err:
                logger.error(f"Failed to upload {artifact_type} artifact: {err}")

        asyncio.run_coroutine_threadsafe(_upload(), loop)

    try:
        # Run synchronous video generation in background thread
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
        await safe_edit_message(status_msg, "✅ Video rendering and delivery complete!")

    except Exception as e:
        logger.exception("Pipeline failed during execution")
        await safe_edit_message(status_msg, f"❌ Pipeline Execution Failed:\n`{str(e)}`")


@Client.on_message(filters.document & filters.private)
async def handle_json_document(client: Client, message: Message):
    """Handles uploaded .json script files."""
    if not message.document.file_name.endswith(".json"):
        return

    temp_path = await message.download()
    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        if not isinstance(script_data, list):
            await message.reply_text("❌ Invalid JSON schema: Root must be an array of scene objects.")
            return

        await execute_pipeline_for_user(client, message, script_data)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@Client.on_message(filters.text & filters.private)
async def handle_json_text(client: Client, message: Message):
    """Handles pasted raw JSON text."""
    text = message.text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            script_data = json.loads(text)
            await execute_pipeline_for_user(client, message, script_data)
        except json.JSONDecodeError as err:
            await message.reply_text(f"❌ JSON Syntax Error:\n`{str(err)}`")
