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

# Full rich onboarding message reflecting the 3-beat rhythm and Mixkit/Wikimedia engine
START_MESSAGE = (
    "🎬 **Welcome to the Automated Video Engine Bot!**\n\n"
    "Send me a `.json` script file or paste your raw JSON text array to trigger video generation.\n\n"
    "📋 **Scene Rhythm (Strict 3-Beat Cycle):**\n"
    "• **Beat 1 (Scenes 1, 4, 7...):** Cinematic Landscape Video (Mixkit / Wikimedia Commons)\n"
    "• **Beat 2 (Scenes 2, 5, 8...):** Looping Background + Stamp-Drop Image + Styled Typography Card\n"
    "• **Beat 3 (Scenes 3, 6, 9...):** Procedural OpenCV Kinetic Animation (Velocity Ramp, 3D Orbit Freeze, Tape Slam)\n"
    "• **Every Scene:** Standardized 1080p 30fps with 0.4s smooth fade transitions\n"
    "• **Real-Time Delivery:** Each TTS audio, generated image, and scene MP4 is uploaded directly to your chat as it completes.\n\n"
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
    "    \"image_source\": \"auto\",\n"
    "    \"headline\": {\n"
    "      \"prefix\": \"Ancient\",\n"
    "      \"highlight\": \"Temple\",\n"
    "      \"suffix\": \"Mysteries\"\n"
    "    }\n"
    "  },\n"
    "  {\n"
    "    \"sentence\": \"മൂന്നാമത്തെ രംഗത്തിൽ അനിമേഷൻ വരുന്നു.\",\n"
    "    \"prompt\": \"cybernetic neon vortex kinetic energy explosion, 8k\",\n"
    "    \"source\": \"bing\"\n"
    "  }\n"
    "]\n"
    "```"
)


def get_bing_cookies():
    """Extracts all Bing cookies from Streamlit secrets or environment."""
    try:
        import streamlit as st
        if "BING_COOKIES" in st.secrets:
            return list(st.secrets["BING_COOKIES"])
        if "BING_COOKIE_U" in st.secrets:
            return st.secrets["BING_COOKIE_U"]
    except Exception:
        pass

    env_cookies = os.getenv("BING_COOKIES") or os.getenv("BING_COOKIE_U") or ""
    return env_cookies


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

    bing_cookies = get_bing_cookies()
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
                bing_cookie=bing_cookies,
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
# COMMAND & MESSAGE HANDLERS
# =====================================================================

@Client.on_message(filters.command("start") & filters.private)
async def handle_start_command(client: Client, message: Message):
    """Responds to /start with the full engine guide and schema documentation."""
    await message.reply_text(START_MESSAGE, disable_web_page_preview=True)


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
                await message.reply_text("❌ JSON root must be an array `[...]` of scene objects.")
                return
            await execute_pipeline_for_user(client, message, script_data)
        except json.JSONDecodeError as err:
            await message.reply_text(f"❌ JSON Syntax Error:\n`{str(err)}`")
    else:
        await message.reply_text(
            "⚠️ Please send a valid JSON script array (starting with `[` and ending with `]`) "
            "or upload a `.json` file.\n\nSend /start to view the expected format and schema."
        )
