import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
import requests
from pyrogram import Client, filters
from pyrogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from pyrogram.errors import FloodWait

from modules.script_parser import process_video_pipeline
from modules.text_parser import parse_input_script

logger = logging.getLogger("VideoEngine.TelegramHandler")

PENDING_JOBS = {}

START_MESSAGE = (
    "🎬 **Welcome to the Automated Video Engine Studio!**\n\n"
    "Send me a `.json` or `.txt` file, or paste your script text.\n\n"
    "⚙️ **Two Generation Modes:**\n"
    "1️⃣ **Default (Normal):** Stock landscape video + stamp card + procedural kinetic animation (plus Newspaper scenes when specified in JSON).\n"
    "2️⃣ **Whiteboard Animation:** 2 sentences per visual — hand-drawn stick-figure/vector sketching, math LaTeX, or Newspaper animation when specified in JSON.\n"
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


def upload_for_direct_url(file_path: str) -> tuple[str, str]:
    """
    Uploads the final video and returns (download_url, provider_label).
    Streams via curl when available to keep memory usage minimal.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    curl_bin = shutil.which("curl")
    errors = []

    # 1. Primary: Litterbox (Direct .mp4 URL, up to 1 GB, 72h retention)
    if size_mb <= 950:
        lb_url = "https://litterbox.catbox.moe/resources/internals/api.php"
        try:
            if curl_bin:
                res = subprocess.run(
                    [
                        curl_bin, "-sS", "--fail", "--max-time", "600",
                        "-F", "reqtype=fileupload",
                        "-F", "time=72h",
                        "-F", f"fileToUpload=@{file_path}",
                        lb_url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=620,
                )
                link = res.stdout.strip()
                if res.returncode == 0 and link.startswith("https://"):
                    return link, "Litterbox (Direct MP4, 72h)"
                errors.append(f"Litterbox curl: {res.stderr.strip() or link}")
            else:
                with open(file_path, "rb") as f:
                    resp = requests.post(
                        lb_url,
                        data={"reqtype": "fileupload", "time": "72h"},
                        files={"fileToUpload": f},
                        timeout=600,
                    )
                link = resp.text.strip()
                if resp.status_code == 200 and link.startswith("https://"):
                    return link, "Litterbox (Direct MP4, 72h)"
                errors.append(f"Litterbox HTTP {resp.status_code}: {link[:120]}")
        except Exception as e:
            errors.append(f"Litterbox error: {e}")

    # 2. Fallback: Gofile (Unlimited size, no auth required)
    gf_url = "https://upload.gofile.io/uploadfile"
    try:
        if curl_bin:
            res = subprocess.run(
                [
                    curl_bin, "-sS", "--fail", "--max-time", "600",
                    "-F", f"file=@{file_path}",
                    gf_url,
                ],
                capture_output=True,
                text=True,
                timeout=620,
            )
            if res.returncode == 0 and res.stdout.strip():
                payload = json.loads(res.stdout)
                if payload.get("status") == "ok":
                    return payload["data"]["downloadPage"], "Gofile"
            errors.append(f"Gofile curl: {res.stderr.strip() or res.stdout[:120]}")
        else:
            with open(file_path, "rb") as f:
                resp = requests.post(gf_url, files={"file": f}, timeout=600)
            payload = resp.json()
            if payload.get("status") == "ok":
                return payload["data"]["downloadPage"], "Gofile"
            errors.append(f"Gofile error: {payload}")
    except Exception as e:
        errors.append(f"Gofile error: {e}")

    raise RuntimeError(" | ".join(errors))


async def safe_edit_message(message: Message, text: str):
    for attempt in range(3):
        try:
            await message.edit_text(text, disable_web_page_preview=True)
            return
        except FloodWait as e:
            await asyncio.sleep(e.value)
        except Exception:
            break


async def prompt_user_mode_selection(client: Client, message: Message, script_data: list):
    """Presents the two interactive mode buttons to the user before starting."""
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
    final_download_info = {"url": None, "provider": None}

    def progress_callback(status_text: str):
        asyncio.run_coroutine_threadsafe(
            safe_edit_message(status_msg, f"⏳ {status_text}"),
            loop
        )

    def artifact_callback(artifact_type: str, file_path: str, caption: str):
        if not os.path.exists(file_path):
            return

        # For the final combined video, generate & send a direct download URL first
        if artifact_type == "final_video":
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                progress_callback(f"🌐 Generating direct download URL ({size_mb:.1f} MB)...")
                dl_url, provider = upload_for_direct_url(file_path)
                final_download_info["url"] = dl_url
                final_download_info["provider"] = provider

                async def _send_link():
                    await client.send_message(
                        chat_id,
                        f"🏆 **Final Combined Video Ready!**\n"
                        f"• **File:** `{os.path.basename(file_path)}` (`{size_mb:.1f} MB`)\n"
                        f"• **Host:** {provider}\n"
                        f"• **Direct Download URL:** {dl_url}",
                        disable_web_page_preview=True
                    )

                link_future = asyncio.run_coroutine_threadsafe(_send_link(), loop)
                link_future.result(timeout=60)
            except Exception as link_err:
                logger.exception(f"Direct URL upload failed: {link_err!r}")

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

        upload_timeout = 1800 if artifact_type == "final_video" else 300
        future = asyncio.run_coroutine_threadsafe(_upload(), loop)
        try:
            future.result(timeout=upload_timeout)
        except Exception as e:
            future.cancel()
            logger.exception(f"Upload failed ({type(e).__name__}) for {artifact_type}: {e!r}")

    newspaper_options = {
        "edition": "alternating",
        "enable_sfx": True,
        "sfx_volume": 0.45,
        "use_tts": True,
    }

    try:
        await loop.run_in_executor(
            None,
            lambda: process_video_pipeline(
                script_data=script_data,
                output_video_path=final_output_video,
                bing_cookie=bing_cookies,
                pipeline_mode=mode,
                progress_callback=progress_callback,
                artifact_callback=artifact_callback,
                newspaper_options=newspaper_options
            )
        )
        completion_text = (
            f"✅ **Render Complete! [{mode.upper()} MODE]**\n"
            f"• Scenes: {len(script_data)}\n"
            f"• Resolution: 1920x1080 (30 FPS)"
        )
        if final_download_info["url"]:
            completion_text += f"\n• **Direct Download:** {final_download_info['url']}"
        await safe_edit_message(status_msg, completion_text)

    except Exception as e:
        logger.exception("Pipeline execution failed")
        await safe_edit_message(status_msg, f"❌ Execution Failed:\n`{str(e)}`")


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

    script_data = PENDING_JOBS.pop(job_key, None)
    if not script_data:
        await callback.answer("⚠️ Session expired. Please re-send your script.", show_alert=True)
        return

    await callback.answer(f"Selected: {mode.title()} Mode")
    await callback.message.edit_reply_markup(reply_markup=None)
    await execute_pipeline_for_user(client, callback.message, script_data, mode=mode)
