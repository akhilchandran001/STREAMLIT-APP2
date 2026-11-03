import asyncio
import os
import sys
import threading
from pyrogram import Client

# Global thread reference and state guard
_bot_thread: threading.Thread | None = None
_stop_event: asyncio.Event | None = None
_loop: asyncio.AbstractEventLoop | None = None
bot_client: Client | None = None
_is_running = False
_lock = threading.Lock()


def get_telegram_credentials() -> tuple[int, str, str]:
    """
    Extracts Telegram bot credentials from Streamlit secrets or OS environment variables.
    """
    api_id = None
    api_hash = None
    bot_token = None

    try:
        import streamlit as st
        api_id = st.secrets.get("TG_API_ID")
        api_hash = st.secrets.get("TG_API_HASH")
        bot_token = st.secrets.get("TG_BOT_TOKEN")
    except Exception:
        pass

    # Fallback to standard environment variables
    if not api_id:
        api_id = os.getenv("TG_API_ID")
    if not api_hash:
        api_hash = os.getenv("TG_API_HASH")
    if not bot_token:
        bot_token = os.getenv("TG_BOT_TOKEN")

    if not api_id or not api_hash or not bot_token:
        raise ValueError(
            "Telegram credentials missing. Set TG_API_ID, TG_API_HASH, and TG_BOT_TOKEN in .streamlit/secrets.toml"
        )

    return int(api_id), str(api_hash), str(bot_token)


async def _run_bot_async(api_id: int, api_hash: str, bot_token: str):
    """
    Initializes the Pyrofork client, registers smart plugin handlers,
    and runs the infinite event wait loop without touching POSIX signals.
    """
    global bot_client, _stop_event, _is_running

    _stop_event = asyncio.Event()

    # Configure Smart Plugins pointing to bot/handlers
    plugins_config = dict(root="bot.handlers")

    bot_client = Client(
        name="video_engine_bot",
        api_id=api_id,
        api_hash=api_hash,
        bot_token=bot_token,
        in_memory=True,  # Prevent SQLite locks on ephemeral container filesystems
        plugins=plugins_config
    )

    print("[Pyrofork Runner] Starting Telegram Bot client...")
    await bot_client.start()
    me = await bot_client.get_me()
    print(f"[Pyrofork Runner] Bot online as @{me.username} (ID: {me.id})")
    _is_running = True

    try:
        # Software pause: Keeps background thread alive without hijacking main thread OS signals
        await _stop_event.wait()
    finally:
        print("[Pyrofork Runner] Stopping Telegram Bot client...")
        _is_running = False
        await bot_client.stop()
        print("[Pyrofork Runner] Telegram Bot stopped.")


def _bot_thread_worker(api_id: int, api_hash: str, bot_token: str):
    """
    Worker function executed in the secondary background thread.
    Allocates and sets a dedicated asyncio event loop.
    """
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    try:
        _loop.run_until_complete(_run_bot_async(api_id, api_hash, bot_token))
    except Exception as e:
        print(f"[Pyrofork Runner] Background bot encountered fatal error: {e}")
    finally:
        _loop.close()


def start_bot() -> bool:
    """
    Idempotent method called by app.py to boot the Telegram bot in the background.
    Safe to call across Streamlit page reruns.
    """
    global _bot_thread, _is_running

    with _lock:
        if _is_running or (_bot_thread is not None and _bot_thread.is_alive()):
            return True

        try:
            api_id, api_hash, bot_token = get_telegram_credentials()
        except Exception as e:
            print(f"[Pyrofork Runner] Skipping bot boot: {e}")
            return False

        _bot_thread = threading.Thread(
            target=_bot_thread_worker,
            args=(api_id, api_hash, bot_token),
            name="PyroforkBotThread",
            daemon=True
        )
        _bot_thread.start()
        print("[Pyrofork Runner] Daemon thread spawned successfully.")
        return True


def stop_bot():
    """
    Gracefully triggers the asyncio event flag to shut down the client.
    """
    global _stop_event, _loop
    if _stop_event and _loop and _loop.is_running():
        _loop.call_soon_threadsafe(_stop_event.set)


def is_bot_active() -> bool:
    """Returns current active status of the Telegram bot."""
    return _is_running and (_bot_thread is not None and _bot_thread.is_alive())
