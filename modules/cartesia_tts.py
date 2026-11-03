import asyncio
import base64
import json
import os
import time
import uuid
import wave
import requests
import websockets

MODEL_ID = "sonic-3.6"
DEFAULT_VOICE_NAME = "Shreeraj_HindiProfile"

# Available voice profiles (Shreeraj set as first entry & default)
VOICE_LIST = [
    # Default Voice
    {"name": "Shreeraj_HindiProfile", "gender": "Male", "id": "c7be5dde-c1e6-4ebe-9096-ddc4b4edb1cc"},
    
    # Alternative Voices
    {"name": "Vijay_Native", "gender": "Male", "id": "374b80da-e622-4dfc-90f6-1eeb13d331c9"},
    {"name": "Latha_Native", "gender": "Female", "id": "b426013c-002b-4e89-8874-8cd20b68373a"},
    {"name": "Janani_TamilProfile", "gender": "Female", "id": "fb7d8d97-9730-4165-bd79-36b5ce61b5f2"},
    {"name": "Sindhu_TeluguProfile", "gender": "Female", "id": "07bc462a-c644-49f1-baf7-82d5599131be"},
    {"name": "Kriti_HindiProfile", "gender": "Female", "id": "5283efe8-07d1-4e3a-b615-2ae4a81c1b73"},
    {"name": "Suresh_MarathiProfile", "gender": "Male", "id": "f227bc18-3704-47fe-b759-8c78a450fdfa"},
    {"name": "Aniket_OdiaProfile", "gender": "Male", "id": "701b0221-b708-4e4e-9905-233976f664bf"}
]


def get_fresh_token() -> str | None:
    """Fetches public temporary session token from Cartesia."""
    url = "https://backend.cartesia.ai/access-token/public"
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36",
        "Referer": "https://cartesia.ai/languages/malayalam"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("token") or data.get("access_token")
    except Exception as e:
        print(f"Failed to fetch Cartesia token: {e}")
        return None


def get_voice_by_name(name: str = DEFAULT_VOICE_NAME) -> dict:
    """Matches selected name against VOICE_LIST, falling back to Shreeraj."""
    for voice in VOICE_LIST:
        if voice["name"].lower() == name.lower():
            return voice
    return VOICE_LIST[0]


async def generate_speech(
    text: str,
    output_wav_path: str,
    voice_name: str = DEFAULT_VOICE_NAME,
    language: str = "ml"
) -> float:
    """
    Synthesizes Malayalam audio with quote encapsulation, writes 44.1kHz WAV,
    and returns exact float duration in seconds.
    """
    token = get_fresh_token()
    if not token:
        raise RuntimeError("Cartesia session token acquisition failed.")

    voice = get_voice_by_name(voice_name)

    # Encapsulate strictly in quotes to prevent abrupt sentence cut-offs
    clean_text = text.strip().strip('"').strip("'")
    formatted_transcript = f'"{clean_text}"'

    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)

    ws_url = f"wss://api.cartesia.ai/tts/websocket?cartesia_version=2024-06-10&api_key={token}"
    payload = {
        "context_id": str(uuid.uuid4()),
        "model_id": MODEL_ID,
        "transcript": formatted_transcript,
        "language": language,
        "voice": {"mode": "id", "id": voice["id"]},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 44100
        }
    }

    audio_buffer = bytearray()
    async with websockets.connect(ws_url, ping_interval=10, ping_timeout=15) as ws:
        await ws.send(json.dumps(payload))

        while True:
            resp = json.loads(await ws.recv())
            msg_type = resp.get("type")

            if msg_type == "chunk":
                raw_chunk = base64.b64decode(resp["data"])
                audio_buffer.extend(raw_chunk)
            elif msg_type == "done":
                break
            elif msg_type == "error":
                raise RuntimeError(f"Cartesia error: {resp.get('error')}")

    # Write PCM data into standard WAV file
    sample_rate = 44100
    num_channels = 1
    sample_width = 2  # 16-bit PCM = 2 bytes

    with wave.open(output_wav_path, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_buffer)

    # Compute exact float duration directly from byte count
    total_frames = len(audio_buffer) / (num_channels * sample_width)
    duration_sec = total_frames / float(sample_rate)

    print(f"TTS complete: '{output_wav_path}' | Voice: {voice['name']} | Duration: {duration_sec:.2f}s")
    return duration_sec
