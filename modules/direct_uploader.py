import json
import os
import shutil
import subprocess
import requests


def upload_for_direct_url(file_path: str) -> tuple[str, str]:
    """
    Uploads a rendered video and returns (download_url, provider_label).
    Uses curl subprocess when available to avoid loading large MP4s into RAM.
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
                    link = payload["data"]["downloadPage"]
                    return link, "Gofile"
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
