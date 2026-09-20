"""Minimal WhatsApp Cloud API client. Stdlib urllib only.

Docs: https://developers.facebook.com/docs/whatsapp/cloud-api
The Graph version is a setting because Meta retires versions on a schedule.
"""

import hashlib
import hmac
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

import settings

_TIMEOUT_S = 30
_MAX_MEDIA_BYTES = 8_000_000


class WhatsAppError(Exception):
    """A Graph API call failed."""


def _base() -> str:
    host = settings.cfg("WA_GRAPH_URL", "https://graph.facebook.com")
    return f"{host.rstrip('/')}/{settings.cfg('WA_GRAPH_VERSION', 'v24.0')}"


def _request(url: str, body: Optional[Dict[str, Any]] = None) -> bytes:
    """Authenticated GET (no body) or JSON POST."""
    headers = {"authorization": f"Bearer {settings.cfg('WA_TOKEN')}"}
    data = None
    if body is not None:
        headers["content-type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as resp:
            return resp.read(_MAX_MEDIA_BYTES + 1)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:300]
        raise WhatsAppError(f"HTTP {err.code}: {detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise WhatsAppError(f"Network: {err}")


def valid_signature(raw_body: bytes, header: Optional[str]) -> bool:
    """Checks Meta's X-Hub-Signature-256 HMAC over the raw request body."""
    secret = settings.cfg("WA_APP_SECRET")
    if not secret:
        return settings.cfg("WA_ALLOW_UNSIGNED") == "1"
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), raw_body,
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len("sha256="):])


def download_media(media_id: str) -> Tuple[bytes, str]:
    """Fetches an inbound media file.

    Returns:
        (content_bytes, mime_type)
    """
    meta = json.loads(_request(f"{_base()}/{media_id}"))
    url = meta.get("url")
    if not url:
        raise WhatsAppError("Media lookup returned no url")
    content = _request(url)
    if len(content) > _MAX_MEDIA_BYTES:
        raise WhatsAppError("Media too large")
    return content, meta.get("mime_type") or "image/jpeg"


def send_text(to: str, text: str, reply_to: Optional[str] = None) -> None:
    """Sends a text message. Inside the 24 hour window this is free-form."""
    body: Dict[str, Any] = {
        "messaging_product": "whatsapp", "to": to, "type": "text",
        "text": {"preview_url": False, "body": text[:4000]}}
    if reply_to:
        body["context"] = {"message_id": reply_to}
    _request(f"{_base()}/{settings.cfg('WA_PHONE_NUMBER_ID')}/messages",
             body)


def mark_read(message_id: str) -> None:
    """Shows the blue ticks so the user knows the photo arrived."""
    try:
        _request(
            f"{_base()}/{settings.cfg('WA_PHONE_NUMBER_ID')}/messages",
            {"messaging_product": "whatsapp", "status": "read",
             "message_id": message_id})
    except WhatsAppError:
        pass  # Cosmetic. Never fail a snap on it.
