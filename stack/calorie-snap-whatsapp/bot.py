"""CalorieSnap WhatsApp bot.

A thin webhook over the v3 API. A user sends a food photo to the WhatsApp
number and gets the per-dish breakdown back as a reply. Text replies
correct the result and feed the same correction flywheel as the web UI.

Endpoints:
    GET  /webhook   Meta verification handshake
    POST /webhook   inbound messages (HMAC verified, deduplicated)
    GET  /health    configuration state

Meta requires a 200 within seconds, so work runs in a background task.
"""

import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

import replies
import settings
import store
import wa

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("caloriesnap.whatsapp")

app = FastAPI(title="CalorieSnap WhatsApp bot", version="1.0.0")


def _api(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """POSTs JSON to the CalorieSnap API. Returns JSON even on 4xx/5xx."""
    base = settings.cfg("CS_API_URL", "http://calorie-snap-api-v3:8021")
    request = urllib.request.Request(
        f"{base.rstrip('/')}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            return json.loads(err.read().decode("utf-8"))
        except ValueError:
            return {"error": {"stage": "api", "message": f"HTTP {err.code}"}}
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        return {"error": {"stage": "api", "message": str(err)}}


def _allowed(wa_id: str) -> bool:
    """Pilot allowlist. Empty means everyone."""
    raw = settings.cfg("WA_ALLOWED_NUMBERS")
    allowed = {n.strip().lstrip("+") for n in raw.split(",") if n.strip()}
    return not allowed or wa_id.lstrip("+") in allowed


@app.get("/health")
def health() -> Dict[str, Any]:
    """Configuration state. Never returns secrets."""
    return {
        "ok": True, "version": app.version,
        "token_set": bool(settings.cfg("WA_TOKEN")),
        "phone_number_id_set": bool(settings.cfg("WA_PHONE_NUMBER_ID")),
        "signature_check": bool(settings.cfg("WA_APP_SECRET")),
        "verify_token_set": bool(settings.cfg("WA_VERIFY_TOKEN")),
        "api_url": settings.cfg("CS_API_URL",
                                "http://calorie-snap-api-v3:8021"),
    }


@app.get("/webhook", response_class=PlainTextResponse)
def verify(request: Request) -> str:
    """Meta's one-time subscription handshake."""
    params = request.query_params
    expected = settings.cfg("WA_VERIFY_TOKEN")
    if (params.get("hub.mode") == "subscribe" and expected
            and params.get("hub.verify_token") == expected):
        return params.get("hub.challenge", "")
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def webhook(request: Request,
                  tasks: BackgroundTasks) -> Dict[str, bool]:
    """Receives events. Verifies, deduplicates, then works in background."""
    raw = await request.body()
    if not wa.valid_signature(raw,
                              request.headers.get("x-hub-signature-256")):
        raise HTTPException(status_code=401, detail="Bad signature")
    try:
        payload = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="Bad JSON")
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for message in (change.get("value") or {}).get("messages", []):
                if message.get("id") and store.first_time(message["id"]):
                    tasks.add_task(handle_message, message)
    return {"received": True}


def handle_message(message: Dict[str, Any]) -> None:
    """Processes one inbound message. Never raises."""
    sender = message.get("from", "")
    try:
        if not _allowed(sender):
            wa.send_text(sender, "CalorieSnap is in a private pilot. "
                                 "Ask Nitin for an invite.")
            return
        kind = message.get("type")
        if kind == "image":
            _handle_photo(sender, message)
        elif kind == "text":
            _handle_text(sender, message["text"].get("body", ""),
                         message.get("id"))
        else:
            wa.send_text(sender, "Send a photo of your plate, or *help*.")
    except wa.WhatsAppError as err:
        log.error("WhatsApp send or media failure: %s", err)
    except Exception:  # pylint: disable=broad-except
        log.exception("Unhandled failure in handle_message")
        try:
            wa.send_text(sender, "Something went wrong on my side. "
                                 "Send the photo again in a minute.")
        except wa.WhatsAppError:
            pass


def _handle_photo(sender: str, message: Dict[str, Any]) -> None:
    key = store.user_key(sender)
    limit = settings.cfg_int("WA_DAILY_PHOTO_LIMIT", 30)
    if store.today(key)["meals"] >= limit:
        wa.send_text(sender, f"You have reached today's limit of {limit} "
                             "photos. It resets at midnight.")
        return
    wa.mark_read(message["id"])
    content, mime = wa.download_media(message["image"]["id"])
    result = _api("/api/snap", {
        "image_b64": base64.b64encode(content).decode("ascii"),
        "media_type": mime if mime.startswith("image/") else "image/jpeg",
        "source": "whatsapp"})
    if result.get("error") or "items" not in result:
        log.error("Snap failed: %s", result.get("error") or result)
        wa.send_text(sender, "I could not read that photo. Send it again "
                             "in a minute.", reply_to=message["id"])
        return
    items = [{
        "id": i["id"], "name": i["dish"], "grams": i.get("portion_g"),
        "cal": i.get("cal"), "suspect": bool(i.get("suspect")),
        "approx": bool(i.get("approx")), "rated": False, "removed": False,
    } for i in result["items"]]
    store.save_session(key, result["event_id"], items)
    if items:
        store.save_meal(key, result["event_id"], result["total_calories"],
                        replies.uncounted(items) > 0)
    wa.send_text(sender, replies.meal_summary(
        items, result["total_calories"]), reply_to=message["id"])


def _handle_text(sender: str, text: str, msg_id: Optional[str]) -> None:
    key = store.user_key(sender)
    commands = replies.parse(text)
    kinds = {c[0] for c in commands}
    if "help" in kinds:
        wa.send_text(sender, replies.HELP_TEXT)
        return
    if kinds == {"today"}:
        wa.send_text(sender, replies.today_summary(store.today(key)))
        return
    session = store.load_session(key)
    if kinds == {"unknown"} or not session:
        wa.send_text(sender, "Send a photo of your plate first, or reply "
                             "*help* to see what I understand.")
        return

    items: List[Dict[str, Any]] = session["items"]
    event_id = session["event_id"]
    total: Optional[float] = None
    notes: List[str] = []
    for command in commands:
        outcome = _apply(command, event_id, items)
        if isinstance(outcome, str):
            notes.append(outcome)
        elif outcome is not None:
            total = outcome
    store.save_session(key, event_id, items)
    if total is not None:
        store.save_meal(key, event_id, total, replies.uncounted(items) > 0)
    if kinds == {"confirm"}:
        wa.send_text(sender, "Logged. " + replies.today_summary(
            store.today(key)))
        return
    reply = "\n".join(notes)
    if total is not None:
        reply = (reply + "\n\n" if reply else "") + replies.meal_summary(
            items, total, updated=True)
    wa.send_text(sender, reply or "Nothing changed.", reply_to=msg_id)


def _apply(command: tuple, event_id: int, items: List[Dict[str, Any]]):
    """Applies one correction through the API.

    Returns the new meal total, a user-facing note (str), or None.
    """
    kind = command[0]
    if kind in ("unknown", "today", "help"):
        return None
    if kind == "confirm":
        total = None
        for item in items:
            if item["rated"] or item["removed"] or not item.get("id"):
                continue
            out = _api("/api/feedback", {
                "event_id": event_id, "item_id": item["id"],
                "verdict": "correct"})
            item["rated"] = True
            total = out.get("total_calories", total)
        return total
    if kind == "add":
        _, name, grams = command
        out = _api("/api/feedback", {
            "event_id": event_id, "verdict": "missing",
            "corrected_dish": name, "corrected_portion_g": grams})
        if "total_calories" not in out:
            return f"Could not add {name}."
        items.append({"id": out.get("item_id"), "name": name,
                      "grams": grams, "cal": out.get("cal"),
                      "suspect": False, "approx": False, "rated": True,
                      "removed": False})
        return out["total_calories"]

    number = command[1]
    if not 1 <= number <= len(items) or items[number - 1]["removed"]:
        return f"There is no dish {number} on this plate."
    item = items[number - 1]
    body: Dict[str, Any] = {"event_id": event_id, "item_id": item["id"]}
    if kind == "remove":
        body["verdict"] = "extra"
    elif kind == "portion":
        body.update(verdict="wrong_portion", corrected_portion_g=command[2])
    else:  # dish
        body.update(verdict="wrong_dish", corrected_dish=command[2],
                    corrected_portion_g=command[3])
    out = _api("/api/feedback", body)
    if "total_calories" not in out:
        return f"Could not update dish {number}."
    item["rated"] = True
    if kind == "remove":
        item["removed"] = True
    else:
        if kind == "dish":
            item["name"] = command[2]
        if body.get("corrected_portion_g"):
            item["grams"] = body["corrected_portion_g"]
        item.update(cal=out.get("cal"), suspect=False, approx=False)
    return out["total_calories"]
