"""End-to-end bot test. Run: python3 tests/test_bot.py

Real v3 API in a uvicorn thread, a stub standing in for Foundry and for
Meta's Graph API, and the bot driven through FastAPI's TestClient with
HMAC-signed webhook payloads.
"""

import hashlib
import hmac
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "calorie-snap-api-v3"))
TMP = tempfile.mkdtemp()
SENT = []
SECRET = "app-secret"


class Stub(BaseHTTPRequestHandler):
    """Foundry (vision + picker) and Graph (media + messages)."""

    def do_GET(self):  # pylint: disable=invalid-name
        assert self.headers["authorization"] == "Bearer wa-token"
        if self.path.endswith("/MEDIA1"):
            port = self.server.server_port
            return self._json({"url": f"http://127.0.0.1:{port}/cdn/x.jpg",
                               "mime_type": "image/jpeg"})
        if self.path == "/cdn/x.jpg":
            self.send_response(200)
            self.end_headers()
            return self.wfile.write(b"\xff\xd8fakejpeg")
        return self._json({}, 404)

    def do_POST(self):  # pylint: disable=invalid-name
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if self.path.endswith("/PHONE1/messages"):
            if body.get("type") == "text":
                SENT.append(body)
            return self._json({"messages": [{"id": "out"}]})
        if body["tools"][0]["name"] == "choose_matches":
            return self._json({"content": [{"type": "tool_use", "input": {
                "choices": []}}], "usage": {}})
        return self._json({"content": [{"type": "tool_use", "input": {
            "items": [
                {"dish": "Naan", "portion_g": 150, "confidence": 0.9},
                {"dish": "Poha", "portion_g": 200, "confidence": 0.9},
                {"dish": "Jalebi", "portion_g": 100, "confidence": 0.9}]}}],
            "usage": {"input_tokens": 900, "output_tokens": 100}})

    def _json(self, payload, code=200):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    stub = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{stub.server_port}"
    os.environ.update({
        "CUISINE_DB": f"{TMP}/c.db", "TELEMETRY_DB": f"{TMP}/t.db",
        "CONFIG_PATH": "/none", "FOUNDRY_ENDPOINT": base,
        "FOUNDRY_API_KEY": "k",
        "WA_CONFIG_PATH": "/none", "WA_DB": f"{TMP}/bot.db",
        "WA_TOKEN": "wa-token", "WA_PHONE_NUMBER_ID": "PHONE1",
        "WA_APP_SECRET": SECRET, "WA_VERIFY_TOKEN": "verify-me",
        "WA_GRAPH_URL": base, "CS_API_URL": "http://127.0.0.1:8098"})
    db = sqlite3.connect(f"{TMP}/c.db")
    db.execute("CREATE TABLE dishes(id INTEGER PRIMARY KEY, indb_code, name,"
               " name_local, region, category, cal_per_100g, cal_per_serving,"
               " serving_unit, protein_g, carb_g, fat_g, fiber_g, data_source,"
               " confidence)")
    for name, c100, serving in [("Naan", 286.4, 152.3), ("Poha", 130, 161),
                                ("Papad", 330, 50), ("Raita", 78, 100)]:
        db.execute("INSERT INTO dishes(name, cal_per_100g, cal_per_serving) "
                   "VALUES(?,?,?)", (name, c100, serving))
    db.commit()
    db.close()

    import uvicorn
    import main as api_main
    server = uvicorn.Server(uvicorn.Config(api_main.app, port=8098,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(1.5)

    from fastapi.testclient import TestClient
    import bot
    client = TestClient(bot.app)

    def post(message, sign=True):
        raw = json.dumps({"entry": [{"changes": [{"value": {
            "messages": [message]}}]}]}).encode()
        sig = "sha256=" + hmac.new(SECRET.encode(), raw,
                                   hashlib.sha256).hexdigest()
        headers = {"x-hub-signature-256": sig if sign else "sha256=bad",
                   "content-type": "application/json"}
        return client.post("/webhook", content=raw, headers=headers)

    def text(body, msg_id):
        return {"from": "15551230000", "id": msg_id, "type": "text",
                "text": {"body": body}}

    # Handshake and security
    ok = client.get("/webhook?hub.mode=subscribe&hub.verify_token=verify-me"
                    "&hub.challenge=4242")
    assert ok.status_code == 200 and ok.text == "4242"
    assert client.get("/webhook?hub.mode=subscribe&hub.verify_token=x"
                      "&hub.challenge=1").status_code == 403
    assert post(text("hi", "m0"), sign=False).status_code == 401
    assert not SENT

    # Help, and text before any photo
    post(text("hi", "m1"))
    assert "Send a photo" in SENT[-1]["text"]["body"]
    post(text("2 papad", "m2"))
    assert "photo of your plate first" in SENT[-1]["text"]["body"]

    # Photo -> breakdown. Meta's retry of the same id is ignored.
    photo = {"from": "15551230000", "id": "m3", "type": "image",
             "image": {"id": "MEDIA1"}}
    post(photo)
    post(photo)
    assert len(SENT) == 3, len(SENT)
    body = SENT[-1]["text"]["body"]
    print(body, "\n---")
    assert body.startswith("*Your plate: 690+ kcal*")
    assert "1. Naan, 150 g: 430" in body and "2. Poha, 200 g: 260" in body
    assert "3. Jalebi, 100 g: not in the database yet" in body
    assert SENT[-1]["context"] == {"message_id": "m3"}

    # The papad fix, a removal, and an addition in one message
    post(text("1 papad 15g, remove 3\nadd raita 80g", "m4"))
    body = SENT[-1]["text"]["body"]
    print(body, "\n---")
    assert body.startswith("*Updated: 372 kcal*"), body
    assert "1. Papad, 15 g: 50" in body and "Jalebi" not in body
    assert "4. Raita, 80 g: 62" in body

    post(text("9 dosa", "m5"))
    assert "no dish 9" in SENT[-1]["text"]["body"]
    post(text("ok", "m6"))
    assert SENT[-1]["text"]["body"] == \
        "Logged. *Today: 372 kcal* across 1 meal."
    post(text("today", "m7"))
    assert "372" in SENT[-1]["text"]["body"]

    # The corrections reached the flywheel, tagged by channel
    tele = sqlite3.connect(f"{TMP}/t.db")
    assert tele.execute("SELECT source FROM events").fetchone()[0] == "whatsapp"
    verdicts = sorted(r[0] for r in tele.execute(
        "SELECT verdict FROM items WHERE verdict IS NOT NULL"))
    assert verdicts == ["correct", "extra", "missing", "wrong_dish"], verdicts
    # ... and the next photo's prompt already knows about naan vs papad
    import telemetry
    assert telemetry.confusion_hints()[0][:2] == ("naan", "papad")
    # No phone number is stored anywhere in the bot DB
    dump = "".join(str(r) for t in ("seen", "sessions", "meals")
                   for r in sqlite3.connect(f"{TMP}/bot.db").execute(
                       f"SELECT * FROM {t}"))
    assert "15551230000" not in dump

    # Pilot allowlist
    os.environ["WA_ALLOWED_NUMBERS"] = "+19990001111"
    post(text("hi", "m8"))
    assert "private pilot" in SENT[-1]["text"]["body"]
    print("ALL PASS")


if __name__ == "__main__":
    main()
