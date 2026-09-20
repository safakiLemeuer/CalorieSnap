"""End-to-end tests. Run: python3 tests/test_api.py

Uses a temp INDB, a temp telemetry DB, and a local stub HTTP server that
stands in for both Foundry and the FastVLM container, so the real urllib
code path is exercised.
"""

import base64
import json
import os
import sqlite3
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
TMP = tempfile.mkdtemp()
STATE = {"claude_items": [], "mode": "tool", "requests": [], "picks": {},
         "picker_calls": 0}


class Stub(BaseHTTPRequestHandler):
    """Fake Foundry (Claude + OpenAI routes) and fake FastVLM."""

    def do_POST(self):  # pylint: disable=invalid-name
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        STATE["requests"].append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if (self.path == "/anthropic/v1/messages"
                and body["tools"][0]["name"] == "choose_matches"):
            STATE["picker_calls"] += 1
            pending = json.loads(body["messages"][0]["content"])
            choices = []
            for entry in pending:
                want = STATE["picks"].get(entry["dish"])
                quality = "same"
                if isinstance(want, tuple):
                    want, quality = want
                idx = (entry["candidates"].index(want)
                       if want in entry["candidates"] else None)
                choices.append({"item": entry["item"], "candidate": idx,
                                "quality": quality})
            return self._send(200, {"content": [{
                "type": "tool_use", "name": "choose_matches",
                "input": {"choices": choices}}],
                "usage": {"input_tokens": 200, "output_tokens": 30}})
        if self.path == "/anthropic/v1/messages":
            STATE["vision_request"] = body
            if STATE["mode"] == "429":
                return self._send(429, {"error": "rate limit"})
            content = ([{"type": "tool_use", "name": "report_dishes",
                         "input": {"items": STATE["claude_items"]}}]
                       if STATE["mode"] == "tool"
                       else [{"type": "text", "text": "Too dark to tell."}])
            return self._send(200, {"content": content, "usage": {
                "input_tokens": 100, "cache_read_input_tokens": 800,
                "output_tokens": 120}})
        if self.path == "/openai/v1/chat/completions":
            return self._send(200, {"choices": [{"message": {"content": json.dumps(
                {"items": [{"dish": "Idli", "portion_g": 120, "confidence": 0.8}]})}}],
                "usage": {"prompt_tokens": 700, "completion_tokens": 40}})
        if self.path == "/describe":
            return self._send(200, {"text": "Answer: poha, 120 calories Answer: poha"})
        return self._send(404, {})

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    server = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    os.environ.update({
        "CUISINE_DB": f"{TMP}/c.db", "TELEMETRY_DB": f"{TMP}/t.db",
        "CONFIG_PATH": f"{TMP}/none.json", "FOUNDRY_ENDPOINT": base,
        "FOUNDRY_API_KEY": "k", "FOUNDRY_MODELS": "gpt-mini",
        "FASTVLM_URL": f"{base}/describe"})
    db = sqlite3.connect(f"{TMP}/c.db")
    db.execute("CREATE TABLE dishes(id INTEGER PRIMARY KEY, indb_code, name,"
               " name_local, region, category, cal_per_100g, cal_per_serving,"
               " serving_unit, protein_g, carb_g, fat_g, fiber_g, data_source,"
               " confidence)")
    for name, c100, serving in [("Poha", 130, 161), ("Poha cutlet", 210, 180),
                                ("Dal makhani", 140, 340), ("Chapati", 300, 104),
                                ("Plain rice", 130, 200), ("Idli", 135, 80),
                                ("Boiled rice (Uble chawal)", 117.2, 351.6),
                                ("Paneer shaslik/tikka", 93.8, 411.7),
                                ("Paneer pulao", 581.9, 4875.8),
                                ("Paneer curry", 176.5, 357.1),
                                ("Gulab Jamun with khoya", 586.1, 918.5),
                                ("Naan", 286.4, 152.3),
                                ("Tandoori chicken", 145.0, 1450.0)]:
        db.execute("INSERT INTO dishes(name, cal_per_100g, cal_per_serving) "
                   "VALUES(?,?,?)", (name, c100, serving))
    for word in ["Spanish", "Fried", "Lemon", "Tomato", "Coconut", "Mint",
                 "Carrot", "Peas", "Corn", "Garlic", "Ginger", "Onion",
                 "Egg", "Soya", "Methi", "Beet", "Cabbage", "Capsicum",
                 "Mushroom", "Cashew"]:
        db.execute("INSERT INTO dishes(name, cal_per_100g, cal_per_serving)"
                   " VALUES(?,?,?)", (f"{word} rice", 170, 500))
    db.commit()
    db.close()

    from fastapi.testclient import TestClient
    import main as app_main
    client = TestClient(app_main.app)
    img = base64.b64encode(b"fakeimage").decode()

    assert client.get("/").status_code == 200
    health = client.get("/api/health").json()
    assert health["models"] == ["claude-haiku-4-5", "gpt-mini", "fastvlm-local"], health
    assert health["indb_dishes"] == 33

    # Claude, forced tool use
    STATE["picks"] = {"Kanda Poha": "Poha"}
    STATE["claude_items"] = [
        {"dish": "Kanda Poha", "name_local": "पोहा", "portion_g": 200, "confidence": 0.9},
        {"dish": "Chapati", "portion_g": 40, "confidence": 0.6},
        {"dish": "Neer Dosa", "portion_g": 80, "confidence": 0.4}]
    res = client.post("/api/snap", json={"image_b64": img}).json()
    headers = STATE["requests"][-1][1]
    body = STATE["vision_request"]
    assert headers["x-api-key"] == "k" and body["temperature"] == 0
    assert body["tool_choice"] == {"type": "tool", "name": "report_dishes"}
    assert body["messages"][0]["content"][0]["source"]["data"] == img
    items = res["items"]
    # "Kanda Poha" is not an exact name: the picker chooses among candidates
    assert [i["match_method"] for i in items] == ["picker", "exact", None]
    assert items[0]["matched_name"] == "Poha"
    assert res["total_calories"] == 380.0
    event = res["event_id"]

    # Feedback flywheel
    post = lambda **kw: client.post("/api/feedback", json={"event_id": event, **kw}).json()
    assert post(item_id=items[0]["id"], verdict="correct")["total_calories"] == 380.0
    assert post(item_id=items[1]["id"], verdict="wrong_portion", corrected_portion_g=80)["cal"] == 240.0
    assert post(item_id=items[2]["id"], verdict="wrong_dish", corrected_dish="Plain rice")["cal"] == 104.0
    out = post(verdict="missing", corrected_dish="Dal makhani", corrected_portion_g=100)
    assert out["cal"] == 140.0 and out["total_calories"] == 744.0
    assert client.post("/api/feedback", json={"event_id": event, "verdict": "nope"}).status_code == 400
    assert client.post("/api/feedback", json={"event_id": event, "item_id": 999, "verdict": "correct"}).status_code == 404

    # Same photo, different answer -> instability is measured
    STATE["claude_items"] = [{"dish": "Poha", "portion_g": 150, "confidence": 0.95}]
    assert client.post("/api/snap", json={"image_b64": img}).json()["items"][0]["match_method"] == "exact"

    # Failure classes are logged with their stage
    STATE["mode"] = "text"
    res = client.post("/api/snap", json={"image_b64": img})
    assert res.status_code == 502 and res.json()["error"]["stage"] == "llm_parse"
    STATE["mode"] = "429"
    assert client.post("/api/snap", json={"image_b64": img}).json()["error"]["stage"] == "llm_http"
    STATE["mode"] = "tool"
    assert client.post("/api/snap", json={"image_b64": "!!"}).status_code == 400
    assert client.post("/api/snap", json={"image_b64": img, "model": "x"}).status_code == 400

    # OpenAI-compatible route and air-gapped FastVLM route
    res = client.post("/api/snap", json={"image_b64": img, "model": "gpt-mini"}).json()
    assert res["items"][0]["cal"] == 162.0 and STATE["requests"][-1][1]["api-key"] == "k"
    res = client.post("/api/snap", json={"image_b64": img, "model": "fastvlm-local"}).json()
    assert res["items"][0]["dish"] == "Poha" and res["items"][0]["cal"] == 161.0, res

    # Admin analytics
    summary = client.get("/api/admin/summary?days=7").json()
    kpis = summary["kpis"]
    assert kpis["snaps"] == 6 and kpis["errors"] == 2 and kpis["accuracy"] == 0.667
    assert kpis["avg_input_tokens"] > 0 and kpis["missing_items"] == 1
    assert summary["consistency"] == {"repeated_photos": 1, "unstable": 1}
    assert summary["db_misses"][0]["name"] == "neer dosa"
    assert summary["confusions"][0]["name"] == "Neer Dosa -> Plain rice"
    assert {m["model"] for m in summary["by_model"]} == {"claude-haiku-4-5", "gpt-mini", "fastvlm-local"}
    assert len(client.get("/api/admin/events?kind=errors").json()) == 2
    assert client.get("/api/admin/events?kind=bogus").status_code == 400
    assert len(client.get("/api/history").json()) == 4
    assert client.get("/api/dishes?q=po").json()[0] == "Poha"
    assert client.get("/api/admin/export.csv").text.count("\n") == 5

    # --- v3.2: matching against real INDB naming ---
    STATE["picks"] = {"Basmati Rice": "Boiled rice (Uble chawal)",
                      "Paneer Tikka": "Paneer shaslik/tikka",
                      "Gulab Jamun": "Gulab Jamun with khoya",
                      "Paneer Tikka Masala": ("Paneer curry", "close"),
                      "Tandoori Chicken": "Tandoori chicken"}
    STATE["claude_items"] = [
        {"dish": "Basmati Rice", "portion_g": 200, "confidence": 0.9},
        {"dish": "Paneer Tikka", "portion_g": 100, "confidence": 0.9},
        {"dish": "Paneer Tikka Masala", "portion_g": 250, "confidence": 0.9},
        {"dish": "Gulab Jamun", "portion_g": 150, "confidence": 0.9},
        {"dish": "Jalebi", "portion_g": 180, "confidence": 0.9},
        {"dish": "Tandoori Chicken", "portion_g": 120, "confidence": 0.9}]
    img2 = base64.b64encode(b"thali").decode()
    calls = STATE["picker_calls"]
    res = client.post("/api/snap", json={"image_b64": img2}).json()
    got = {i["dish"]: i for i in res["items"]}
    assert STATE["picker_calls"] == calls + 1  # one batched call per snap
    assert got["Basmati Rice"]["matched_name"] == "Boiled rice (Uble chawal)"
    assert got["Basmati Rice"]["cal"] == 234.4
    assert got["Paneer Tikka"]["matched_name"] == "Paneer shaslik/tikka"
    # nearest equivalent is graded "close" and shown as approximate
    assert got["Paneer Tikka Masala"]["matched_name"] == "Paneer curry"
    assert got["Paneer Tikka Masala"]["match_method"] == "picker_close"
    assert got["Paneer Tikka Masala"]["approx"] is True
    assert got["Basmati Rice"]["approx"] is False
    # big legitimate serving + sane per-100g + known portion: not suspect
    assert got["Tandoori Chicken"]["cal"] == 174.0
    assert got["Tandoori Chicken"]["suspect"] is False
    assert got["Jalebi"]["matched_id"] is None
    # implausible INDB row is flagged, not silently trusted
    assert got["Gulab Jamun"]["suspect"] is True
    counted = sum(i["cal"] for i in res["items"]
                  if i["cal"] and not i["suspect"])
    assert res["total_calories"] == round(counted, 1)
    assert got["Gulab Jamun"]["cal"] > 800  # shown, but outside the total
    assert got["Basmati Rice"]["suspect"] is False

    # second snap: decisions come from the alias cache, not the picker
    STATE["picks"] = {}
    res = client.post("/api/snap", json={"image_b64": img2}).json()
    got = {i["dish"]: i for i in res["items"]}
    assert got["Basmati Rice"]["match_method"] == "alias"
    assert got["Paneer Tikka Masala"]["match_method"] == "alias_close"
    assert got["Paneer Tikka Masala"]["approx"] is True
    assert got["Basmati Rice"]["cal"] == 234.4

    # user says the match was wrong -> alias is dropped
    client.post("/api/feedback", json={
        "event_id": res["event_id"], "item_id": got["Paneer Tikka"]["id"],
        "verdict": "wrong_dish", "corrected_dish": "Naan"})
    import telemetry
    assert telemetry.alias_get("Paneer Tikka") is None
    assert telemetry.alias_get("Basmati Rice") is not None

    # --- v3.4: right dish, wrong INDB row ---
    STATE["picks"] = {"Steamed Rice": "Spanish rice"}  # picker gets it wrong
    STATE["claude_items"] = [{"dish": "Steamed Rice", "portion_g": 200,
                              "confidence": 0.9}]
    res = client.post("/api/snap", json={"image_b64": img2}).json()
    bad = res["items"][0]
    assert bad["matched_name"] == "Spanish rice" and bad["cal"] == 340.0
    fixed = client.post("/api/feedback", json={
        "event_id": res["event_id"], "item_id": bad["id"],
        "verdict": "wrong_match",
        "corrected_dish": "Boiled rice (Uble chawal)"}).json()
    assert fixed["cal"] == 234.4, fixed
    assert telemetry.alias_get("Steamed Rice")[1] == "manual"
    assert client.post("/api/feedback", json={
        "event_id": res["event_id"], "item_id": bad["id"],
        "verdict": "wrong_match", "corrected_dish": "nope"}).status_code == 400
    # the pin wins on the next snap, even though the picker would still err
    res = client.post("/api/snap", json={"image_b64": img2}).json()
    assert res["items"][0]["matched_name"] == "Boiled rice (Uble chawal)"
    assert res["items"][0]["match_method"] == "alias"
    # admin pin / clear
    out = client.post("/api/admin/alias", json={
        "dish": "Jeera Rice", "indb_name": "boiled rice (uble chawal)"}).json()
    assert out["alias"] == "Boiled rice (Uble chawal)"
    assert client.post("/api/admin/alias", json={
        "dish": "Jeera Rice", "indb_name": "x"}).status_code == 400
    client.post("/api/admin/alias", json={"dish": "Jeera Rice"})
    assert telemetry.alias_get("Jeera Rice") is None
    # every full-coverage row survives the cut (20 sibling rice rows + 3)
    import indb as indb_mod
    conn = indb_mod.connect()
    assert len(indb_mod.candidates(conn, "Basmati Rice")) >= 22
    conn.close()

    # --- v3.5: corrections steer the next prompt ---
    STATE["picks"] = {}
    STATE["claude_items"] = [{"dish": "Naan", "portion_g": 150,
                              "confidence": 0.95}]
    res = client.post("/api/snap", json={"image_b64": img2}).json()
    before = STATE["vision_request"]["system"][0]["text"]
    assert "Papad: very thin" in before and "said 'naan'" not in before
    client.post("/api/feedback", json={
        "event_id": res["event_id"], "item_id": res["items"][0]["id"],
        "verdict": "wrong_dish", "corrected_dish": "Papad",
        "corrected_portion_g": 15})
    client.post("/api/feedback", json={
        "event_id": res["event_id"], "verdict": "missing",
        "corrected_dish": "Green chutney"})
    client.post("/api/snap", json={"image_b64": img2})
    after = STATE["vision_request"]["system"][0]["text"]
    assert "said 'naan', was actually 'papad' (1x)" in after, after[-400:]
    assert "green chutney" in after

    # picker down -> substring fallback still answers
    STATE["mode2"] = None
    import vision
    real = vision.pick_matches
    def broken(pending):
        raise vision.PipelineError("llm_http", "down")
    vision.pick_matches = broken
    STATE["claude_items"] = [{"dish": "Kanda Poha Special", "portion_g": 100,
                              "confidence": 0.8}]
    res = client.post("/api/snap", json={"image_b64": img2}).json()
    assert res["items"][0]["match_method"] == "contained", res
    vision.pick_matches = real

    summary = client.get("/api/admin/summary").json()
    assert summary["kpis"]["suspect_hits"] == 2
    assert summary["kpis"]["wrong_matches"] == 1
    assert summary["bad_matches"][0]["name"] == (
        "Steamed Rice -> Boiled rice (Uble chawal)")
    # wrong_match is a lookup failure, not a model failure
    assert summary["kpis"]["accuracy"] is not None
    import main as app_mod
    assert summary["by_version"][0]["version"] == app_mod.app.version
    assert summary["by_version"][0]["match_rate"] is not None
    # a generic head noun with 40 sibling rows must still surface the row
    import indb
    conn = indb.connect()
    names = [r["name"] for r in indb.candidates(conn, "Basmati Rice")]
    conn.close()
    assert "Boiled rice (Uble chawal)" in names, names
    explain = client.get("/api/admin/match?dish=Jeera Rice").json()
    assert explain["candidates"] and explain["picker"]["choice"] is None
    assert summary["suspect_rows"][0]["name"] == "Gulab Jamun with khoya"
    assert {"picker", "alias"} <= {m["name"] for m in summary["by_match_method"]}

    # Prompt grounding with cached INDB names
    os.environ["INDB_NAMES_IN_PROMPT"] = "1"
    client.post("/api/snap", json={"image_b64": img})
    system = STATE["vision_request"]["system"]
    assert system[1]["cache_control"] == {"type": "ephemeral"} and "Dal makhani" in system[1]["text"]

    print("ALL PASS")


if __name__ == "__main__":
    main()
