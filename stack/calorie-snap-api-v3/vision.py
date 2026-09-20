"""Dish identification. Cloud (Microsoft Foundry) and air-gapped paths.

identify() is the single entry point. It routes by model name:

    claude*        Anthropic Messages route on Foundry, forced tool use so
                   the reply is schema-valid JSON by construction.
    fastvlm-local  The on-prem FastVLM container. Single dish, no portion.
    anything else  OpenAI-compatible chat route on Foundry, JSON mode.

All HTTP is stdlib urllib. The model never supplies calories.
"""

import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import config
import indb
import telemetry
from errors import PipelineError

_HTTP_TIMEOUT_S = 45
_MAX_OUTPUT_TOKENS = 700
_USER_TEXT = "Identify all dishes."

_SYSTEM_PROMPT = (
    "You identify Indian food in photos. List every distinct dish on the "
    "plate, including breads, rice, chutneys, pickles, and sides. Use "
    "common dish names as found in Indian nutrition databases. Estimate "
    "each portion in grams from visual cues such as plate and bowl size. "
    "Skip garnishes and sprinkles under about 10 grams, such as herbs, "
    "seeds, or a lemon wedge. If the food is not Indian, name it "
    "plainly. "
    "If unsure, give your best guess with lower confidence. Never "
    "estimate calories."
)
# Visual tells for dishes that look alike in a photo. Kept short on
# purpose: the learned hints below carry the long tail.
_LOOKALIKES = (
    "Lookalikes, check the visual tells before naming:\n"
    "- Papad: very thin, crisp, brittle, blistered with small bubbles, "
    "pale or speckled, often a large disc resting on top of other food. "
    "Weighs 10 to 20 g. Naan: thick, soft, leavened, teardrop shape, "
    "charred puffed patches, often glossy with butter. Roti or chapati: "
    "thin, soft, matte, whole-wheat brown, folded. Paratha: layered, "
    "shiny with oil.\n"
    "- Dal makhani is dark brown and creamy. Rajma shows whole kidney "
    "beans. Chana masala shows whole round chickpeas.\n"
    "- Poha is flat flaked rice, yellow. Upma is grainy semolina, pale.\n"
    "- Every distinct item counts, including anything lying on top of or "
    "partly hidden behind another dish."
)


def _learned_hints() -> str:
    """Prompt lines built from user corrections. Empty when none exist."""
    try:
        confusions = telemetry.confusion_hints()
        missed = telemetry.missed_hints()
    except Exception:  # pylint: disable=broad-except
        return ""  # Hints are an optimization. Never fail a snap on them.
    lines = []
    if confusions:
        lines.append(
            "Users corrected these past mistakes. When you are about to "
            "name the first dish, look again for the second:")
        lines += [f"- said '{pred}', was actually '{actual}' ({n}x)"
                  for pred, actual, n in confusions]
    if missed:
        lines.append(
            "Users often had to add these by hand because they were "
            "missed. Look for them: "
            + ", ".join(dish for dish, _ in missed) + ".")
    return "\n".join(lines)


_JSON_SUFFIX = (
    '\nReturn only JSON, no prose: {"items":[{"dish":"<name>",'
    '"name_local":"<local name or null>","portion_g":<int>,'
    '"confidence":<0-1>}]}'
)
_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "dish": {"type": "string",
                     "description": "Canonical English dish name"},
            "name_local": {"type": ["string", "null"]},
            "portion_g": {"type": "integer"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["dish", "portion_g", "confidence"],
    }}},
    "required": ["items"],
}

# Tokens the FastVLM fallback parser must never accept as a dish name.
_GARBAGE = {"and", "it", "has", "approximately", "about", "around", "the",
            "this", "that", "which"}


def identify(model: str, image_b64: str, media_type: str
             ) -> Tuple[List[Dict[str, Any]], str, int, int]:
    """Identifies the dishes in one photo.

    Returns:
        (items, raw_output, input_tokens, output_tokens). Each item has
        dish, name_local, portion_g, confidence.

    Raises:
        PipelineError: with the failing stage.
    """
    if model == config.LOCAL_MODEL:
        return _identify_local(image_b64)
    endpoint = config.cfg("FOUNDRY_ENDPOINT").rstrip("/")
    key = config.cfg("FOUNDRY_API_KEY")
    if not endpoint or not key:
        raise PipelineError(
            "config", "Set FOUNDRY_ENDPOINT and FOUNDRY_API_KEY", 503)
    if model.lower().startswith("claude"):
        return _identify_claude(endpoint, key, model, image_b64, media_type)
    return _identify_openai(endpoint, key, model, image_b64, media_type)


_PICKER_PROMPT = (
    "You match dish names to rows of an Indian nutrition database. For "
    "each item pick one candidate and grade it.\n"
    "quality 'same': the same dish, possibly under another name "
    "('Basmati Rice' is 'Boiled rice (Uble chawal)').\n"
    "quality 'close': no identical row exists, but this candidate is the "
    "nearest nutritional equivalent: same main ingredient, same cooking "
    "method, similar fat and sugar level ('Chicken Tikka Masala' is "
    "close to a chicken curry with gravy).\n"
    "When the item is a generic name ('Raita', 'Dal', 'Rice') and the "
    "candidates are specific variants, pick the plainest, most common "
    "variant and grade it 'same'.\n"
    "Use candidate null when nothing qualifies. Sharing one ingredient is "
    "not enough: 'Paneer Tikka Masala' is not 'Paneer pulao', and a "
    "sweet is never a match for a savory dish."
)
_PICKER_SCHEMA = {
    "type": "object",
    "properties": {"choices": {"type": "array", "items": {
        "type": "object",
        "properties": {"item": {"type": "integer"},
                       "candidate": {"type": ["integer", "null"]},
                       "quality": {"type": "string",
                                   "enum": ["same", "close"]}},
        "required": ["item", "candidate"]}}},
    "required": ["choices"],
}


def picker_available() -> bool:
    """True when a Claude deployment can run the text-only picker."""
    return (config.cloud_configured()
            and picker_model().lower().startswith("claude"))


def picker_model() -> str:
    """Deployment used for matching. Defaults to the main model."""
    return config.cfg("PICKER_MODEL") or config.cfg(
        "FOUNDRY_MODEL", "claude-haiku-4-5")


def pick_matches(pending: List[Dict[str, Any]]
                 ) -> Tuple[Dict[int, Tuple[Optional[int], str]], int, int]:
    """Asks the model which INDB candidate, if any, each dish is.

    Args:
        pending: [{"item": int, "dish": str, "candidates": [str, ...]}]

    Returns:
        ({item: (candidate index or None, 'same' | 'close')},
         input_tokens, output_tokens)
    """
    data = _post_json(
        f"{config.cfg('FOUNDRY_ENDPOINT').rstrip('/')}"
        "/anthropic/v1/messages",
        {"x-api-key": config.cfg("FOUNDRY_API_KEY"),
         "anthropic-version": "2023-06-01"},
        {
            "model": picker_model(),
            "max_tokens": 500,
            "temperature": 0,
            "system": _PICKER_PROMPT,
            "tools": [{"name": "choose_matches",
                       "description": "Report one choice per item.",
                       "input_schema": _PICKER_SCHEMA}],
            "tool_choice": {"type": "tool", "name": "choose_matches"},
            "messages": [{"role": "user", "content": json.dumps(
                pending, ensure_ascii=False)}],
        })
    usage = data.get("usage", {})
    choices: Dict[int, Tuple[Optional[int], str]] = {}
    for block in data.get("content", []):
        if block.get("type") != "tool_use":
            continue
        for choice in (block.get("input") or {}).get("choices", []):
            if isinstance(choice, dict) and isinstance(
                    choice.get("item"), int):
                picked = choice.get("candidate")
                quality = ("close" if choice.get("quality") == "close"
                           else "same")
                choices[choice["item"]] = (
                    picked if isinstance(picked, int) else None, quality)
    return (choices, int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)))


def _vision_prompt() -> str:
    """Base prompt, static lookalike tells, then learned hints."""
    parts = [_SYSTEM_PROMPT, _LOOKALIKES, _learned_hints()]
    return "\n\n".join(part for part in parts if part)


def _system_blocks() -> List[Dict[str, Any]]:
    """System prompt, optionally grounded in INDB names and cached.

    With INDB_NAMES_IN_PROMPT=1 the full canonical name list is appended
    and marked for prompt caching, so repeat calls bill it at the cached
    rate. Names the model picks then match INDB exactly.
    """
    blocks: List[Dict[str, Any]] = [{"type": "text",
                                     "text": _vision_prompt()}]
    if config.cfg("INDB_NAMES_IN_PROMPT") == "1":
        names = "\n".join(indb.all_names())
        blocks.append({
            "type": "text",
            "text": ("Prefer these exact dish names when one fits. Use "
                     "another name only when none does:\n" + names),
            "cache_control": {"type": "ephemeral"}})
    return blocks


def _identify_claude(endpoint: str, key: str, model: str, image_b64: str,
                     media_type: str):
    data = _post_json(
        f"{endpoint}/anthropic/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
        {
            "model": model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "system": _system_blocks(),
            "tools": [{"name": "report_dishes",
                       "description": "Report every dish in the photo.",
                       "input_schema": _ITEMS_SCHEMA}],
            "tool_choice": {"type": "tool", "name": "report_dishes"},
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": image_b64}},
                {"type": "text", "text": _USER_TEXT}]}],
        })
    usage = data.get("usage", {})
    tok_in = (int(usage.get("input_tokens", 0))
              + int(usage.get("cache_read_input_tokens", 0))
              + int(usage.get("cache_creation_input_tokens", 0)))
    tok_out = int(usage.get("output_tokens", 0))
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            payload = block.get("input") or {}
            return (_clean_items(payload.get("items")),
                    json.dumps(payload, ensure_ascii=False),
                    tok_in, tok_out)
    text = "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")
    return parse_json_items(text), text, tok_in, tok_out


def _identify_openai(endpoint: str, key: str, model: str, image_b64: str,
                     media_type: str):
    data = _post_json(
        f"{endpoint}/openai/v1/chat/completions", {"api-key": key},
        {
            "model": model,
            "max_completion_tokens": _MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system",
                 "content": _vision_prompt() + _JSON_SUFFIX},
                {"role": "user", "content": [
                    {"type": "text", "text": _USER_TEXT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}"}},
                ]}],
        })
    choices = data.get("choices") or [{}]
    text = (choices[0].get("message") or {}).get("content") or ""
    usage = data.get("usage", {})
    return (parse_json_items(text), text,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)))


def _identify_local(image_b64: str):
    """Air-gapped path: FastVLM container, single dish.

    Contract (configurable, see README): POST FASTVLM_URL with JSON
    {FASTVLM_IMAGE_FIELD: <base64>}. The reply is JSON. The text is read
    from FASTVLM_TEXT_FIELD, or the first common text key found.
    """
    url = config.cfg("FASTVLM_URL")
    if not url:
        raise PipelineError("config", "Set FASTVLM_URL", 503)
    field = config.cfg("FASTVLM_IMAGE_FIELD", "image_b64")
    try:
        data = _post_json(url, {}, {field: image_b64})
    except PipelineError as err:
        raise PipelineError("local_vlm", err.message, err.status)
    text_field = config.cfg("FASTVLM_TEXT_FIELD")
    keys = [text_field] if text_field else [
        "text", "description", "response", "output", "answer", "result",
        "raw"]
    raw = next((str(data[k]) for k in keys
                if isinstance(data.get(k), str)), "")
    if not raw:
        raise PipelineError(
            "local_vlm", f"No text field in FastVLM reply: {list(data)}")
    dish = parse_fastvlm_answer(raw)
    if not dish:
        raise PipelineError("llm_parse", "No dish name in FastVLM output")
    return ([{"dish": dish, "name_local": None, "portion_g": None,
              "confidence": None}], raw, 0, 0)


def parse_fastvlm_answer(raw: str) -> Optional[str]:
    """Extracts the dish from FastVLM output.

    Primary: the "Answer: X, calories" pattern. FastVLM often loops the
    answer, so only the first occurrence is used. Fallback: the first
    1 to 4 word phrase that contains no garbage words.
    """
    match = re.search(r"answer\s*:\s*([^,.\n]+)", raw, re.I)
    candidates = [match.group(1)] if match else re.split(r"[,.\n]", raw)
    for candidate in candidates:
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", candidate)
        if (1 <= len(words) <= 4
                and not any(w.lower() in _GARBAGE for w in words)):
            return " ".join(words).title()
    return None


def parse_json_items(raw: str) -> List[Dict[str, Any]]:
    """Extracts items[] from free text, tolerating code fences."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise PipelineError("llm_parse", "No JSON object in model output")
    try:
        payload = json.loads(raw[start:end + 1])
    except ValueError as err:
        raise PipelineError("llm_parse", f"Invalid JSON: {err}")
    return _clean_items(payload.get("items"))


def _clean_items(items: Any) -> List[Dict[str, Any]]:
    """Validates and normalizes raw items."""
    if not isinstance(items, list):
        raise PipelineError("llm_parse", "Model output has no items[]")
    cleaned = []
    for item in items:
        if not isinstance(item, dict) or not item.get("dish"):
            continue
        cleaned.append({
            "dish": str(item["dish"]).strip()[:80],
            "name_local": (str(item["name_local"]).strip()[:80]
                           if item.get("name_local") else None),
            "portion_g": _to_float(item.get("portion_g")),
            "confidence": _to_float(item.get("confidence")),
        })
    return cleaned


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _post_json(url: str, headers: Dict[str, str],
               body: Dict[str, Any]) -> Dict[str, Any]:
    """POSTs JSON with urllib and returns the decoded JSON reply."""
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request,
                                    timeout=_HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:400]
        raise PipelineError("llm_http", f"HTTP {err.code}: {detail}")
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise PipelineError("llm_http", f"Network: {err}")
    except ValueError as err:
        raise PipelineError("llm_http", f"Non-JSON response: {err}")
