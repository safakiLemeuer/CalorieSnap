"""Message formatting and the plain-text correction grammar.

Corrections a user can send after a photo:
    ok                 every dish is right
    2 papad 15g        dish 2 is actually papad, 15 g
    2 60g              dish 2 is right, the portion is 60 g
    remove 3           dish 3 is not on my plate
    add raita 80g      a dish the photo missed
    today              running total for today
Several corrections can share one message, separated by commas, semicolons,
or new lines.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

Command = Tuple[Any, ...]

_CONFIRM = {"ok", "okay", "yes", "y", "correct", "right", "sahi", "theek",
            "thik", "haan", "\U0001F44D"}
_HELP = {"help", "hi", "hello", "hey", "start", "namaste", "menu", "?"}
_TODAY = {"today", "total", "aaj"}

HELP_TEXT = (
    "*CalorieSnap*\n"
    "Send a photo of your plate. I name each dish and look up its calories "
    "in the ICMR-NIN Indian food database.\n\n"
    "After a photo you can reply:\n"
    "*ok* if everything is right\n"
    "*2 papad 15g* to change dish 2\n"
    "*2 60g* to change only the portion\n"
    "*remove 3* if dish 3 is not on your plate\n"
    "*add raita 80g* for something I missed\n"
    "*today* for your running total\n\n"
    "These are estimates for general wellness. They are not medical advice."
)


def parse(text: str) -> List[Command]:
    """Turns a reply into commands. Unknown text yields [("unknown",)]."""
    commands: List[Command] = []
    for part in re.split(r"[;\n,]+", text.strip()):
        chunk = re.sub(r"\s+", " ", part.strip().lower())
        if chunk:
            commands.append(_parse_one(chunk))
    return commands or [("unknown",)]


def _parse_one(chunk: str) -> Command:
    if chunk in _CONFIRM:
        return ("confirm",)
    if chunk in _HELP:
        return ("help",)
    if chunk in _TODAY:
        return ("today",)
    match = (re.fullmatch(r"(?:remove|delete|no|not)\s+(\d{1,2})", chunk)
             or re.fullmatch(r"(\d{1,2})\s+(?:no|remove|nahi|delete)",
                             chunk))
    if match:
        return ("remove", int(match.group(1)))
    match = re.fullmatch(r"add\s+(.+?)(?:\s+(\d{1,4})\s*(?:g|gm|gram|grams))?",
                         chunk)
    if match:
        return ("add", _title(match.group(1)), _grams(match.group(2)))
    match = re.fullmatch(r"(\d{1,2})\s+(\d{1,4})\s*(?:g|gm|gram|grams)",
                         chunk)
    if match:
        return ("portion", int(match.group(1)), float(match.group(2)))
    match = re.fullmatch(
        r"(\d{1,2})\s+(.+?)(?:\s+(\d{1,4})\s*(?:g|gm|gram|grams))?", chunk)
    if match and re.search(r"[a-z]", match.group(2)):
        return ("dish", int(match.group(1)), _title(match.group(2)),
                _grams(match.group(3)))
    return ("unknown",)


def _title(name: str) -> str:
    return " ".join(word.capitalize() for word in name.split())[:80]


def _grams(value: Optional[str]) -> Optional[float]:
    return float(value) if value else None


def uncounted(items: List[Dict[str, Any]]) -> int:
    """Dishes on the plate that contribute nothing to the total."""
    return sum(1 for i in items if not i.get("removed")
               and (i.get("cal") is None or i.get("suspect")))


def meal_summary(items: List[Dict[str, Any]], total: float,
                 updated: bool = False) -> str:
    """The reply to a photo, or to a correction."""
    if not items:
        return ("I could not find food in that photo. Try a closer shot with "
                "the whole plate in frame, or tell me what it was: "
                "*add poha 200g*")
    missing = uncounted(items)
    head = "Updated" if updated else "Your plate"
    lines = [f"*{head}: {round(total):,}{'+' if missing else ''} kcal*"]
    for number, item in enumerate(items, start=1):
        if item.get("removed"):
            continue
        grams = f", {round(item['grams'])} g" if item.get("grams") else ""
        if item.get("cal") is None:
            value = "not in the database yet"
        elif item.get("suspect"):
            value = "not counted, database value under review"
        else:
            value = f"{round(item['cal']):,}"
            if item.get("approx"):
                value += " (closest match)"
        lines.append(f"{number}. {item['name']}{grams}: {value}")
    if missing:
        lines.append(f"\n{missing} dish{'es' if missing > 1 else ''} not "
                     "counted, so the real total is higher.")
    if not updated:
        lines.append("\nRight? Reply *ok*. To fix: *2 papad 15g*, "
                     "*remove 3*, *add raita 80g*.")
    return "\n".join(lines)


def today_summary(stats: Dict[str, Any]) -> str:
    """Reply to "today"."""
    if not stats["meals"]:
        return "Nothing logged today. Send a photo of your next meal."
    plus = "+" if stats["partial"] else ""
    meals = "meal" if stats["meals"] == 1 else "meals"
    return f"*Today: {stats['total']:,}{plus} kcal* across " \
           f"{stats['meals']} {meals}."
