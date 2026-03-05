import re
from typing import Any, Dict, Optional

try:
    from .lexicons import (
        HOT_SYNONYMS,
        COLD_SYNONYMS,
        INTENSIFIERS_STRONG,
        INTENSIFIERS_WEAK,
        FAN_LEVEL_WORDS,
        SET_VERBS,
        TEMP_WORDS,
        FAN_WORDS,
        MIN_TEMP_C,
        MAX_TEMP_C,
        MIN_FAN,
        MAX_FAN,
    )
except ImportError:
    # Allow running as a script (no package context) by importing local module
    from lexicons import (
        HOT_SYNONYMS,
        COLD_SYNONYMS,
        INTENSIFIERS_STRONG,
        INTENSIFIERS_WEAK,
        FAN_LEVEL_WORDS,
        SET_VERBS,
        TEMP_WORDS,
        FAN_WORDS,
        MIN_TEMP_C,
        MAX_TEMP_C,
        MIN_FAN,
        MAX_FAN,
    )


NUM_UNIT_REGEX = re.compile(
    r"(?P<num>\d{1,3})(?:\s?)(?P<unit>degrees|degree|°|c|celsius|f|fahrenheit)?",
    flags=re.IGNORECASE,
)


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_numbers_with_units(text: str) -> Optional[Dict[str, Any]]:
    for match in NUM_UNIT_REGEX.finditer(text):
        num = int(match.group("num"))
        unit = match.group("unit")
        if unit:
            unit = unit.lower()
        return {"num": num, "unit": unit}
    return None


def _maybe_convert_to_celsius(num: int, unit: Optional[str]) -> int:
    # If explicit unit provided
    if unit:
        if unit in {"f", "fahrenheit"}:
            return round((num - 32) * 5 / 9)
        # default to celsius for c/celsius/degree(s)/°
        return num
    # No unit: heuristic — if value > 60, assume Fahrenheit
    if num > 60:
        return round((num - 32) * 5 / 9)
    return num


def _intensity(text: str) -> int:
    tokens = set(text.split())
    has_strong = any(w in tokens for w in INTENSIFIERS_STRONG)
    has_weak = any(w in tokens for w in INTENSIFIERS_WEAK)
    if has_strong and has_weak:
        return 2
    if has_strong:
        return 3
    if has_weak:
        return 1
    return 2  # default medium


def _contains_any(text: str, vocab: set) -> bool:
    return any(word in text for word in vocab)


def parse_command(
    text: str,
    current_temp_c: Optional[int] = None,
    current_fan: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parse user text into structured AC actions.

    Returns a dict with keys 'temperature', 'fan_speed', 'meta'.
    - temperature: {type: 'absolute'|'relative', value_c?: int, delta?: int, confidence: float}
    - fan_speed: {type: 'absolute'|'relative', value?: int, delta?: int, confidence: float}
    - meta: {notes: list[str]}
    """
    original_text = text
    text = _normalize_text(text)

    temperature: Optional[Dict[str, Any]] = None
    fan_speed: Optional[Dict[str, Any]] = None
    notes = []

    # Absolute temperature setting patterns
    if _contains_any(text, SET_VERBS) and _contains_any(text, TEMP_WORDS):
        num_unit = _extract_numbers_with_units(text)
        if num_unit:
            value_c = _maybe_convert_to_celsius(num_unit["num"], num_unit["unit"])
            clamped = _clamp(value_c, MIN_TEMP_C, MAX_TEMP_C)
            if clamped != value_c:
                notes.append("temperature clamped to device bounds")
            temperature = {
                "type": "absolute",
                "value_c": clamped,
                "confidence": 0.98,
            }

    # Fan absolute level
    if _contains_any(text, FAN_WORDS):
        # Look for explicit level number 0..4
        level_number = None
        for token in re.findall(r"\d+", text):
            val = int(token)
            if MIN_FAN <= val <= MAX_FAN:
                level_number = val
                break
        # Or level by word (low/medium/high/max)
        if level_number is None:
            for word, level in FAN_LEVEL_WORDS.items():
                if word in text:
                    level_number = level
                    break
        if level_number is not None:
            fan_speed = {
                "type": "absolute",
                "value": _clamp(level_number, MIN_FAN, MAX_FAN),
                "confidence": 0.95,
            }

    # Relative adjustments from feelings
    if temperature is None:
        if _contains_any(text, HOT_SYNONYMS):
            mag = _intensity(text)  # 1..3
            delta = -mag  # colder
            temperature = {"type": "relative", "delta": delta, "confidence": 0.8}
            # Fan nudge up unless already specified
            if fan_speed is None:
                fan_speed = {"type": "relative", "delta": +1, "confidence": 0.6}
        elif _contains_any(text, COLD_SYNONYMS):
            mag = _intensity(text)
            delta = +mag  # warmer
            temperature = {"type": "relative", "delta": delta, "confidence": 0.8}
            if fan_speed is None:
                fan_speed = {"type": "relative", "delta": -1, "confidence": 0.6}

    # Generic increase/decrease without explicit numbers
    if temperature is None and _contains_any(text, TEMP_WORDS):
        if any(w in text for w in ["increase", "higher", "up", "raise"]):
            temperature = {"type": "relative", "delta": +1, "confidence": 0.6}
        elif any(w in text for w in ["decrease", "lower", "down", "reduce", "cooler", "colder"]):
            temperature = {"type": "relative", "delta": -1, "confidence": 0.6}

    if fan_speed is None and _contains_any(text, FAN_WORDS):
        if any(w in text for w in ["increase", "higher", "up", "raise", "faster", "stronger"]):
            fan_speed = {"type": "relative", "delta": +1, "confidence": 0.7}
        elif any(w in text for w in ["decrease", "lower", "down", "reduce", "slower", "weaker"]):
            fan_speed = {"type": "relative", "delta": -1, "confidence": 0.7}

    # If still nothing inferred, default no-ops with low confidence
    if temperature is None and fan_speed is None:
        notes.append("no actionable intent detected")

    result: Dict[str, Any] = {
        "temperature": temperature,
        "fan_speed": fan_speed,
        "meta": {"notes": notes, "original": original_text},
    }
    return result


if __name__ == "__main__":
    # Simple manual test
    samples = [
        "set temperature to 30 degrees",
        "i am feeling hot",
        "i am freezing",
        "set fan to high",
        "make the ac 75",
        "raise the temp",
        "turn the fan down",
    ]
    for s in samples:
        print(s, "->", parse_command(s))


