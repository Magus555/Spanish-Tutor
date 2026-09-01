import re
import unicodedata
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_similarity(user_text: str, target_phrase: str) -> float:
    a = normalize_text(user_text)
    b = normalize_text(target_phrase)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if b in a or a in b:
        return max(0.85, SequenceMatcher(None, a, b).ratio())
    return SequenceMatcher(None, a, b).ratio()


def is_spanish_utterance(user_text: str) -> bool:
    """Heuristic: short Spanish practice attempts should not be treated as English."""
    if re.search(r"[áéíóúñü¿¡]", user_text):
        return True
    normalized = normalize_text(user_text)
    spanish_cues = (
        "hola", "gracias", "por favor", "quisiera", "cafe", "leche",
        "cuenta", "cuanto", "cuesta", "uno", "una", "con",
    )
    words = normalized.split()
    if not words:
        return False
    spanish_hits = sum(1 for w in words if w in spanish_cues)
    english_cues = ("yes", "no", "what", "mean", "hold", "help", "repeat", "how")
    english_hits = sum(1 for w in words if w in english_cues)
    if english_hits >= 1 and spanish_hits == 0:
        return False
    return spanish_hits >= 1 or len(words) <= 3
