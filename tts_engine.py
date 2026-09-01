import re
import sounddevice as sd
import numpy as np
from kokoro_onnx import Kokoro

SPANISH_CHARS = re.compile(r"[áéíóúñü¿¡]")
SPANISH_WORDS = re.compile(
    r"\b("
    r"el|la|los|las|un|una|es|soy|estoy|hola|gracias|qué|que|cómo|como|"
    r"muy|bien|por|favor|diga|di|repeat|ahora|vamos|clase|lección|estás|estas|por favor"
    r")\b",
    re.IGNORECASE,
)
ENGLISH_WORDS = re.compile(
    r"\b("
    r"the|is|are|was|were|means|word|hello|you|your|that|this|what|how|does|say|repeat|now|"
    r"and|to|a|of|in|for|on|with|as|it|at|be|this|have|from|or|by|hot|but"
    r")\b",
    re.IGNORECASE,
)
QUOTE_PATTERN = re.compile(r'("[^"]+"|\'[^\']+\')')


class TTSEngine:
    # Changed default english_voice to something else (e.g., 'af_sarah')
    def __init__(self, spanish_voice: str = "ef_dora", english_voice: str = "af_sarah"):
        print("Loading Kokoro TTS model...")
        self.kokoro = Kokoro("kokoro-v1_0.onnx", "voices-v1_0.bin")
        self.sample_rate = 24000

        voices = self.kokoro.get_voices()

        if spanish_voice in voices:
            self.es_voice = spanish_voice
        else:
            es_matches = [v for v in voices if v.startswith(("ef_", "es_", "em_"))]
            self.es_voice = es_matches[0] if es_matches else voices[0]

        if english_voice in voices:
            self.en_voice = english_voice
        else:
            en_matches = [v for v in voices if v.startswith(("af_", "am_", "bf_", "bm_"))]
            self.en_voice = en_matches[0] if en_matches else self.es_voice

        print(
            f"[TTS Info]: Spanish voice '{self.es_voice}', English voice '{self.en_voice}'"
        )
        self.is_speaking = False

    def _normalize_text(self, text: str) -> str:
        """Replace or expand contractions so apostrophes don't cause text splitting issues."""
        cleaned = text
        # Expand or flatten common English contractions globally
        cleaned = re.sub(r"\bwe're\b", "we are", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bwe'll\b", "we will", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bi'm\b", "I am", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bi'll\b", "I will", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\byou're\b", "you are", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\byou'll\b", "you will", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\byou've\b", "you have", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bthey're\b", "they are", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bthey'll\b", "they will", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bit's\b", "it is", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bdon't\b", "do not", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bcan't\b", "cannot", cleaned, flags=re.IGNORECASE)
        
        # Catch-all fallback for any remaining apostrophe-based suffixes 
        # (e.g., turning "he'll" into "he ll" or just removing the apostrophe so it doesn't break tokens)
        cleaned = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1\2", cleaned)
        
        return cleaned

    def _is_spanish_text(self, text: str) -> bool:
        normalized = self._normalize_text(text).lower()
        clean_word_text = normalized.replace("'", "")
        
        if SPANISH_CHARS.search(clean_word_text):
            return True
            
        words = re.findall(r"\b\w+\b", clean_word_text)
        if not words:
            return False

        spanish_hits = len(SPANISH_WORDS.findall(clean_word_text))
        english_hits = len(ENGLISH_WORDS.findall(clean_word_text))

        forced_english_tokens = {"we", "are", "i", "am", "you", "they", "it", "is", "do", "can"}
        if any(token in words for token in forced_english_tokens):
            return False

        if english_hits > spanish_hits:
            return False
        if english_hits >= 2 and spanish_hits == 0:
            return False

        if spanish_hits >= 2:
            return True
        return spanish_hits >= 1 and len(words) <= 4 and english_hits == 0

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]
    
    def _split_language_segments(self, text: str) -> list[tuple[str, str]]:
        """Split text into (segment, lang) pairs using explicit [es]...[/es] tags or generic [...] brackets."""
        clean_text = re.sub(r"[*_#`]", "", text).strip()
        if not clean_text:
            return []

        # Match either [es]...[/es] OR fallback generic brackets [...]
        tag_pattern = re.compile(r"(?:\[es\](.*?)\[/es\]|\[(.*?)\])", re.DOTALL | re.IGNORECASE)
        
        segments: list[tuple[str, str]] = []
        cursor = 0

        for match in tag_pattern.finditer(clean_text):
            # Everything before the tag is English explanation
            before = clean_text[cursor : match.start()].strip()
            if before:
                segments.append((before, "en-us"))

            # Group 1 is inside [es]...[/es], Group 2 is inside generic [...]
            spanish_phrase = (match.group(1) or match.group(2) or "").strip()
            if spanish_phrase:
                segments.append((spanish_phrase, "es"))

            cursor = match.end()

            # Catch any trailing English text after the last tag
        tail = clean_text[cursor:].strip()
        if tail:
            segments.append((tail, "en-us"))

        # Fallback if no tags/brackets were found at all
        if not segments:
            segments.append((clean_text, "en-us"))

        return segments



    def _should_interrupt(self, interrupt_check_fn, barge_in_check_fn) -> bool:
        if interrupt_check_fn and interrupt_check_fn():
            return True
        if barge_in_check_fn and barge_in_check_fn():
            return True
        return False

    def speak(
        self,
        text: str,
        interrupt_check_fn=None,
        barge_in_check_fn=None,
        barge_in_stream=None,
    ):
        """Synthesize and play text, switching EN/ES voice per segment."""
        if not text.strip():
            return

        segments = self._split_language_segments(text)
        self.is_speaking = True
        interrupted = False

        try:
            if barge_in_stream is not None:
                barge_in_stream.start()

            for segment_text, lang in segments:
                if self._should_interrupt(interrupt_check_fn, barge_in_check_fn):
                    interrupted = True
                    break

                voice = self.es_voice if lang == "es" else self.en_voice
                samples, sr = self.kokoro.create(
                    segment_text, voice=voice, speed=1.0, lang=lang
                )

                chunk_size = 1024
                with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as stream:
                    for i in range(0, len(samples), chunk_size):
                        if self._should_interrupt(interrupt_check_fn, barge_in_check_fn):
                            interrupted = True
                            stream.stop()
                            break
                        stream.write(samples[i : i + chunk_size])

                if interrupted:
                    break

            if interrupted:
                print("\n[Skipped — start speaking when ready]")
        except Exception as exc:
            print(f"[TTS Error]: {exc}")
        finally:
            if barge_in_stream is not None:
                barge_in_stream.stop()
            self.is_speaking = False
