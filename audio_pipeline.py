import os
import sys

# ---------------------------------------------------------------------------
# Inject CUDA DLL paths on Windows BEFORE CTranslate2 imports
# ---------------------------------------------------------------------------
venv_base = sys.prefix
_dll_candidates = [
    os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cublas", "lib"),
    os.path.join(venv_base, "Lib", "site-packages", "nvidia", "cudnn", "lib"),
    os.path.join(venv_base, "Lib", "site-packages", "torch", "lib"),
]

for dll_dir in _dll_candidates:
    if os.path.isdir(dll_dir):
        os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

import torch
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import time


def _cuda_runtime_ready() -> bool:
    """True only if cublas DLL is present (ctranslate2 can use GPU)."""
    for dll_dir in _dll_candidates:
        if os.path.isfile(os.path.join(dll_dir, "cublas64_12.dll")):
            return True
    return False


def _load_whisper_model(model_size: str = "small") -> tuple[WhisperModel, str]:
    """Load Whisper on GPU when DLLs exist, otherwise CPU."""
    if _cuda_runtime_ready():
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print("[STT] Using CUDA (float16)")
            return model, "cuda"
        except Exception as exc:
            print(f"[STT] CUDA init failed ({exc}), falling back to CPU")

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print("[STT] Using CPU (int8)")
    return model, "cpu"


ALLOWED_STT_LANGUAGES = frozenset({"es", "en"})
MIN_LANGUAGE_CONFIDENCE = 0.5


class AudioPipeline:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        print("Loading Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )

        print("Loading Faster-Whisper (STT)...")
        self.stt, self._stt_device = _load_whisper_model("small")
        self.window_size_samples = 512

    def is_user_speaking(self, audio_chunk, threshold=0.6, min_rms=0.0) -> bool:
        if len(audio_chunk) < self.window_size_samples:
            return False

        chunk_slice = audio_chunk[: self.window_size_samples]
        if min_rms > 0:
            rms = float(np.sqrt(np.mean(chunk_slice ** 2)))
            if rms < min_rms:
                return False

        tensor_chunk = torch.from_numpy(chunk_slice).float()

        with torch.no_grad():
            speech_prob = self.vad_model(tensor_chunk, self.sample_rate).item()

        return speech_prob > threshold

    def record_until_silence(self, silence_duration=0.6, max_duration=15.0) -> np.ndarray:
        print("\n[Listening... Speak into your mic]")

        audio_buffer = []
        speaking_started = False
        silence_start = None
        start_time = time.time()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.window_size_samples,
        ) as stream:
            while True:
                chunk, _ = stream.read(self.window_size_samples)
                chunk_flat = chunk.flatten()

                if len(chunk_flat) < self.window_size_samples:
                    continue

                audio_buffer.append(chunk_flat)
                is_speech = self.is_user_speaking(chunk_flat)

                if is_speech:
                    if not speaking_started:
                        speaking_started = True
                        print("[Speech detected...]")
                    silence_start = None
                elif speaking_started:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= silence_duration:
                        print("[Finished speaking]")
                        break

                if time.time() - start_time > max_duration:
                    break

        if not audio_buffer:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio_buffer)

    def transcribe(self, audio_data: np.ndarray) -> str:
        if len(audio_data) == 0:
            return ""

        try:
            return self._run_transcription(audio_data)
        except RuntimeError as exc:
            if "cublas" not in str(exc).lower() and self._stt_device != "cuda":
                raise
            print(f"[STT] Transcription failed on {self._stt_device}: {exc}")
            print("[STT] Retrying on CPU...")
            self.stt = WhisperModel("small", device="cpu", compute_type="int8")
            self._stt_device = "cpu"
            return self._run_transcription(audio_data)

    def _transcribe_with_language(self, audio_data: np.ndarray, language: str) -> tuple[str, float]:
        segments, _ = self.stt.transcribe(
            audio_data,
            language=language,
            beam_size=1,
            vad_filter=True,
            initial_prompt="Hola, gracias, café, cómo estás, por favor.",
        )
        seg_list = list(segments)
        text = " ".join(segment.text for segment in seg_list).strip()
        if not seg_list:
            return text, float("-inf")
        score = sum(segment.avg_logprob for segment in seg_list) / len(seg_list)
        return text, score

    def _run_transcription(self, audio_data: np.ndarray) -> str:
        segments, info = self.stt.transcribe(
            audio_data,
            language=None,
            beam_size=1,
            vad_filter=True,
            initial_prompt="Hola, gracias, café, cómo estás, por favor.",
        )
        seg_list = list(segments)
        lang = info.language or ""
        confidence = info.language_probability or 0.0

        if lang in ALLOWED_STT_LANGUAGES and confidence >= MIN_LANGUAGE_CONFIDENCE:
            transcript = " ".join(segment.text for segment in seg_list).strip()
            print(f"[STT] Detected language: {lang} ({confidence:.0%})")
            return transcript

        if lang and lang not in ALLOWED_STT_LANGUAGES:
            print(
                f"[STT] Ignoring unlikely language '{lang}' ({confidence:.0%}) "
                "— retrying as Spanish/English"
            )

        es_text, es_score = self._transcribe_with_language(audio_data, "es")
        en_text, en_score = self._transcribe_with_language(audio_data, "en")

        if en_score > es_score:
            print(f"[STT] Using English (confidence score {en_score:.2f})")
            return en_text

        print(f"[STT] Using Spanish (confidence score {es_score:.2f})")
        return es_text

    def listen_and_transcribe(self) -> str:
        audio_data = self.record_until_silence()
        if len(audio_data) == 0:
            return ""
        return self.transcribe(audio_data)

    def create_barge_in_monitor(
        self,
        consecutive_frames: int = 6,
        threshold: float = 0.78,
        min_rms: float = 0.025,
        grace_seconds: float = 1.0,
    ):
        """Return (check_fn, stream_ctx) for mic barge-in during TTS playback."""
        speech_streak = {"count": 0}
        armed_at = {"time": None}

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.window_size_samples,
        )

        def check() -> bool:
            if armed_at["time"] is None:
                armed_at["time"] = time.time()

            try:
                chunk, _ = stream.read(self.window_size_samples)
            except Exception:
                return False

            if time.time() - armed_at["time"] < grace_seconds:
                return False

            if self.is_user_speaking(
                chunk.flatten(), threshold=threshold, min_rms=min_rms
            ):
                speech_streak["count"] += 1
                if speech_streak["count"] >= consecutive_frames:
                    return True
            else:
                speech_streak["count"] = 0
            return False

        return check, stream
