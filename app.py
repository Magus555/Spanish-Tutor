import sys

import requests

from curriculum_manager import CurriculumManager
from audio_pipeline import AudioPipeline
from tts_engine import TTSEngine
from session_controls import InterruptHandler
import re

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:14b"


class VoiceTutorApp:
    def __init__(self):
        print("Initializing Spanish Voice Tutor components...")
        self.curriculum = CurriculumManager()
        self.audio = AudioPipeline(sample_rate=16000)
        self.tts = TTSEngine(spanish_voice="ef_dora", english_voice="af_bella")
        self.interrupts = InterruptHandler()

    def call_ollama(self, prompt: str, system_prompt: str) -> str:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9},
        }
        try:
            res = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except Exception as exc:
            print(f"\n[Error calling Ollama]: {exc}")
            return "Lo siento, tuve un problema. Can you repeat that?"

    def _speak_with_barge_in(self, text: str):
        # Strip any rogue spaces right after the opening bracket AND before the closing bracket
        cleaned_text = re.sub(r'\[es\]\s*', '[es]', text)
        cleaned_text = re.sub(r'\s*\[/es\]', '[/es]', cleaned_text)
        
        barge_in_check, barge_in_stream = self.audio.create_barge_in_monitor()
        self.interrupts.arm()
        try:
            self.tts.speak(
                cleaned_text,
                interrupt_check_fn=self.interrupts.check,
                barge_in_check_fn=barge_in_check,
                barge_in_stream=barge_in_stream,
            )
        finally:
            self.interrupts.disarm()

    def start_session(self):
        print("\n==================================================")
        print("    SPANISH VOICE TUTOR (LOCAL GPU PIPELINE)    ")
        print("==================================================")
        print("Speak into your microphone. Pause ~1s when finished.")
        print("While the tutor is speaking: press Enter to skip.")
        print("Or talk over the tutor to interrupt (works best with headphones).")
        print("Press Ctrl+C at any time to exit.\n")

        # Check and generate initial targets if empty
        if self.curriculum.get_next_target_word() is None:
            self.curriculum.generate_more_targets(self.call_ollama)

        system_prompt = self.curriculum.build_system_prompt()
        initial_greeting = self.call_ollama(
            "Start the lesson and introduce yourself briefly.", system_prompt
        )

        print(f"\n🤖 Tutor: {initial_greeting}")
        self._speak_with_barge_in(initial_greeting)

        try:
            while True:
                # Check if we ran out of targets during conversation
                if self.curriculum.get_next_target_word() is None:
                    self.curriculum.generate_more_targets(self.call_ollama)
                    system_prompt = self.curriculum.build_system_prompt()
                    
                print("\n🎤 Listening... (Spanish or English)")

                user_text = self.audio.listen_and_transcribe()
                if not user_text or not user_text.strip():
                    print("⚠️ No clear speech recognized. Listening again...")
                    continue

                print(f"👤 You: {user_text}")

                self.curriculum.update_curriculum(user_text)

                system_prompt = self.curriculum.build_system_prompt()
                tutor_response = self.call_ollama(user_text, system_prompt)

                print(f"🤖 Tutor: {tutor_response}")
                self._speak_with_barge_in(tutor_response)

        except KeyboardInterrupt:
            print("\n\nSession ended. ¡Hasta luego!")
            sys.exit(0)


if __name__ == "__main__":
    app = VoiceTutorApp()
    app.start_session()
