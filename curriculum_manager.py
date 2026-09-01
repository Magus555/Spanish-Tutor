import json
from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel, Field
import re

class Word(BaseModel):
    word: str
    meaning: str
    mastery: float = 0.0
    introduced: bool = True
    last_reviewed: str = Field(default_factory=lambda: datetime.now().isoformat())

class TargetWord(BaseModel):
    word: str
    meaning: str
    introduced: bool = False

class Curriculum(BaseModel):
    current_level: str
    active_topic: str
    known_words: List[Word]
    target_words: List[TargetWord]

class CurriculumManager:
    def __init__(self, filepath: str = "curriculum.json"):
        self.filepath = filepath
        self.curriculum = self.load_curriculum()
        self.apply_decay()  # Apply time-based decay on load

    def load_curriculum(self) -> Curriculum:
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Curriculum(**data)

    def save_curriculum(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.curriculum.model_dump(), f, indent=2, ensure_ascii=False)

    def apply_decay(self):
        """Reduces mastery slightly for known words if days have passed since last review."""
        now = datetime.now()
        updated = False
        for kw in self.curriculum.known_words:
            if hasattr(kw, 'last_reviewed') and kw.last_reviewed:
                try:
                    last_date = datetime.fromisoformat(kw.last_reviewed)
                    days_passed = (now - last_date).days
                    if days_passed > 1:
                        # Decay mastery by 5% per day past the first day, min 0.1
                        decay_amount = (days_passed - 1) * 0.05
                        kw.mastery = max(0.1, kw.mastery - decay_amount)
                        updated = True
                except Exception:
                    pass
        if updated:
            self.save_curriculum()

    def get_next_target_word(self) -> TargetWord | None:
        """Returns the next unintroduced target word."""
        for word in self.curriculum.target_words:
            if not word.introduced:
                return word
        return None

    def mark_word_introduced(self, target_word_str: str):
        """Moves a word from target list to introduced/known list with incremental mastery."""
        for tw in self.curriculum.target_words:
            if tw.word.lower() == target_word_str.lower():
                tw.introduced = True
                
                # Check if already in known words
                existing_kw = next((kw for kw in self.curriculum.known_words if kw.word.lower() == tw.word.lower()), None)
                if existing_kw:
                    # Gradual build-up: increase mastery incrementally (max 1.0)
                    existing_kw.mastery = min(1.0, existing_kw.mastery + 0.3)
                    existing_kw.last_reviewed = datetime.now().isoformat()
                else:
                    # First time introduction starts at lower mastery so it takes practice
                    self.curriculum.known_words.append(
                        Word(word=tw.word, meaning=tw.meaning, mastery=0.3, last_reviewed=datetime.now().isoformat())
                    )
                self.save_curriculum()
                break

    def update_curriculum(self, user_text: str):
        """Update word progress when the user uses target vocabulary."""
        user_clean = re.sub(r"[^\w\s]", "", user_text).lower()
        
        # Only check unintroduced targets
        for tw in self.curriculum.target_words:
            if not tw.introduced:
                target_clean = re.sub(r"[^\w\s]", "", tw.word).lower()
                if target_clean in user_clean or any(word in user_clean for word in target_clean.split() if len(word) > 3):
                    self.mark_word_introduced(tw.word)
                    print(f"[Curriculum]: Progress made on -> '{tw.word}'")
                    break

        # Update review timestamp and mastery for matched known words
        for kw in self.curriculum.known_words:
            kw_clean = re.sub(r"[^\w\s]", "", kw.word).lower()
            if kw_clean in user_clean:
                kw.mastery = min(1.0, kw.mastery + 0.2)
                kw.last_reviewed = datetime.now().isoformat()
                
        self.save_curriculum()

    def generate_more_targets(self, call_llm_fn):
        """Asks the LLM to generate new target words when the list is empty."""
        print("\n[Curriculum]: Generating next lesson plan...")
        known_list = ", ".join([w.word for w in self.curriculum.known_words])
        
        prompt = f"""Based on the A1 Spanish topic '{self.curriculum.active_topic}', generate 3 new, logical next-step vocabulary words or short phrases for a beginner student.
        The student already knows these words: [{known_list}].
        
        Return your response STRICTLY as a JSON array of objects, with no extra text or markdown blocks, matching this exact format:
        [
          {{"word": "spanish phrase", "meaning": "english meaning", "introduced": false}},
          {{"word": "spanish phrase 2", "meaning": "english meaning 2", "introduced": false}}
        ]"""
        
        response_text = call_llm_fn(prompt, "You are a helpful curriculum designer. Output only raw JSON.")
        try:
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            new_targets_data = json.loads(clean_json)
            for item in new_targets_data:
                self.curriculum.target_words.append(TargetWord(**item))
            self.save_curriculum()
            print("[Curriculum]: Successfully added new words to your session!")
        except Exception as exc:
            print(f"[Curriculum Error]: Failed to parse generated targets: {exc}")

    def build_system_prompt(self) -> str:
        """Generates the conversational system prompt for the LLM."""
        known_str = ", ".join([f"'{w.word}' ({w.meaning} - {int(w.mastery * 100)}% mastery)" for w in self.curriculum.known_words])
        next_target = self.get_next_target_word()
        target_str = f"'{next_target.word}' ({next_target.meaning})" if next_target else "None"

        return f"""You are an encouraging, conversational Spanish voice tutor named Matea. You help an A1 beginner student learn naturally through spoken chat.

Topic: {self.curriculum.active_topic}
Level: {self.curriculum.current_level}

STUDENT'S STRICT VOCABULARY LIMITS:
- KNOWN WORDS: [{known_str}]
- CURRENT TARGET WORD/PHRASE: {target_str}

🚨 **ABSOLUTE LANGUAGE RULE (CRITICAL)**: 
You must speak **100% English** for all conversational feedback, explanations, praise, and questions. 
- NEVER reply in conversational Spanish (no "¡Perfecto! Has usado...", no "Muy bien..."). 
- The ONLY Spanish allowed in your entire output is the single target phrase wrapped inside [es] tags. If you write any Spanish outside of [es] tags, you fail your instructions.

CONVERSATION & TUTORING RULES:
1. ONE TASK AT A TIME: Ask *one* question or give *one* practice phrase, then immediately stop and wait for the student.
2. STRUCTURED TURNS: 
   - Step 1: Briefly acknowledge their last input in plain English (e.g., "Great job!").
   - Step 2: Introduce the next target word or ask them to try a phrase.
   - Step 3: Use a clear cue like "Your turn to try:" followed strictly by the Spanish target wrapped in [es] tags.
3. BREVITY: Keep your turns short (1-2 short sentences max)."""
