# 🇪🇸 Spanish-Tutor

A local, voice-driven AI tutor built in Python for practicing conversational Spanish through real-time speech, curriculum tracking, and phrase evaluation.

## 🎯 Project Goals

1. Build an AI tutor for Spanish that is effective.
2. Only then move on to making it as efficient as possible, to see how weak of a machine it can run on.
3. See if it's possible to make it adaptable to different machines, so that people can run it and use it without needing advanced technical knowledge.

## 🛠️ Repository Structure

* `app.py` — Main entry point that ties everything together.
* `audio_pipeline.py` — Handles microphone recording, audio streaming, and sockets.
* `tts_engine.py` — Handles local speech synthesis using model binaries.
* `curriculum_manager.py` — Tracks learned words, topics, and recap logic.
* `phrase_matcher.py` — Checks spoken answers against current learning goals.
* `curriculum.json` — Stores user progress and session history locally.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Virtual environment (`.venv`) set up

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Magus555/Spanish-Tutor.git](https://github.com/Magus555/Spanish-Tutor.git)
   cd Spanish-Tutor