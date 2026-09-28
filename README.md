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
* [Ollama](https://ollama.com/) installed and running locally
  * Assumes Ollama is running on the default local endpoint (`http://localhost:11434`).
  * Ensure you have pulled the model referenced in the code (e.g., `ollama pull llama3`).

> **Note on Hardware:** Running a local LLM alongside TTS synthesis requires sufficient CPU/GPU and RAM. Latency and responsiveness will depend on your system specs.

### Installation

1. Clone the repository:
   git clone https://github.com/Magus555/Spanish-Tutor.git
   cd Spanish-Tutor

2. Set up the virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. **Download Voice Weights:**
   Download `voices-v1.0.bin` and `kokoro-v1.0.onnx` from the [Kokoro ONNX releases](https://github.com/thewhitetulip/kokoro-onnx/releases) and place them in the project root.
   * *Note:* The code defaults to the `es_dora` Spanish voice embedding included in `voices-v1.0.bin`.

5. Run the Application:
   python app.py