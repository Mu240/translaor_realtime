# Real-Time Translator

A web application that provides real-time speech transcription, translation, and text-to-speech functionality.

## Features
- Real-time transcription using OpenAI Whisper API.
- Translation into multiple languages.
- Text-to-speech output using gTTS.
- Supports multiple languages including English, Spanish, French, German, etc.
- Runs on localhost or HTTPS environments.

## Prerequisites
- Python 3.x
- Flask
- OpenAI API key
- Required Python packages (listed in `requirements.txt`)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Mu240/translaor_realtime.git
   cd translaor_realtime
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set the OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your_api_key_here'
   ```
   Or create a `.env` file with `OPENAI_API_KEY=your_api_key_here`.

5. Run the application:
   ```
   python main.py
   ```
6. Open your browser and navigate to `http://localhost:5000`.

## Usage
- Access the app via `http://localhost:5000`.
- Allow microphone access when prompted.
- Select the target language from the dropdown.
- Speak naturally; the app will transcribe, translate, and provide audio output.

## Files
- `main.py`: Flask application backend.
- `index.html`: Frontend interface.
- `requirements.txt`: Python dependencies.
- `templates/`: Directory for HTML templates.
- `.env`: Environment variables (e.g., API key).
