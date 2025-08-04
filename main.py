from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
from openai import OpenAIError
import os
import io
import base64
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
# Restrict CORS to trusted origins in production (replace with your frontend URL)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})  # Example origin

# Validate OpenAI API key
openai.api_key = "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"
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Create thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=3)

@app.route('/')
def index():
    return render_template('index.html')

def process_transcription(wav_io):
    """Process transcription in separate thread"""
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_io,
            response_format="json"
        )
        detected_language = transcription_response.language if hasattr(transcription_response, 'language') else 'unknown'
        if detected_language == 'unknown':
            logger.warning("Language detection failed")
        return transcription_response.text, detected_language
    except OpenAIError as e:
        logger.error(f"OpenAI API error in transcription: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error in transcription: {e}")
        return None, None

def process_translation(transcription, target_lang):
    """Process translation in separate thread"""
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        translation_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Translate the following text to {target_lang}. Return only the translation."},
                {"role": "user", "content": transcription}
            ],
            max_tokens=200,
            temperature=0.1
        )
        return translation_response.choices[0].message.content.strip()
    except OpenAIError as e:
        logger.error(f"OpenAI API error in translation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in translation: {e}")
        return None

def get_gtts_language_code(target_lang):
    """Map target language to gTTS language codes"""
    lang_mapping = {
        'spanish': 'es',
        'french': 'fr',
        'german': 'de',
        'italian': 'it',
        'portuguese': 'pt',
        'russian': 'ru',
        'chinese': 'zh-CN',  # Mandarin
        'japanese': 'ja',
        'korean': 'ko',
        'arabic': 'ar',
        'hindi': 'hi',
        'english': 'en',
        'dutch': 'nl',
        'swedish': 'sv',
        'norwegian': 'no',
        'danish': 'da',
        'finnish': 'fi',
        'turkish': 'tr',
        'polish': 'pl',
        'czech': 'cs',
        'hungarian': 'hu',
        'romanian': 'ro',
        'bulgarian': 'bg',
        'croatian': 'hr',
        'serbian': 'sr',
        'slovak': 'sk',
        'slovenian': 'sl',
        'estonian': 'et',
        'latvian': 'lv',
        'lithuanian': 'lt',
        'maltese': 'mt',
        'greek': 'el',
        'hebrew': 'he',
        'thai': 'th',
        'vietnamese': 'vi',
        'indonesian': 'id',
        'malay': 'ms',
        'tamil': 'ta',
        'telugu': 'te',
        'bengali': 'bn',
        'gujarati': 'gu',
        'kannada': 'kn',
        'malayalam': 'ml',
        'marathi': 'mr',
        'punjabi': 'pa',
        'urdu': 'ur'
    }
    return lang_mapping.get(target_lang.lower(), 'en')

def process_tts(text, target_lang):
    """Process TTS using Google TTS in separate thread"""
    try:
        lang_code = get_gtts_language_code(target_lang)
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts_io = io.BytesIO()
        tts.write_to_fp(tts_io)
        tts_io.seek(0)
        return base64.b64encode(tts_io.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

@app.route("/transcribe", methods=["POST"])
def transcribe():
    start_time = time.time()
    try:
        data = request.get_json()
        audio_b64 = data.get("audio")
        target_lang = data.get("target_language")
        if not audio_b64 or not target_lang:
            return jsonify({"error": "Missing audio or target_language"}), 400

        # Decode base64 audio
        audio_data = base64.b64decode(audio_b64)
        audio_io = io.BytesIO(audio_data)

        # Convert to WAV
        try:
            audio = AudioSegment.from_file(audio_io, format="webm")
        except CouldntDecodeError as e:
            logger.error(f"Audio format error: {e}")
            return jsonify({"error": "Unsupported audio format. Use webm or compatible formats."}), 400

        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        wav_io.name = "audio.wav"
        logger.info(f"Audio preprocessing: {time.time() - start_time:.2f}s")

        # Transcription
        transcription_future = executor.submit(process_transcription, wav_io)
        transcription, detected_language = transcription_future.result()
        if not transcription:
            return jsonify({"error": "Transcription failed"}), 500
        logger.info(f"Transcription completed: {time.time() - start_time:.2f}s")

        # Translation
        translation_future = executor.submit(process_translation, transcription, target_lang)
        translation = translation_future.result()
        if not translation:
            return jsonify({"error": "Translation failed"}), 500
        logger.info(f"Translation completed: {time.time() - start_time:.2f}s")

        # TTS
        tts_future = executor.submit(process_tts, translation, target_lang)
        tts_b64 = tts_future.result()
        if not tts_b64:
            return jsonify({"error": "TTS failed"}), 500

        logger.info(f"Total processing time: {time.time() - start_time:.2f}s")
        return jsonify({
            "transcription": transcription,
            "detected_language": detected_language,
            "translation": translation,
            "tts_audio": tts_b64,
            "processing_time": round(time.time() - start_time, 2)
        })

    except ValueError as e:
        logger.error(f"Input error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in transcribe: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe_streaming", methods=["POST"])
def transcribe_streaming():
    try:
        data = request.get_json()
        audio_b64 = data.get("audio")
        target_lang = data.get("target_language")
        if not audio_b64 or not target_lang:
            return jsonify({"error": "Missing audio or target_language"}), 400

        # Decode base64 audio
        audio_data = base64.b64decode(audio_b64)
        audio_io = io.BytesIO(audio_data)

        # Convert to WAV
        try:
            audio = AudioSegment.from_file(audio_io, format="webm")
        except CouldntDecodeError as e:
            logger.error(f"Audio format error: {e}")
            return jsonify({"error": "Unsupported audio format. Use webm or compatible formats."}), 400

        if len(audio) < 1000:  # Less than 1 second
            return jsonify({"status": "too_short"}), 400

        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        wav_io.name = "audio.wav"

        # Quick transcription
        try:
            client = openai.OpenAI(api_key=openai.api_key)
            transcription_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_io,
                response_format="text"
            )
        except OpenAIError as e:
            logger.error(f"OpenAI API error in streaming: {e}")
            return jsonify({"error": str(e)}), 500

        return jsonify({
            "transcription": transcription_response,
            "status": "partial"
        })

    except ValueError as e:
        logger.error(f"Input error in streaming: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error in streaming: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":

    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
