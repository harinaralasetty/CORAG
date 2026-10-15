import io

import soundfile as sf
import speech_recognition as sr


def _mp3_as_wav(path):
    """SpeechRecognition reads only WAV, AIFF and FLAC, so decode MP3 to an in-memory 16-bit WAV."""
    try:
        data, rate = sf.read(path, dtype="int16")
    except sf.SoundFileError as e:
        raise ValueError(f"Could not decode the MP3 file: {e}") from e
    wav = io.BytesIO()
    sf.write(wav, data, rate, format="WAV", subtype="PCM_16")
    wav.seek(0)
    return wav


def transcribe_audio(file):
    recognizer = sr.Recognizer()
    if str(file).lower().endswith(".mp3"):
        file = _mp3_as_wav(file)
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    # Determine document theme if applicable
    document_theme = text[:2000]
    return text, document_theme
