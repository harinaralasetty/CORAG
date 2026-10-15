import io

import soundfile as sf
import speech_recognition as sr


MAX_AUDIO_SECONDS = 3600  # decoded audio is ~11x the MP3 size in memory, so refuse very long files
SEGMENT_SECONDS = 45  # Google's free Web Speech API rejects or truncates requests much longer than a minute


def _mp3_as_wav(path):
    """SpeechRecognition reads only WAV, AIFF and FLAC, so decode MP3 to an in-memory 16-bit WAV."""
    try:
        seconds = sf.info(path).duration
        if seconds > MAX_AUDIO_SECONDS:
            raise ValueError(f"The MP3 file is {seconds / 60:.0f} minutes long; the limit is {MAX_AUDIO_SECONDS // 60} minutes.")
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
        file = _mp3_as_wav(file)  # refuses long MP3 files before decoding
    with sr.AudioFile(file) as source:
        if source.DURATION > MAX_AUDIO_SECONDS:  # taken from the WAV header, before any audio is read
            raise ValueError(f"The WAV file is {source.DURATION / 60:.0f} minutes long; the limit is {MAX_AUDIO_SECONDS // 60} minutes.")
        audio = recognizer.record(source)
    # Slice the recording rather than calling record(duration=...) in a loop: that drops a chunk at every cut
    step = int(SEGMENT_SECONDS * audio.sample_rate) * audio.sample_width
    texts = []
    for start in range(0, len(audio.frame_data), step):
        segment = sr.AudioData(audio.frame_data[start:start + step], audio.sample_rate, audio.sample_width)
        try:
            texts.append(recognizer.recognize_google(segment))
        except sr.UnknownValueError:
            continue  # silence or noise in this segment
        except sr.RequestError as e:
            raise ValueError(f"The Google Web Speech service could not transcribe the audio: {e}") from e
    if not texts:
        raise ValueError("No speech could be recognised in the audio file")
    text = " ".join(texts)
    # Determine document theme if applicable
    document_theme = text[:2000]
    return text, document_theme
