import speech_recognition as sr

def transcribe_audio(file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio)
    # Determine document theme if applicable
    document_theme = text[:2000]
    return text, document_theme
