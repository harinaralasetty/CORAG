"""Decode real audio through transcribe_audio; only the network call to Google is faked."""
import os

import pytest
import speech_recognition as sr

from preprocessing.audio_processor import transcribe_audio

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")  # 1 s excerpts of the project's sample recording


@pytest.fixture
def heard(monkeypatch):
    """Replace the Google call and record the audio it would have been sent."""
    calls = []

    def fake_recognize_google(self, audio, *args, **kwargs):
        calls.append(audio)
        return "transcribed words"

    monkeypatch.setattr(sr.Recognizer, "recognize_google", fake_recognize_google)
    return calls


@pytest.mark.parametrize("name", ["speech_sample.wav", "speech_sample.mp3"])
def test_wav_and_mp3_are_decoded_and_transcribed(heard, name):
    text, theme = transcribe_audio(os.path.join(FIXTURES, name))
    assert (text, theme) == ("transcribed words", "transcribed words")
    audio = heard[0]
    assert (audio.sample_rate, audio.sample_width) == (44100, 2)
    seconds = len(audio.frame_data) / audio.sample_width / audio.sample_rate  # mono after decoding
    assert 0.9 < seconds < 1.1
    assert audio.get_raw_data().strip(b"\0")  # real sound, not silence


def test_unreadable_mp3_gives_a_clear_error(heard, tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(ValueError, match="Could not decode the MP3 file"):
        transcribe_audio(str(bad))
    assert heard == []
