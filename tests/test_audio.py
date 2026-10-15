"""Decode real audio through transcribe_audio; only the network call to Google is faked."""
import os
import types

import pytest
import soundfile as sf
import speech_recognition as sr

from preprocessing import audio_processor
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


def test_very_long_mp3_is_refused_before_it_is_decoded(heard, monkeypatch):
    monkeypatch.setattr(sf, "info", lambda path: types.SimpleNamespace(duration=7200))

    def must_not_decode(*args, **kwargs):
        raise AssertionError("a refused file must not be decoded")

    monkeypatch.setattr(sf, "read", must_not_decode)
    with pytest.raises(ValueError, match="120 minutes long"):
        transcribe_audio(os.path.join(FIXTURES, "speech_sample.mp3"))
    assert heard == []


def google_returns(monkeypatch, *outcomes):
    """Make recognize_google give each outcome in turn (raising exceptions) and record the audio it was sent."""
    calls = []

    def fake_recognize_google(self, audio, *args, **kwargs):
        calls.append(audio)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(sr.Recognizer, "recognize_google", fake_recognize_google)
    return calls


@pytest.mark.parametrize("name", ["speech_sample.wav", "speech_sample.mp3"])
def test_audio_is_sent_in_consecutive_short_segments(monkeypatch, name):
    monkeypatch.setattr(audio_processor, "SEGMENT_SECONDS", 0.3)
    path = os.path.join(FIXTURES, name)
    calls = google_returns(monkeypatch, "one", "two", "three", "four")
    text, theme = transcribe_audio(path)
    assert (text, theme) == ("one two three four", "one two three four")
    assert len(calls) == 4
    frames = [len(audio.frame_data) // audio.sample_width for audio in calls]
    assert all(n <= 0.3 * 44100 for n in frames)
    assert sum(frames) == len(sf.read(path)[0])  # nothing dropped or repeated between segments


def test_segments_without_speech_are_skipped(monkeypatch):
    monkeypatch.setattr(audio_processor, "SEGMENT_SECONDS", 0.3)
    google_returns(monkeypatch, "one", sr.UnknownValueError(), "three", sr.UnknownValueError())
    text, _ = transcribe_audio(os.path.join(FIXTURES, "speech_sample.wav"))
    assert text == "one three"


def test_audio_without_any_speech_gives_a_clear_error(monkeypatch):
    google_returns(monkeypatch, sr.UnknownValueError())
    with pytest.raises(ValueError, match="No speech could be recognised in the audio file"):
        transcribe_audio(os.path.join(FIXTURES, "speech_sample.wav"))


def test_speech_service_failure_gives_a_clear_error(monkeypatch):
    google_returns(monkeypatch, sr.RequestError("recognition request failed: Bad Request"))
    with pytest.raises(ValueError, match="Google Web Speech service.*Bad Request"):
        transcribe_audio(os.path.join(FIXTURES, "speech_sample.wav"))


def test_very_long_wav_is_refused_before_it_is_read(heard, monkeypatch):
    monkeypatch.setattr(audio_processor, "MAX_AUDIO_SECONDS", 0.5)

    def must_not_read(*args, **kwargs):
        raise AssertionError("a refused file must not be read")

    monkeypatch.setattr(sr.Recognizer, "record", must_not_read)
    with pytest.raises(ValueError, match="The WAV file is .* minutes long; the limit is"):
        transcribe_audio(os.path.join(FIXTURES, "speech_sample.wav"))
    assert heard == []
