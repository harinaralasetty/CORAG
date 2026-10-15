import json

import pytest

from chat_management import chat_utils


def test_a_crash_while_saving_keeps_the_previous_chats(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_utils, "THREADS_INFO_FILE", str(tmp_path / "chat_threads_info.json"))
    chat_utils.save_chat_threads_info({"t1": {"name": "Kept"}})

    def crash_midway(obj, fp, **kwargs):
        fp.write('{"t1": {"na')
        raise OSError("disk full")

    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(json, "dump", crash_midway)
        chat_utils.save_chat_threads_info({"t1": {"name": "Kept"}, "t2": {"name": "New"}})

    assert chat_utils.load_chat_threads_info() == {"t1": {"name": "Kept"}}
    assert [p.name for p in tmp_path.iterdir()] == ["chat_threads_info.json"]  # no temp file left behind
