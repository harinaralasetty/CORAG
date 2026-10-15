import os
import stat

from file_utils import save_json_atomic


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_new_files_are_readable_like_ordinary_files(tmp_path):
    target = tmp_path / "chat_threads_info.json"
    save_json_atomic(str(target), {"a": 1})
    assert mode(target) == 0o644  # mkstemp alone would leave 0o600


def test_saving_keeps_an_existing_files_permissions(tmp_path):
    target = tmp_path / "chunks.json"
    target.write_text("{}")
    os.chmod(target, 0o640)
    save_json_atomic(str(target), {"b": 2})
    assert mode(target) == 0o640
