"""Crash-safe saves: write a temp file beside the target, fsync it, then rename it into place."""
import json
import os
import tempfile


def make_temp_path(path):
    """Create an empty temp file in the target's folder, so os.replace stays a same-disk rename."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=os.path.basename(path) + ".", suffix=".tmp")
    os.close(fd)
    return tmp


def fsync_file(path):
    with open(path, "r+b") as f:
        os.fsync(f.fileno())


def write_json_temp(path, data, **json_kwargs):
    """Write data as JSON to a synced temp file beside path; return the temp path for os.replace."""
    tmp = make_temp_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_kwargs)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        os.remove(tmp)
        raise
    return tmp


def save_json_atomic(path, data, **json_kwargs):
    """Replace path with data as JSON; a crash leaves either the old file or the new one, never a partial one."""
    os.replace(write_json_temp(path, data, **json_kwargs), path)
