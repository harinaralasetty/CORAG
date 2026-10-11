"""Test setup that must run before any app module is imported."""
import os
import tempfile

# Never touch the real data folder or real APIs from tests
os.environ["CORAG_DATA_DIR"] = tempfile.mkdtemp(prefix="corag_test_data_")
os.environ["GOOGLE_API_KEY"] = "test-key"  # invalid on purpose: any accidental real call fails fast

pytest_plugins = ["nicegui.testing.user_plugin"]
