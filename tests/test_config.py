def test_the_app_listens_on_this_computer_only_by_default():
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "CORAG_HOST"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = lambda extra: subprocess.run([sys.executable, "-c", "import config; print(config.HOST)"], cwd=root,
                                       env={**env, **extra}, capture_output=True, text=True, check=True).stdout.strip()
    assert run({}) == "127.0.0.1"
    assert run({"CORAG_HOST": "0.0.0.0"}) == "0.0.0.0"
