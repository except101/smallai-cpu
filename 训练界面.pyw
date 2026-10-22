"""Double-click launcher for the local training dashboard, without a console."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent
ADDRESS = "http://127.0.0.1:8765"


def ready():
    try:
        with urlopen(ADDRESS + "/api/health", timeout=0.5) as response:
            return json.load(response).get("app") == "smallai-training-ui"
    except (URLError, OSError, ValueError):
        return False


def main():
    if not ready():
        output = ROOT / "outputs"
        output.mkdir(exist_ok=True)
        executable = Path(sys.executable)
        if executable.name.lower() == "pythonw.exe":
            executable = executable.with_name("python.exe")
        with (output / "ui_server.log").open("a", encoding="utf-8") as stream:
            subprocess.Popen([str(executable), "-m", "smallai", "ui", "--no-browser"], cwd=ROOT,
                             stdout=stream, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for _ in range(80):
            if ready():
                break
            time.sleep(0.25)
        else:
            from tkinter import Tk, messagebox
            root = Tk()
            root.withdraw()
            messagebox.showerror("SmallAI", "训练界面启动失败。请查看 outputs/ui_server.log，或在项目目录运行：\npython -m smallai ui")
            root.destroy()
            return
    if "--no-browser" not in sys.argv:
        webbrowser.open(ADDRESS)


if __name__ == "__main__":
    main()
