#!/usr/bin/env python
"""Launch run_overnight.py in the background on a Linux server.

From the project root:
    python scripts/launch_overnight.py
Or use the virtual environment explicitly:
    .venv-reu-server/bin/python scripts/launch_overnight.py
"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="python_path", help="Path to server virtual environment Python")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Run a foreground synthetic smoke check only")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without launching training")
    args = parser.parse_args()
    if sys.platform == "win32":
        parser.error("Upload this launcher to the Linux server and run it there.")
    local_python = ROOT / ".venv-reu-server" / "bin" / "python"
    interpreter = str(Path(args.python_path).expanduser().resolve()) if args.python_path else (
        str(local_python) if local_python.is_file() else sys.executable)
    runner = ROOT / "scripts" / "run_overnight.py"
    command = [interpreter, "-u", str(runner), "--device", args.device]
    if args.smoke or args.dry_run:
        command.append("--smoke" if args.smoke else "--dry-run")
        if args.smoke and args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    # Fail visibly before detaching when the selected environment is unsuitable.
    check = "import torch, optuna, yaml, torch_geometric, rdkit; "
    if args.device == "cuda":
        check += "assert torch.cuda.is_available(), 'CUDA unavailable in selected Python'; "
    check += "print('Environment check passed')"
    subprocess.run([interpreter, "-c", check], cwd=ROOT, check=True)
    name = "outputs/assay/overnight/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / name
    log_path = ROOT / (name + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command.extend(["--output-dir", str(output)])
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    (ROOT / (name + ".pid")).write_text(str(process.pid) + "\n", encoding="utf-8")
    print(f"Started background process: {process.pid}\nPython: {interpreter}\nLog: {log_path}")
    print(f"Results after completion: {output / 'eval/eval_results.json'}")
    print("You can disconnect SSH. Check the log for progress or errors.")
    print("This starts a new experiment each time; do not launch it again while it is running.")


if __name__ == "__main__":
    main()
