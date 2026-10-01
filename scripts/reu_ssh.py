"""Password SSH for this project's unprivileged server account.

Requires paramiko and python-dotenv in a project-local Python environment.
The host key must already be trusted in ~/.ssh/known_hosts.
"""

import argparse
import os
from pathlib import Path
import shlex
import sys
import time

from dotenv import dotenv_values
import paramiko


ROOT = Path(__file__).resolve().parents[1]


def credentials():
    values = dotenv_values(ROOT / ".env") if (ROOT / ".env").is_file() else {}
    result = {name: os.environ.get(name) or values.get(name) for name in ("SSH_HOST", "SSH_USER", "SSH_PASSWORD")}
    missing = [name for name, value in result.items() if not value]
    if missing:
        raise ValueError("Missing SSH environment variables: " + ", ".join(missing))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("hello", "run"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "hello":
        if args.command:
            parser.error("hello takes no remote command")
        command = "printf 'hello world\\n'"
    else:
        words = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not words:
            parser.error("run requires a command after --")
        command = shlex.join(words)

    cfg = credentials()
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    if not known_hosts.is_file():
        raise RuntimeError("No trusted ~/.ssh/known_hosts file; verify the server key out of band first")

    client = paramiko.SSHClient()
    client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        client.connect(
            hostname=cfg["SSH_HOST"],
            username=cfg["SSH_USER"],
            password=cfg["SSH_PASSWORD"],
            look_for_keys=False,
            allow_agent=False,
            timeout=10,
            auth_timeout=10,
            banner_timeout=10,
        )
        client.get_transport().set_keepalive(30)
        _, stdout, _ = client.exec_command(command, timeout=15 if args.mode == "hello" else None)
        channel = stdout.channel
        while True:
            active = False
            if channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(32768))
                sys.stdout.flush()
                active = True
            if channel.recv_stderr_ready():
                sys.stderr.buffer.write(channel.recv_stderr(32768))
                sys.stderr.flush()
                active = True
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            if not active:
                if not client.get_transport().is_active():
                    raise RuntimeError("SSH disconnected; remote job state is unknown. Inspect before relaunching.")
                time.sleep(0.05)
        status = channel.recv_exit_status()
        if args.mode == "hello":
            print(f"Remote exit status: {status}")
        return status
    finally:
        client.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, paramiko.SSHException, OSError) as exc:
        print(f"SSH check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
