"""Test a Smartloop launchd handoff on macOS using two local slp binaries.

No installer, VM, or model weights are needed. The test uses a temporary
home, an ephemeral loopback port, and its own temporary GUI launchd job.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, timeout=15)


def launchctl(*args):
    return run("/bin/launchctl", *args)


def checked(result):
    if result.returncode:
        raise RuntimeError(f"{result.args}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout


def process_running(pid):
    status = run("/bin/ps", "-p", str(pid), "-o", "stat=").stdout.strip()
    return bool(status) and not status.startswith("Z")


class Probe:
    def __init__(self, root, domain):
        self.root = root
        self.home = root / "home"
        self.home.mkdir()
        self.label = f"com.smartloop.smoke.{root.name}"
        self.domain = domain
        self.target = f"{domain}/{self.label}"
        self.plist = root / f"{self.label}.plist"
        self.link = root / "slp"
        self.base_url = None
        self.opener = build_opener(ProxyHandler({}))

        with self.plist.open("wb") as file:
            plistlib.dump({
                "Label": self.label,
                "ProgramArguments": [str(self.link), "agent", "start"],
                "WorkingDirectory": str(self.home),
                "RunAtLoad": True,
                "StandardOutPath": str(root / "launchd.log"),
                "StandardErrorPath": str(root / "launchd.log"),
                "EnvironmentVariables": {
                    "HOME": str(self.home),
                    "XDG_CONFIG_HOME": str(root / "config"),
                    "XDG_CACHE_HOME": str(root / "cache"),
                    "XDG_DATA_HOME": str(root / "data"),
                    "SLP_HOME": str(self.home),
                    "SLP_EXE": "",  # Neutralize any launchd override; use default self-reexec.
                    "SLP_PROJECT_ID": "",
                    "SLP_MODEL_HOST": "",
                    "SLP_API_HOST": "127.0.0.1",
                    "SLP_API_PORT": "0",
                    "SLP_API_PRELOAD_MODEL": "0",
                    "SLP_PARENT_WATCH_INTERVAL_S": "0.5",
                    "SLP_PARENT_WATCH_GRACE_S": "1",
                    "NO_PROXY": "127.0.0.1,localhost",
                    "no_proxy": "127.0.0.1,localhost",
                },
            }, file)

    def stage(self, source, name):
        payload = self.root / name
        payload.mkdir()
        shutil.copy2(source, payload / "slp")
        if (source.parent / "lib").is_dir():
            shutil.copytree(source.parent / "lib", payload / "lib")
        return payload / "slp"

    def point_launcher_at(self, binary):
        next_link = self.root / "slp-next"
        next_link.symlink_to(binary)
        next_link.replace(self.link)

    def start(self):
        checked(launchctl("bootstrap", self.domain, str(self.plist)))

    def stop(self):
        result = launchctl("bootout", self.target)
        if result.returncode and launchctl("print", self.target).returncode == 0:
            checked(result)

    def stop_and_wait(self, pids):
        self.stop()
        deadline = time.monotonic() + 15
        while True:
            loaded = launchctl("print", self.target).returncode == 0
            survivors = [pid for pid in pids if process_running(pid)]
            if not loaded and not survivors:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"test job not fully unloaded: {self.target}, live PIDs: {survivors}"
                )
            time.sleep(0.2)

    def request(self, path, payload=None, timeout=10):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"{path}: expected HTTP 200, got {response.status}")
                return json.load(response)
        except HTTPError as exc:
            with exc:
                body = exc.read(4096).decode(errors="replace")
            raise RuntimeError(f"{path}: HTTP {exc.code}: {body}") from exc

    def wait_for_supervisor(self, binary, old_pid=None):
        deadline = time.monotonic() + 60
        reason = "no launchd PID"
        while time.monotonic() < deadline:
            job = launchctl("print", self.target)
            match = re.search(r"^\s*pid = (\d+)\s*$", job.stdout, re.MULTILINE)
            if job.returncode == 0 and match:
                pid = int(match[1])
                if old_pid is not None and (pid == old_pid or process_running(old_pid)):
                    reason = f"waiting for old PID {old_pid} to exit"
                else:
                    try:
                        port = int((self.home / "server.port").read_text().strip())
                        if not 0 < port <= 65535:
                            raise ValueError(f"invalid port: {port}")
                        self.base_url = f"http://127.0.0.1:{port}"
                        self.request("/health", timeout=2)
                        status = self.request("/agents", timeout=2)
                        if status["supervised"] and status["supervisor"]["pid"] == pid:
                            files = run("/usr/sbin/lsof", "-nP", "-a", "-p", str(pid), "-d", "txt", "-Fn")
                            if f"n{binary}" in files.stdout.splitlines():
                                print(f"launchd PID {pid} is serving {binary.parent.name} on port {port}", flush=True)
                                return pid
                            reason = f"PID {pid} did not map {binary}"
                        else:
                            reason = f"API does not report launchd PID {pid}: {status}"
                    except (OSError, RuntimeError, ValueError, KeyError) as exc:
                        reason = str(exc)
            time.sleep(0.5)
        raise RuntimeError(f"supervisor did not come up for {binary}: {reason}")

    def create_project(self, name):
        cwd = self.root / name
        cwd.mkdir()
        return self.request("/v1/projects", {"name": name, "cwd": str(cwd)})["id"]

    def cold_start(self, project, supervisor_pid):
        def agents():
            status = self.request("/agents")
            if not status["supervised"] or status["supervisor"]["pid"] != supervisor_pid:
                raise RuntimeError(f"unexpected supervisor while probing project: {status}")
            return [agent for agent in status["agents"] if agent["project_id"] == project]

        if agents():
            raise RuntimeError(f"expected project {project} to be cold")
        self.request(f"/v1/projects/{quote(project, safe='')}/models", timeout=60)
        records = agents()
        if len(records) != 1 or not (
            records[0]["alive"] and records[0]["pid"] > 0
            and records[0]["pid"] != supervisor_pid and records[0]["port"] > 0
        ):
            raise RuntimeError(f"expected a separate live project agent: {records}")
        print(f"Cold project {project} started agent PID {records[0]['pid']}", flush=True)
        return records[0]["pid"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path, help="path to the first slp binary")
    parser.add_argument("new", type=Path, help="path to the replacement slp binary")
    args = parser.parse_args()
    if platform.system() != "Darwin" or os.getuid() == 0:
        parser.error("run as a regular user on macOS")
    domain = f"gui/{os.getuid()}"
    if launchctl("print", domain).returncode:
        parser.error("run from a logged-in macOS desktop session")

    with tempfile.TemporaryDirectory(prefix="slp-handoff-", dir="/tmp") as tmp:
        root = Path(tmp).resolve()
        probe = Probe(root, domain)
        pids = []
        try:
            old = probe.stage(args.old.resolve(strict=True), "old")
            new = probe.stage(args.new.resolve(strict=True), "new")
            if launchctl("print", probe.target).returncode == 0:
                raise RuntimeError(f"unexpected existing job: {probe.target}")
            try:
                probe.point_launcher_at(old)
                probe.start()
                old_pid = probe.wait_for_supervisor(old)
                pids.append(old_pid)
                pids.append(probe.cold_start(probe.create_project("before-handoff"), old_pid))
                cold = probe.create_project("after-handoff")

                # Match the installer's order: change the symlink, unload, reload.
                probe.point_launcher_at(new)
                probe.stop_and_wait(pids)
                pids.clear()
                probe.start()
                new_pid = probe.wait_for_supervisor(new, old_pid=old_pid)
                pids.append(new_pid)
                pids.append(probe.cold_start(cold, new_pid))
            finally:
                probe.stop_and_wait(pids)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            logs = []
            for path in (root / "launchd.log", probe.home / "server.log"):
                if path.is_file():
                    logs.append(f"--- {path.name} ---\n{path.read_text(errors='replace')[-6000:]}")
            parser.exit(1, "\n".join([f"Launchd handoff smoke test failed: {exc}", *logs]) + "\n")
    print("Smartloop launchd handoff passed; temporary job and home removed.")


if __name__ == "__main__":
    main()
