"""Verify that a built slp can start its own project agent without weights."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


def smoke(binary, root, *, move_old_payload=False):
    home = root / "smartloop"
    executable = binary
    if move_old_payload:
        old_payload = root / "old-version"
        new_payload = root / "new-version"
        for payload in (old_payload, new_payload):
            payload.mkdir()
            shutil.copy2(binary, payload / "slp")
            if (binary.parent / "lib").is_dir():
                shutil.copytree(binary.parent / "lib", payload / "lib")
        executable = root / "slp"
        executable.symlink_to(old_payload / "slp")
    # In particular, leave SLP_EXE unset: exercise the supervisor's default re-exec.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("SLP_")}
    env.update(
        HOME=str(root),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_CACHE_HOME=str(root / "cache"),
        XDG_DATA_HOME=str(root / "data"),
        SLP_HOME=str(home),
        SLP_API_PORT="0",
        SLP_API_PRELOAD_MODEL="0",
        NO_PROXY="localhost,127.0.0.1",
        no_proxy="localhost,127.0.0.1",
    )
    opener = build_opener(ProxyHandler({}))

    def request(base, path, payload=None, timeout=10, *, allow_503=False):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"{path}: expected HTTP 200, got {response.status}")
                body = json.load(response)
                return (response.status, body) if allow_503 else body
        except HTTPError as exc:
            with exc:
                body = exc.read(4096).decode(errors="replace")
            if allow_503 and exc.code == 503:
                return exc.code, json.loads(body)
            raise RuntimeError(f"{path}: HTTP {exc.code}: {body}") from exc

    with (root / "backend.log").open("wb") as log:
        server = subprocess.Popen(
            [str(executable), "agent", "start", "--no-service", "--host", "127.0.0.1"],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"slp exited during startup: {server.returncode}")
                try:
                    port = int((home / "server.port").read_text().strip())
                    if not 0 < port <= 65535:
                        raise ValueError(f"invalid port: {port}")
                    base = f"http://127.0.0.1:{port}"
                    request(base, "/health", timeout=2)
                    break
                except (FileNotFoundError, ValueError, URLError, TimeoutError):
                    time.sleep(0.2)
            else:
                raise RuntimeError("slp did not become healthy within 30 seconds")

            def create_project(name):
                project = root / name
                project.mkdir()
                return request(base, "/v1/projects", {"name": name, "cwd": str(project)})["id"]

            def models(project_id, *, allow_503=False):
                return request(
                    base, f"/v1/projects/{quote(project_id, safe='')}/models",
                    timeout=60, allow_503=allow_503,
                )

            def agent_pids():
                status = request(base, "/agents")
                if not status["supervised"] or status["supervisor"]["pid"] != server.pid:
                    raise RuntimeError(f"expected the original test supervisor: {status}")
                pids = {}
                for agent in status["agents"]:
                    pid = agent["pid"]
                    project_id = agent["project_id"]
                    if not (agent["alive"] and pid > 0 and pid != server.pid and agent["port"] > 0):
                        raise RuntimeError(f"expected a live child agent: {agent}")
                    if project_id in pids or pid in pids.values():
                        raise RuntimeError(f"expected distinct project agents: {status}")
                    pids[project_id] = pid
                return pids

            warm = create_project("project-a")
            cold = create_project("project-b") if move_old_payload else None
            if agent_pids():
                raise RuntimeError("expected no project agents before the first models request")

            models(warm)
            before = agent_pids()
            if before.keys() != {warm}:
                raise RuntimeError(f"expected only the first project agent: {before}")
            if not move_old_payload:
                print("Smartloop cold project-agent launch passed")
                return

            next_link = root / "slp-next"
            next_link.symlink_to(new_payload / "slp")
            next_link.replace(executable)
            moved_payload = root / "moved-old-version"
            old_payload.rename(moved_payload)
            if (old_payload / "slp").exists() or executable.resolve(strict=True) != new_payload / "slp":
                raise RuntimeError("expected the original path gone and the launcher pointing to the new copy")

            models(warm)
            if agent_pids() != before:
                raise RuntimeError("warm project did not retain its original agent")

            status, body = models(cold, allow_503=True)
            if status == 503:
                detail = body.get("detail", "") if isinstance(body, dict) else ""
                if not isinstance(detail, str) or not (
                    f"could not spawn an agent for project {cold}" in detail and "(os error 2)" in detail
                ):
                    raise RuntimeError(f"unexpected cold-project HTTP 503: {body}")
                if agent_pids() != before:
                    raise RuntimeError("a failed cold start changed the live agent registry")
                print(f"Cold project reproduced the spawn 503: {detail}", flush=True)
                moved_payload.rename(old_payload)
                models(cold)
                print("Cold project recovered after restoring the original executable path")
            else:
                print("Cold project returned HTTP 200 after the symlink swap and payload move")

            after = agent_pids()
            if after.keys() != {warm, cold} or after[warm] != before[warm]:
                raise RuntimeError(f"expected both project agents, with the warm PID unchanged: {after}")
            print("Warm project kept its original agent; moved-payload diagnostic completed")
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument(
        "--move-old-payload",
        action="store_true",
        help="diagnose cold starts after a disposable installer-style symlink swap and payload move",
    )
    args = parser.parse_args()
    try:
        binary = args.binary.resolve(strict=True)
    except OSError as exc:
        parser.error(str(exc))

    # Keep the model host's Unix socket path short, including on macOS CI.
    with tempfile.TemporaryDirectory(prefix="slp-smoke-", dir="/tmp") as tmp:
        root = Path(tmp).resolve()
        try:
            smoke(binary, root, move_old_payload=args.move_old_payload)
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
            logs = []
            for path in (root / "backend.log", root / "smartloop" / "server.log"):
                if path.is_file():
                    logs.append(f"--- {path.name} ---\n{path.read_text(errors='replace')[-6000:]}")
            parser.exit(1, "\n".join([f"Smoke test failed: {exc}", *logs]) + "\n")


if __name__ == "__main__":
    main()
