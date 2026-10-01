#!/usr/bin/env python3
"""PostToolUse hook: synchronizes C++ build and Python stubs when GoTorch implementation changes."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    # 1. Parse JSON payload from stdin
    payload = {}
    try:
        raw_input = sys.stdin.read()
        if raw_input.strip():
            payload = json.loads(raw_input)
    except Exception as e:
        sys.stderr.write(f"[Hook Warning] Failed to parse stdin JSON: {e}\n")

    # 2. Determine workspace directory
    workspace_paths = payload.get("workspacePaths", [])
    if workspace_paths and os.path.isdir(workspace_paths[0]):
        workspace_dir = Path(workspace_paths[0])
    else:
        workspace_dir = Path(__file__).resolve().parent.parent.parent

    gotorch_dir = workspace_dir / "GoTorch"
    cpp_dir = gotorch_dir / "backend" / "cppsrc"
    sync_marker = workspace_dir / ".agents" / ".last_stubs_sync"

    last_sync = sync_marker.stat().st_mtime if sync_marker.exists() else 0.0

    # 3. Check for modified files in GoTorch
    modified_gotorch = []
    modified_cpp = []

    for root, _, files in os.walk(gotorch_dir):
        for f in files:
            if f.endswith((".py", ".cpp", ".hpp", ".h", ".cc")):
                f_path = Path(root) / f
                try:
                    mtime = f_path.stat().st_mtime
                    if mtime > last_sync:
                        modified_gotorch.append(f_path)
                        if cpp_dir in f_path.parents or f_path.parent == cpp_dir:
                            modified_cpp.append(f_path)
                except OSError:
                    continue

    if not modified_gotorch:
        # No files changed since last sync
        print(json.dumps({}))
        return

    sys.stderr.write(
        f"[Hook] Detected {len(modified_gotorch)} changed GoTorch file(s).\n"
    )

    # 4. If C++ files changed, trigger make build
    if modified_cpp:
        sys.stderr.write("[Hook] C++ backend source modified. Running 'make build'...\n")
        try:
            build_res = subprocess.run(
                ["make", "build"],
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
            )
            if build_res.returncode != 0:
                sys.stderr.write(f"[Hook Error] 'make build' failed:\n{build_res.stderr}\n")
            else:
                sys.stderr.write("[Hook] 'make build' succeeded.\n")
        except Exception as e:
            sys.stderr.write(f"[Hook Error] Could not run make build: {e}\n")

    # 5. Run stubgen
    sys.stderr.write("[Hook] Regenerating stubs via stubgen...\n")
    stubgen_bin = workspace_dir / ".venv" / "bin" / "stubgen"
    if not stubgen_bin.exists():
        stubgen_bin = Path("stubgen")

    try:
        stub_res = subprocess.run(
            [
                str(stubgen_bin),
                "-p",
                "GoTorch",
                "-o",
                "tests/stubs",
                "--include-docstrings",
            ],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
        )
        if stub_res.returncode != 0:
            sys.stderr.write(f"[Hook Error] stubgen failed:\n{stub_res.stderr}\n")
        else:
            sys.stderr.write("[Hook] Stubs successfully updated in tests/stubs/GoTorch/.\n")
    except Exception as e:
        sys.stderr.write(f"[Hook Error] Could not run stubgen: {e}\n")

    # 6. Update sync marker timestamp
    sync_marker.parent.mkdir(parents=True, exist_ok=True)
    sync_marker.touch()

    # 7. Output empty JSON object as required by PostToolUse contract
    print(json.dumps({}))


if __name__ == "__main__":
    main()
