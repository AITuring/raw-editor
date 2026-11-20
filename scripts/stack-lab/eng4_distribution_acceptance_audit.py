#!/usr/bin/env python3
"""Audit release signing and Linux distribution acceptance evidence.

Evidence-only: this script never signs, installs, or publishes an artifact.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def command_version(command: str, args: list[str]) -> dict[str, object]:
    path = shutil.which(command)
    if not path:
        return {"available": False, "path": None, "version": None}
    try:
        result = subprocess.run(
            [path, *args], capture_output=True, text=True, check=False, timeout=10
        )
        lines = (result.stdout or result.stderr).strip().splitlines()
        return {
            "available": True,
            "path": path,
            "version": lines[0] if lines else None,
            "exit_code": result.returncode,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": True, "path": path, "version": None, "error": str(exc)}


def package_files() -> list[Path]:
    found: list[Path] = []
    for root in (ROOT / "src-tauri" / "target", ROOT / "output" / "acceptance"):
        if not root.exists():
            continue
        for pattern in ("*.deb", "*.rpm", "*.AppImage"):
            found.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(found))


def workflow_signing_evidence(*texts: str) -> dict[str, bool]:
    joined = "\n".join(texts).lower()
    return {
        "apple_codesign": "codesign" in joined or "apple_identity" in joined,
        "apple_notarization": "notariz" in joined or "notarytool" in joined,
        "apple_stapling": "staple" in joined,
        "windows_signing": "signtool" in joined or "azure_signing" in joined,
        "linux_package_signing": "dpkg-sig" in joined or "rpmsign" in joined,
    }


def main() -> int:
    build = read(".github/workflows/build.yml")
    release = read(".github/workflows/release.yml")
    tauri = read("src-tauri/tauri.conf.json")
    workflow_text = build + "\n" + release
    packages = package_files()
    tools = {
        "file": command_version("file", ["--version"]),
        "dpkg_deb": command_version("dpkg-deb", ["--version"]),
        "rpm": command_version("rpm", ["--version"]),
    }
    signing = workflow_signing_evidence(build, release)
    android_signing = "ANDROID_KEY_BASE64" in workflow_text
    linux_matrix = all(
        marker in workflow_text
        for marker in ("ubuntu-22.04", "ubuntu-24.04", ".deb", ".rpm", ".AppImage")
    )
    tauri_targets_all = '"targets": "all"' in tauri
    explicit_acceptance = any(
        token in workflow_text.lower()
        for token in ("dpkg -i", "apt install", "rpm -i", "appimage", "smoke")
    )
    checks = [
        {
            "name": "Linux build matrix and package upload declared",
            "status": "pass" if linux_matrix and tauri_targets_all else "fail",
            "evidence": "release.yml has Ubuntu matrix and package globs; tauri targets=all",
        },
        {
            "name": "Linux package artifact present in current checkout",
            "status": "pass" if packages else "open",
            "artifacts": [str(path.relative_to(ROOT)) for path in packages],
            "note": "No local package means metadata/install checks cannot be accepted.",
        },
        {
            "name": "Linux package inspection tools",
            "status": "pass"
            if tools["file"]["available"]
            and tools["dpkg_deb"]["available"]
            and tools["rpm"]["available"]
            else "open",
            "tools": tools,
        },
        {
            "name": "Linux install and smoke acceptance",
            "status": "pass" if packages and explicit_acceptance else "open",
            "evidence": "No install/smoke command or acceptance record is present in release workflows.",
        },
        {
            "name": "macOS signing, notarization, and stapling workflow",
            "status": "pass"
            if signing["apple_codesign"]
            and signing["apple_notarization"]
            and signing["apple_stapling"]
            else "open",
            "evidence": signing,
        },
        {
            "name": "Windows signing workflow",
            "status": "pass" if signing["windows_signing"] else "open",
            "evidence": signing,
        },
        {
            "name": "Linux package signing workflow",
            "status": "pass" if signing["linux_package_signing"] else "open",
            "evidence": signing,
        },
        {
            "name": "Signing evidence is limited to Android keystore path",
            "status": "pass"
            if android_signing and not any(signing.values())
            else "open",
            "evidence": {
                "android_keystore_path": android_signing,
                "desktop_signing": signing,
            },
        },
    ]
    result = {
        "schema": 1,
        "task": "ENG.4",
        "method": "release_signing_linux_distribution_evidence_audit_v1",
        "production_impact": "read_only_no_signing_no_install_no_publish",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "android_signing_declared": android_signing,
        "checks": checks,
        "status": "pass" if all(c["status"] == "pass" for c in checks) else "open",
        "decision": "keep signing/notarization/Linux distribution acceptance open until signed artifacts and install/smoke evidence are supplied",
    }
    output = ROOT / "docs/benchmarks/eng4-distribution-acceptance-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "packages": len(packages)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
