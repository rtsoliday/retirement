"""Validate and package the Codex submission draft with Python's standard library."""

import hashlib
import json
from pathlib import Path
import re
import struct
from urllib.parse import urlparse
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PROGRAM = Path(__file__).resolve().parents[1]
SOURCE = PROGRAM / "plugin-submission"
OUTPUT = PROGRAM / "output" / "plugin"
EXPECTED = {".codex-plugin/plugin.json", ".mcp.json", "assets/icon.png", "LICENSE"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_limit(value, maximum, label):
    require(isinstance(value, str) and value.strip(), f"Missing {label}")
    require(len(value) <= maximum, f"{label} exceeds {maximum} characters")
    require(not re.search(r"[\x00-\x08\x0b-\x1f\x7f]", value), f"Control character in {label}")


def https(value, label):
    text_limit(value, 1024, label)
    url = urlparse(value)
    require(url.scheme == "https" and url.hostname and not url.username and not url.password, f"Invalid HTTPS {label}")


def main():
    paths = sorted(p for p in SOURCE.rglob("*") if p.is_file())
    require(not any(p.is_symlink() for p in SOURCE.rglob("*")), "Do not package symlinks")
    entries = {p.relative_to(SOURCE).as_posix(): p.read_bytes() for p in paths}
    require(set(entries) == EXPECTED, f"Unexpected package contents: {set(entries) ^ EXPECTED}")
    manifest = json.loads(entries[".codex-plugin/plugin.json"])
    mcp = json.loads(entries[".mcp.json"])
    require("$schema" not in manifest, "Standalone Codex manifest must omit $schema")
    require(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", manifest["name"]) and len(manifest["name"]) <= 64, "Invalid package name")
    require(re.fullmatch(r"\d+\.\d+\.\d+", manifest["version"]), "Use a semantic release version")
    text_limit(manifest["description"], 4000, "Package description")
    text_limit(manifest["author"]["name"], 120, "Author")
    require(manifest["id"] == "plugin_asdk_app_sites_88da36bd37fc81918f7671ab31340ec8", "Preserve the existing plugin identity")
    require(manifest["mcpServers"] == "./.mcp.json", "Incorrect MCP config reference")
    ui = manifest["interface"]
    for field, limit in (("displayName", 30), ("shortDescription", 30), ("longDescription", 4000), ("developerName", 80), ("category", 120)):
        text_limit(ui[field], limit, field)
    for field in ("websiteURL", "supportURL", "privacyPolicyURL", "termsOfServiceURL"):
        https(ui[field], field)
    require(len(ui["capabilities"]) <= 20, "Too many capabilities")
    for capability in ui["capabilities"]:
        text_limit(capability, 120, "Capability")
    prompts = ui["defaultPrompt"]
    require(1 <= len(prompts) <= 3 and len(set(prompts)) == len(prompts), "Invalid starter prompt count or duplicates")
    for prompt in prompts:
        text_limit(prompt, 128, "Starter prompt")
        require("@" not in prompt and "\n" not in prompt, "Invalid starter prompt")
    for field in ("logo", "composerIcon"):
        require(ui[field].startswith("./") and ui[field][2:] in entries, f"Missing {field} asset")
        data = entries[ui[field][2:]]
        require(data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR", "Expected PNG artwork")
        width, height = struct.unpack(">II", data[16:24])
        require(width == height and 48 <= width <= 4096 and len(data) <= 5 * 1024 * 1024, "Invalid icon size")
    require(entries["assets/icon.png"] == (PROGRAM / "dist" / "apple-touch-icon.png").read_bytes(), "Icon differs from existing website artwork")
    require(entries["LICENSE"] == (PROGRAM.parents[1] / "LICENSE").read_bytes(), "License differs from repository notice")
    servers = mcp["mcpServers"]
    require(set(servers) == {"retirement_forecast"}, "Exactly one existing remote server is expected")
    require(servers["retirement_forecast"] == {"url": "https://retirement-readiness-lab-web.rtsoliday123.chatgpt.site/mcp"}, "Preserve the deployed endpoint and omit credentials")
    extension = manifest["extensions"]["com.openai"]
    review = extension["review"]
    require("test_credentials" not in review and "reviewer_instructions" not in review, "Enter reviewer access privately in the dashboard")
    require(review["commerce"] is False, "This package does not initiate commerce")
    cases = review["test_cases"]
    require(len(cases["positive"]) == 5 and len(cases["negative"]) == 3, "Expected five positive and three negative review cases")
    known_tools = {"explain_forecast_methodology", "create_retirement_forecast", "compare_retirement_scenarios"}
    for kind, group in cases.items():
        for case in group:
            for field in ("description", "prompt", "expected_behavior"):
                text_limit(case[field], 4000, f"{kind} case {field}")
            if kind == "positive":
                require(set(case["tools_triggered"].split(", ")) <= known_tools, "Unknown review tool")
                require(case["tools_triggered"], "Positive case has no tools")
    require(extension["publication"]["countries"] == ["US"], "Review U.S. draft availability before changing it")
    for name, data in entries.items():
        if name.endswith(".json"):
            source = data.decode("utf-8")
            require(not re.search(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}|-----BEGIN .*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._-]{12,}", source), f"Potential credential in {name}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / f"{manifest['name']}-{manifest['version']}.zip"
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = ZipInfo(name, (2026, 10, 5, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    with ZipFile(destination) as archive:
        require(archive.testzip() is None, "ZIP CRC check failed")
        require(set(archive.namelist()) == EXPECTED, "ZIP is missing a dot file or includes extra files")
        require(all(archive.read(name) == data for name, data in entries.items()), "Archive content changed")
    result = {
        "package": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
        "entries": sorted(entries),
        "iconPixels": [width, height],
        "listingLengths": {key: len(ui[key]) for key in ("displayName", "shortDescription", "longDescription")},
        "starterPromptLengths": [len(p) for p in prompts],
        "reviewCases": {key: len(value) for key, value in cases.items()},
        "category": ui["category"],
        "status": "locally validated draft; not uploaded or submitted",
        "pending": ["publisher and listing review", "dashboard acceptance of Finance category", "reviewer account and review-case runs", "demo recording URL", "developer/domain verification", "dashboard connection, scans and policy review"]
    }
    (OUTPUT / "package-validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
