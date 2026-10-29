#!/usr/bin/env python3
"""Regenerate the module map in docs/source-layout.md.

    python3 scripts/source-layout.py          # print the generated section
    python3 scripts/source-layout.py --write  # replace it in docs/source-layout.md

Rust: every tracked `x.rs` with a sibling `x/` directory (and `lib.rs` with
`lib/`) is a facade; its direct child files are listed with line count,
whether they are compiled only for tests, and the first line of their `//!`
documentation (test modules moved out of a facade without a header get a
derived description). TypeScript: `Name.ts(x)` with an exactly-named sibling
`Name/` directory, listed with each part's exported names. CSS: stylesheets
that `@import` more than one local part. Finally every tracked source file
over 1000 lines.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOC = os.path.join(ROOT, "docs", "source-layout.md")
BEGIN = "<!-- source-layout:begin -->"
END = "<!-- source-layout:end -->"
RUST_SRC = "src-tauri/src/"


def tracked_files():
    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, check=True).stdout
    return sorted(line for line in out.splitlines() if line and os.path.exists(os.path.join(ROOT, line)))


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as handle:
        return handle.read()


def line_count(path):
    return read(path).count("\n")


def doc_line(path):
    for line in read(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("//!"):
            text = stripped[3:].strip()
            if text:
                return text.replace("|", "\\|")
        elif stripped and not stripped.startswith("#!["):
            break
    return ""


def declaration_is_test_only(parent_text, stem):
    attributes = r"(?:#\[[^\]]*\]\s*)*"
    plain = re.compile(
        r"#\[cfg\(test\)\]\s*" + attributes + r"(?:pub(?:\([^)]*\))?\s+)?mod\s+" + re.escape(stem) + r"\s*;")
    by_path = re.compile(
        r"#\[cfg\(test\)\]\s*" + attributes + r"#\[path\s*=\s*\"(?:[^\"]*/)?" + re.escape(stem) + r"\.rs\"\]")
    return bool(plain.search(parent_text) or by_path.search(parent_text))


def facade_of(child):
    """The file that declares `child` (`a/b.rs` for `a/b/c.rs`, `lib.rs` for `lib/c.rs`)."""
    directory = os.path.dirname(child)
    if directory == RUST_SRC + "lib":
        return RUST_SRC + "lib.rs"
    return directory + ".rs"


def declaring_files(path):
    """Files that may declare `path`: its facade, or (for #[path] siblings) files in the same directory."""
    candidates = [facade_of(path)]
    directory = os.path.dirname(path)
    candidates += [os.path.join(directory, name) for name in ("mod.rs", "lib.rs")]
    candidates += [os.path.join(directory, name) for name in sorted(os.listdir(os.path.join(ROOT, directory)))
                   if name.endswith(".rs")]
    return [c for c in candidates if c != path and os.path.isfile(os.path.join(ROOT, c))]


def is_test_only(path, seen=None):
    seen = set() if seen is None else seen
    if path in seen:
        return False
    seen.add(path)
    if re.match(r"\s*#!\[cfg\(test\)\]", read(path)):
        return True
    stem = os.path.splitext(os.path.basename(path))[0]
    for parent in declaring_files(path):
        if declaration_is_test_only(read(parent), stem):
            return True
    parent = facade_of(path)
    if os.path.isfile(os.path.join(ROOT, parent)) and parent != path and not path.endswith("/lib.rs"):
        mentions = re.search(r"\bmod\s+" + re.escape(stem) + r"\s*;", read(parent))
        if mentions and is_test_only(parent, seen):
            return True
    return False


def derived_description(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    if "test_access" in stem:
        return f"测试访问入口（原内联 `mod {stem}`）"
    if stem == "tests" or stem.endswith("_tests"):
        return f"单元测试（原内联 `mod {stem}`）"
    return ""


def rust_section(files):
    facades = []
    for path in files:
        if not path.startswith(RUST_SRC) or not path.endswith(".rs"):
            continue
        directory = RUST_SRC + "lib" if path == RUST_SRC + "lib.rs" else path[:-3]
        children = [f for f in files if f.endswith(".rs") and os.path.dirname(f) == directory]
        if children:
            facades.append((path, children))
    out = ["### Rust：facade 与子模块", ""]
    for facade, children in facades:
        rel = facade[len(RUST_SRC):]
        tag = "，仅测试" if is_test_only(facade) else ""
        out.append(f"#### `{rel}`（facade {line_count(facade):,} 行{tag}，{len(children)} 个子模块）")
        out.append("")
        out.append("| 子模块 | 行数 | 仅测试 | 说明 |")
        out.append("| --- | ---: | :---: | --- |")
        for child in children:
            test_only = is_test_only(child)
            nested = os.path.isdir(os.path.join(ROOT, child[:-3]))
            name = os.path.basename(child) + ("（另见下方同名 facade）" if nested else "")
            description = doc_line(child) or derived_description(child)
            out.append(f"| `{name}` | {line_count(child):,} | {'✓' if test_only else ''} | {description} |")
        out.append("")
    return out


def ts_exports(path):
    text = read(path)
    names = re.findall(
        r"^export\s+(?:default\s+)?(?:declare\s+)?(?:async\s+)?(?:function|const|let|class|interface|type|enum)\s+(\w+)",
        text, re.M)
    for group in re.findall(r"^export\s+(?:type\s+)?\{([^}]*)\}", text, re.M):
        names += [n.strip().split(" as ")[-1] for n in group.split(",") if n.strip()]
    return names


def ts_section(files):
    out = ["### 前端：文件与同名目录下的 parts", ""]
    for path in files:
        if not path.startswith("src/") or not re.search(r"\.tsx?$", path):
            continue
        base = os.path.splitext(path)[0]
        parent = os.path.join(ROOT, os.path.dirname(base))
        if os.path.basename(base) not in os.listdir(parent):
            continue
        parts = [f for f in files if f.startswith(base + "/") and re.search(r"\.tsx?$", f)]
        if not parts:
            continue
        out.append(f"#### `{path}`（{line_count(path):,} 行，{len(parts)} 个 part）")
        out.append("")
        out.append("| part | 行数 | 导出 |")
        out.append("| --- | ---: | --- |")
        for part in parts:
            names = ", ".join(f"`{n}`" for n in ts_exports(part))
            out.append(f"| `{part[len(base) + 1:]}` | {line_count(part):,} | {names} |")
        out.append("")
    for path in files:
        if not path.startswith("src/") or not path.endswith(".css"):
            continue
        imports = re.findall(r"^@import\s+['\"](\.{1,2}/[^'\"]+\.css)['\"]", read(path), re.M)
        parts = [os.path.normpath(os.path.join(os.path.dirname(path), target)) for target in imports]
        if len(parts) < 2:
            continue
        out.append(f"#### `{path}`（{line_count(path):,} 行，按顺序 `@import` {len(parts)} 个文件）")
        out.append("")
        out.append("| 文件 | 行数 |")
        out.append("| --- | ---: |")
        for part in parts:
            out.append(f"| `{part}` | {line_count(part):,} |")
        out.append("")
    return out


def long_files_section(files):
    rows = []
    for path in files:
        if re.search(r"\.(rs|ts|tsx|js|jsx|mjs|css)$", path):
            count = line_count(path)
            if count > 1000:
                rows.append((count, path))
    out = ["### 仍超过 1000 行的源文件", "", "| 文件 | 行数 |", "| --- | ---: |"]
    out += [f"| `{path}` | {count:,} |" for count, path in sorted(rows, reverse=True)]
    out.append("")
    return out


def generate():
    files = tracked_files()
    return "\n".join(rust_section(files) + ts_section(files) + long_files_section(files)).rstrip() + "\n"


def main():
    section = generate()
    if "--write" not in sys.argv:
        sys.stdout.write(section)
        return
    text = read(DOC)
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    with open(DOC, "w", encoding="utf-8") as handle:
        handle.write(text[:start] + "\n\n" + section + "\n" + text[end:])
    # Keep the document in the repository's Prettier style (aligned tables).
    prettier = os.path.join(ROOT, "node_modules", ".bin", "prettier")
    if os.path.exists(prettier):
        subprocess.run([prettier, "--write", DOC], check=True, capture_output=True)


if __name__ == "__main__":
    main()
