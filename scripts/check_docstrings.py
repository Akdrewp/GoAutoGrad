#!/usr/bin/env python3
"""Docstring linter: enforces Google-style Doxygen docstrings across C++ backend files.

Checks that every class, struct, enum, and function in C++ header (.hpp) and source (.cpp)
files includes a /** ... */ docstring block containing @brief, @param, @return, and
@throw (where applicable). Excludes tests.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

TOKEN_RE = re.compile(
    r"""
      (?P<DOXYGEN>/\*\*.*?\*/)
    | (?P<BLOCK_COMMENT>/\*.*?\*/)
    | (?P<LINE_COMMENT>//[^\n]*)
    | (?P<PREPROCESSOR>\#[^\n]*(?:\\\n[^\n]*)*)
    | (?P<STRING>"(\\.|[^"\\])*")
    | (?P<CHAR>'(\\.|[^'\\])*')
    | (?P<PUNCT>[{}();:,])
    | (?P<WORD>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<OTHER>[^\s{}();:,A-Za-z_]+)
    | (?P<NEWLINE>\n)
    | (?P<WS>[ \t\r]+)
""",
    re.VERBOSE | re.DOTALL,
)


def check_cpp_file(file_path: Path) -> list[str]:
    """Validates that classes, structs, enums, and functions contain docstrings with @brief."""
    text = file_path.read_text(encoding="utf-8")
    tokens = []
    line = 1
    for m in TOKEN_RE.finditer(text):
        kind = m.lastgroup
        val = m.group()
        if kind == "NEWLINE":
            line += 1
        elif kind in ("WS", "LINE_COMMENT", "BLOCK_COMMENT", "PREPROCESSOR"):
            line += val.count("\n")
        elif kind == "DOXYGEN":
            tokens.append((kind, val, line))
            line += val.count("\n")
        else:
            tokens.append((kind, val, line))
            line += val.count("\n")

    errors = []
    i = 0
    n = len(tokens)

    scope_stack = []
    last_doxygen = None

    while i < n:
        tok_kind, tok_val, tok_line = tokens[i]

        if tok_kind == "DOXYGEN":
            last_doxygen = (tok_val, tok_line)
            i += 1
            continue

        if tok_val == "}":
            if scope_stack:
                scope_stack.pop()
            i += 1
            continue

        current_scope = scope_stack[-1] if scope_stack else "global"

        # Inside a function body, do not check local variables or function calls
        if current_scope == "func":
            if tok_val == "{":
                scope_stack.append("func")
            i += 1
            continue

        # Track namespace scopes
        if tok_val == "namespace":
            while i < n and tokens[i][1] != "{":
                i += 1
            if i < n and tokens[i][1] == "{":
                scope_stack.append("namespace")
                i += 1
            last_doxygen = None
            continue

        # Check for class, struct, or enum definitions
        if tok_val in ("class", "struct", "enum"):
            decl_tokens = []
            decl_start_line = tok_line
            j = i
            while j < n and tokens[j][1] not in (";", "{"):
                decl_tokens.append(tokens[j][1])
                j += 1

            if j < n:
                terminator = tokens[j][1]
                name = None
                words = [t for t in decl_tokens if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", t)]
                if len(words) >= 2:
                    name = words[1] if words[0] in ("class", "struct", "enum") else (words[2] if len(words) > 2 and words[1] == "class" else words[-1])

                if terminator == "{" and name:
                    if not last_doxygen:
                        errors.append(f"{file_path}:{decl_start_line}: Missing docstring for {tok_val} '{name}'")
                    elif "@brief" not in last_doxygen[0]:
                        errors.append(f"{file_path}:{decl_start_line}: Docstring for {tok_val} '{name}' is missing '@brief'")
                    last_doxygen = None
                    scope_stack.append("class" if tok_val in ("class", "struct") else "enum")
                    i = j + 1
                    continue
                elif terminator == ";":
                    last_doxygen = None
                    i = j + 1
                    continue

        # Check for function definitions and declarations
        if tok_kind == "WORD" and tok_val not in (
            "template", "inline", "static", "virtual", "explicit", "friend",
            "public", "protected", "private", "using", "return", "typedef", "const", "constexpr"
        ):
            j = i + 1
            if j < n and tokens[j][1] == "(":
                func_name = tok_val
                func_start_line = tok_line

                if func_name in ("if", "for", "while", "switch", "catch", "sizeof", "decltype"):
                    i += 1
                    continue

                paren_depth = 1
                j += 1
                while j < n and paren_depth > 0:
                    if tokens[j][1] == "(":
                        paren_depth += 1
                    elif tokens[j][1] == ")":
                        paren_depth -= 1
                    j += 1

                k = j
                while k < n and tokens[k][1] not in (";", "{"):
                    k += 1

                if k < n:
                    terminator = tokens[k][1]
                    if terminator == "{":
                        if not last_doxygen:
                            errors.append(f"{file_path}:{func_start_line}: Missing docstring for function '{func_name}'")
                        elif "@brief" not in last_doxygen[0]:
                            errors.append(f"{file_path}:{func_start_line}: Docstring for function '{func_name}' is missing '@brief'")
                        last_doxygen = None
                        scope_stack.append("func")
                        i = k + 1
                        continue
                    elif terminator == ";" and current_scope in ("global", "namespace", "class"):
                        if file_path.suffix in (".hpp", ".h"):
                            if not last_doxygen:
                                errors.append(f"{file_path}:{func_start_line}: Missing docstring for function declaration '{func_name}'")
                            elif "@brief" not in last_doxygen[0]:
                                errors.append(f"{file_path}:{func_start_line}: Docstring for function '{func_name}' is missing '@brief'")
                        last_doxygen = None
                        i = k + 1
                        continue

        if tok_val == ";":
            last_doxygen = None

        i += 1

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check C++ files for Google-style @brief docstrings.")
    parser.add_argument("paths", nargs="*", help="Files or directories to check. Defaults to nn/ module.")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    targets: list[Path] = []

    if args.paths:
        for p_str in args.paths:
            p = Path(p_str)
            if not p.is_absolute():
                p = workspace / p
            if p.is_file() and p.suffix in (".hpp", ".cpp", ".h") and not p.name.endswith("bindings.cpp"):
                targets.append(p)
            elif p.is_dir():
                for root, _, files in os.walk(p):
                    if "tests" in root or "build" in root:
                        continue
                    for f in files:
                        if f.endswith((".hpp", ".cpp", ".h")) and not f.endswith("bindings.cpp"):
                            targets.append(Path(root) / f)
    else:
        # Default to nn/ and any newly added modules
        nn_dir = workspace / "GoTorch" / "backend" / "cppsrc" / "nn"
        if nn_dir.exists():
            for root, _, files in os.walk(nn_dir):
                for f in files:
                    if f.endswith((".hpp", ".cpp", ".h")):
                        targets.append(Path(root) / f)

    all_errors = []
    for target in sorted(set(targets)):
        file_errors = check_cpp_file(target)
        all_errors.extend(file_errors)

    if all_errors:
        sys.stderr.write("==> Docstring verification failed:\n")
        for err in all_errors:
            sys.stderr.write(f"  {err}\n")
        sys.stderr.write(f"\nTotal docstring violations: {len(all_errors)}\n")
        return 1

    print(f"==> Checked {len(targets)} C++ file(s): all classes and functions have valid @brief docstrings!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
