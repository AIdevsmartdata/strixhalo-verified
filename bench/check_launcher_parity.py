#!/usr/bin/env python3
"""check_launcher_parity.py -- check that launch/qwen38-flash-next.sh runs the configuration of a production launcher.

Usage: python3 bench/check_launcher_parity.py PRODUCTION_LAUNCHER [PUBLIC_LAUNCHER]
       (PUBLIC_LAUNCHER defaults to launch/qwen38-flash-next.sh next to this directory)

Neither script is run. Both are parsed (continuation lines joined, shell quoting removed) and compared:
  - every `export NAME=VALUE`: same names, same values, same order;
  - the llama-server arguments: same tokens, same order.
The only differences allowed:
  - the values of LLAMA_MTP_HEAD_IDS and LD_LIBRARY_PATH (paths);
  - the values of -m, --mmproj, -md and --cache-dir (paths, which become $MODEL, $MMPROJ, $MTP and $CACHE_DIR);
  - --chat-template-file and its value, present in production only;
  - --port, which is "$PORT" in the public launcher, whose default must be the production port;
  - a trailing "$@" in the public launcher (extra arguments passed through).
It also checks that the public launcher keeps the GPU clock floor and holds no absolute path outside a short allowlist.
Absolute paths of the production launcher are printed as <path>. Exit status: 0 if everything matches, 1 otherwise.
"""
import os
import re
import shlex
import sys

ENV_PATHS = {  # name -> value expected in the public launcher
    "LLAMA_MTP_HEAD_IDS": "$MTP_IDS",
    "LD_LIBRARY_PATH": "$RUNTIME:${ROCM_PATH:-/opt/rocm}/lib",
}
ARG_PATHS = {"-m": "$MODEL", "--mmproj": "$MMPROJ", "-md": "$MTP", "--cache-dir": "$CACHE_DIR"}
PROD_ONLY_ARGS = {"--chat-template-file"}  # each takes one value
PUBLIC_PATH_ALLOWLIST = ("/path/", "/opt/rocm", "/sys/class/drm/", "/bin/bash", "/dev/null")


def logical_lines(text):
    """Shell lines with backslash-newline continuations joined."""
    out, cur = [], ""
    for raw in text.splitlines():
        if raw.endswith("\\"):
            cur += raw[:-1] + " "
            continue
        out.append(cur + raw)
        cur = ""
    if cur:
        out.append(cur)
    return out


def parse(path):
    text = open(path, encoding="utf-8").read()
    exports, args, port_default = [], None, None
    for line in logical_lines(text):
        s = line.strip()
        if s.startswith("export "):
            tok = shlex.split(s, comments=True)
            for t in tok[1:]:
                name, _, value = t.partition("=")
                exports.append((name, value))
        elif s.startswith("exec ") and "llama-server" in s:
            tok = shlex.split(s, comments=True)
            k = next(i for i, t in enumerate(tok) if t.endswith("llama-server"))
            args = tok[k + 1:]
        m = re.match(r"PORT=\$\{PORT:-([^}]*)\}", s)
        if m:
            port_default = m.group(1)
    if args is None:
        sys.exit(f"{path}: no 'exec ... llama-server' line found")
    return text, exports, args, port_default


def shown(value):
    return "<path>" if value.startswith("/") or ":/" in value else value


def compare_env(prod, pub, problems, allowed):
    pn, qn = [n for n, _ in prod], [n for n, _ in pub]
    if pn != qn:
        missing = [n for n in pn if n not in qn]
        extra = [n for n in qn if n not in pn]
        problems.append(f"export names differ: missing {missing}, extra {extra}"
                        + ("" if missing or extra else " (same set, different order)"))
    pub_d = dict(pub)
    for name, pval in prod:
        if name not in pub_d:
            continue
        qval = pub_d[name]
        if name in ENV_PATHS:
            if qval != ENV_PATHS[name]:
                problems.append(f"export {name}: public value {qval!r}, expected {ENV_PATHS[name]!r}")
            else:
                allowed.append(f"export {name}: {shown(pval)} -> {qval}")
        elif pval != qval:
            problems.append(f"export {name}: production {shown(pval)!r}, public {qval!r}")


def compare_args(prod, pub, port_default, problems, allowed):
    i = j = 0
    while i < len(prod):
        p = prod[i]
        if p in PROD_ONLY_ARGS:
            allowed.append(f"arg {p} {shown(prod[i + 1])}: production only")
            i += 2
            continue
        q = pub[j] if j < len(pub) else None
        if q != p:
            problems.append(f"arg #{i}: production {shown(p)!r}, public {q!r}")
            i += 1
            j += 1
            continue
        if p in ARG_PATHS:
            pv, qv = prod[i + 1], pub[j + 1] if j + 1 < len(pub) else None
            if qv != ARG_PATHS[p]:
                problems.append(f"arg {p}: public value {qv!r}, expected {ARG_PATHS[p]!r}")
            else:
                allowed.append(f"arg {p}: {shown(pv)} -> {qv}")
            i += 2
            j += 2
            continue
        if p == "--port":
            pv, qv = prod[i + 1], pub[j + 1] if j + 1 < len(pub) else None
            if qv != "$PORT":
                problems.append(f"arg --port: public value {qv!r}, expected '$PORT'")
            elif port_default != pv:
                problems.append(f"arg --port: production {pv}, public default {port_default}")
            else:
                allowed.append(f"arg --port: {pv} -> $PORT (default {port_default})")
            i += 2
            j += 2
            continue
        i += 1
        j += 1
    rest = pub[j:]
    if rest == ["$@"]:
        allowed.append('arg "$@": public only (extra arguments, appended last)')
    elif rest:
        problems.append(f"public launcher has extra arguments {rest}")


def check_public_text(text, problems):
    if "power_dpm_force_performance_level" not in text:
        problems.append("public launcher lost the GPU clock floor (power_dpm_force_performance_level)")
    bad = set()
    for m in re.finditer(r"(?:^|[\s\"'=:(-])(/[A-Za-z0-9_.][^\s\"'):;,]*)", text, re.M):
        path = m.group(1)
        if not path.startswith(PUBLIC_PATH_ALLOWLIST):
            bad.add(path)
    for path in sorted(bad):
        problems.append(f"public launcher holds an absolute path outside the allowlist: {path}")


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    prod_path = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    pub_path = sys.argv[2] if len(sys.argv) == 3 else os.path.join(here, "..", "launch", "qwen38-flash-next.sh")
    _, prod_env, prod_args, _ = parse(prod_path)
    pub_text, pub_env, pub_args, port_default = parse(pub_path)

    problems, allowed = [], []
    compare_env(prod_env, pub_env, problems, allowed)
    compare_args(prod_args, pub_args, port_default, problems, allowed)
    check_public_text(pub_text, problems)

    same_env = sum(1 for (n, v) in prod_env if (n, v) in pub_env)
    print(f"exports: production {len(prod_env)}, public {len(pub_env)}, identical name and value {same_env}")
    print(f"llama-server tokens: production {len(prod_args)}, public {len(pub_args)}")
    print("allowed differences:")
    for a in allowed:
        print(f"  {a}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  {p}")
        print("RESULT: MISMATCH")
        return 1
    print("RESULT: PARITY (same environment and same llama-server arguments, apart from the differences above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
