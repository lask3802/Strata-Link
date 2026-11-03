#!/usr/bin/env python3
"""tools/dsv41/remote_sessions_test.py - one remote stage worker, several main processes one after another: does every
session print the first one's tokens?  On one Linux PC and one GPU, over the loopback, as remote_slots_test.py runs
the split (a worker --stage-worker, layers K to the last; a main process --remote-stage 127.0.0.1:PORT --stage-end K,
serve).  It needs a GPU.

    STRATA_STAGE_TLS=DIR python3 tools/dsv41/remote_sessions_test.py --exe build/strata-dsv41 --pack PACK --out /tmp/rss
        [--ids-dir DIR] [--k 20] [--port 27941] [--ctx 20000] [--gpu 0] [--sessions 4] [--max-new 24]
        [--engine-args "--vram-experts 0.3 --mmap-experts on --ssd-cache 0 --no-expert-cache"]
        [--main-args "..."] [--worker-args "..."] [--main-env K=V ...] [--worker-env K=V ...] [--fresh-worker]

Each session is one main process that reads one prompt (the shared system prompt + agent A's task, slot_keep_test.py's
prompts, the shared start pinned as the server sends it) and decodes --max-new greedy tokens, reuse and slots off, then
quits.  The worker stays up across the sessions (--fresh-worker: a new worker before each one).  Printed for each
session, besides its tokens against session 1's:
  - the GPU's memory in use before the main process starts, and the worker's own (nvidia-smi), so that a worker that
    grew after its first session shows;
  - the main process's placement as its INFO line says it (expert_slots, cpu_experts, ram_experts): with a share of
    the free memory (--vram-experts F) a main process that starts beside a bigger worker places fewer experts on the
    GPU, and the experts the CPU lane takes instead round their sums another way.
Exit 0: every session printed session 1's tokens; 1: one did not; 2: the engines did not start.  The summary is also
written to OUT/sessions.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slot_keep_test as SK  # noqa: E402
from remote_slots_test import Worker  # noqa: E402

INFO_KEYS = ("expert_slots", "cpu_experts", "ram_experts")


def gpu_memory(gpu: str) -> dict:
    """The GPU's memory in use (MiB) and each process's on it, from nvidia-smi ({} when it cannot be asked)."""
    out: dict = {}
    try:
        used = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
        out["used_mib"] = int(used.splitlines()[0]) if used else None
        apps = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        out["apps"] = {int(p): int(m) for p, m in (l.split(",") for l in apps.splitlines() if l.count(",") == 1)}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return out


class Main(SK.Engine):
    """SK.Engine that keeps the lines before READY (the INFO line with the placement)."""

    def start(self, timeout: float) -> None:
        self.info: list[str] = []
        _line = self._line

        def keep(end: float) -> str:
            line = _line(end)
            if line.startswith("INFO "):
                self.info.append(line)
            return line

        self._line = keep
        try:
            super().start(timeout)
        finally:
            self._line = _line

    def placement(self) -> dict:
        got: dict = {}
        for line in self.info:
            for k in INFO_KEYS:
                m = re.search(rf"\b{k}=(-?\d+)", line)
                if m:
                    got[k] = int(m.group(1))
        return got


def env_pairs(items: list[str]) -> dict:
    out = {}
    for kv in items:
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="one remote stage worker, main processes one after another")
    ap.add_argument("--exe", default=str(SK.ROOT / "build" / "strata-dsv41"))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids-dir", default="")
    ap.add_argument("--k", type=int, default=20, help="the worker's first layer")
    ap.add_argument("--port", type=int, default=27941)
    ap.add_argument("--ctx", type=int, default=20000)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--len-sys", type=int, default=1200)
    ap.add_argument("--len-task", type=int, default=150)
    ap.add_argument("--engine-args", default="--vram-experts 0.3 --mmap-experts on --ssd-cache 0 --no-expert-cache",
                    help="flags for both processes")
    ap.add_argument("--main-args", default="", help="more flags for the main processes only")
    ap.add_argument("--worker-args", default="", help="more flags for the worker only")
    ap.add_argument("--main-env", action="append", default=[], help="K=V for the main processes (repeatable; "
                    "%%s in V becomes the session's number)")
    ap.add_argument("--worker-env", action="append", default=[], help="K=V for the worker (repeatable)")
    ap.add_argument("--fresh-worker", action="store_true", help="a new worker before each session")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(a.exe):
        print(f"{a.exe}: no engine - build it first", flush=True)
        return 2
    if not os.environ.get("STRATA_STAGE_TLS"):
        print("the link is TLS: strata --stage-keygen DIR on this PC, then STRATA_STAGE_TLS=DIR", flush=True)
        return 2
    ids_dir = Path(a.ids_dir or tempfile.mkdtemp(prefix="sessions-ids-"))
    try:
        P = SK.prompts(ids_dir, a.pack, a.len_sys, a.len_task, 80, 2)
    except (OSError, ImportError, ValueError) as e:
        print(f"the prompts cannot be made: {e}", flush=True)
        return 2
    PIN = len(P["sys"])
    prompt = P["sys"] + P["taskA"]
    thr = str(max(1, (os.cpu_count() or 2) // 2))
    base_env = {"CUDA_VISIBLE_DEVICES": a.gpu, "DS4_ENGRAM_CACHE_MB": "64"}
    common = ["--cpu-threads", thr, *shlex.split(a.engine_args)]
    main_args = ["--pack", a.pack, "--dspark", "off", "--threads", "on", "--vision", "off", "--expert-profile", "none",
                 *common, "--max-context", str(a.ctx), "--remote-stage", f"127.0.0.1:{a.port}", "--stage-end", str(a.k),
                 *shlex.split(a.main_args), "--serve"]
    wenv = {**base_env, **env_pairs(a.worker_env)}

    def worker(name: str) -> Worker:
        return Worker(a.exe, ["--pack", a.pack, *common, "--vision", "off", "--max-context", str(a.ctx), "--stage-worker",
                              str(a.port), "--stage-bind", "127.0.0.1", "--stage-begin", str(a.k),
                              *shlex.split(a.worker_args)], out, name,
                      {**wenv, "STRATA_DS4_SLOT_DIR": str(out / f"{name}-slots")})

    print(f"prompt {len(prompt)} tokens (pin {PIN}), {a.sessions} sessions, max_new {a.max_new}, "
          f"{'a new worker each' if a.fresh_worker else 'one worker'}; main: {' '.join(main_args[1:])}", flush=True)
    rows: list[dict] = []
    wk = None
    try:
        for i in range(1, a.sessions + 1):
            if wk is None or a.fresh_worker:
                if wk is not None:
                    wk.stop()
                wk = worker(f"worker-{i}" if a.fresh_worker else "worker")
                wk.start()
            before = gpu_memory(a.gpu)
            env = {"STRATA_DS4_SLOT_KEEP": "0", "STRATA_DS4_SLOT_DIR": str(out / f"s{i}-slots"), "STRATA_DS4_REUSE": "0",
                   **{k: v.replace("%s", str(i)) for k, v in env_pairs(a.main_env).items()}}
            os.environ.update(base_env)
            eng = Main(a.exe, main_args, env)
            t0 = time.monotonic()
            try:
                eng.start(3600.0)
                load_s = time.monotonic() - t0
                during = gpu_memory(a.gpu)
                r = eng.request(prompt, a.max_new, pin=PIN)
            except RuntimeError as e:
                r = {"error": str(e)}
            finally:
                if eng.proc is not None:
                    try:
                        eng.quit()
                    except RuntimeError as e:
                        print("   ", e, flush=True)
                (out / f"s{i}-main.err").write_text("\n".join(eng.stderr_lines()), encoding="utf-8")
            if "error" in r:
                print(f"s{i}: {r['error']}", flush=True)
                return 2
            wpid = wk.proc.pid if wk.proc is not None else -1
            row = {"session": i, "load_s": round(load_s, 1), "tokens": r["tokens"], "reused": r["reused"],
                   "placement": eng.placement(), "gpu_used_mib_before": before.get("used_mib"),
                   "worker_mib_before": before.get("apps", {}).get(wpid),
                   "main_mib": during.get("apps", {}).get(eng.proc.pid) if eng.proc is not None else None}
            ref = rows[0]["tokens"] if rows else r["tokens"]
            d = next((j for j, (x, y) in enumerate(zip(r["tokens"], ref)) if x != y), None)
            row["first_diff"] = d
            rows.append(row)
            print(f"s{i}: load {load_s:.0f} s, GPU {row['gpu_used_mib_before']} MiB in use before (worker "
                  f"{row['worker_mib_before']}), main {row['main_mib']} MiB, placement {row['placement']}: "
                  + ("= s1" if r["tokens"] == ref else f"differs from s1 at token {d}") + f" {r['tokens'][:10]}",
                  flush=True)
    finally:
        if wk is not None:
            wk.stop()
        (out / "sessions.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    same = all(r["tokens"] == rows[0]["tokens"] for r in rows)
    moved = len({json.dumps(r["placement"], sort_keys=True) for r in rows}) > 1
    print(("every session printed s1's tokens" if same else "a session did not print s1's tokens")
          + ("; the main's placement moved between sessions" if moved else "; the main's placement the same each time"),
          flush=True)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
