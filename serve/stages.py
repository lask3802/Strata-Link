"""The Monitor tab's Workers card: a remote stage's workers (docs/REMOTE_STAGE.md), one entry per worker of the chain.

The main process's own log says what it knows of its link: the worker's address and layers once the hello is through
("strata serve: remote stage HOST:PORT: layers 26-47 there (...)"), the round trips with STRATA_REMOTE_TIMING=1, and
a lost link.  What only the worker PC knows - its GPU and VRAM, its engine's version, its expert cache, its own time
per window, a relay's next worker - comes from its node agent (tools/stage_node.py: GET /info and GET /log over
HTTPS, with this PC's stage key - the engine's STRATA_STAGE_TLS folder - and the agent's certificate pinned to that
folder's peers/, as the engine's link pins its worker's).  The agents are asked only when the run
config says so ("stage_agents": true, or a map of worker address to agent URL: those only), on a thread of their
own, at most every AGENT_EVERY_S and only while a Monitor tab asks for them (GET /metrics?workers=1), so neither a
slow agent nor the agent's nvidia-smi is ever on a request's path.

The lines read are the engine's (a reconnect within a run: "waiting for the worker", "connected again", with
INFO remote_stage_wait= on stdout while a request waits; the hand-off rows' format, ", rows bf16"), and the older
remote stage's without them.  The run config is read when the server starts.

DeepSeek-V4.1's expert server (strata-dsv41 --remote-experts HOST:PORT; docs/DEEPSEEK41.md) is the other kind of link:
this PC runs every layer, and the expert server (another PC's GPU) computes some of the CPU lane's experts.  Its card
("kind": "experts") has one row: the server's address, GPU and the experts it holds as the engine's "ds4: remote
experts: ..." lines say (waiting, the server taken, the upload, the warm-up's round trip, the fills, a link given up),
the requests served while it was up, and the server PC's VRAM from its node agent (GET /info only: the agent does not
run the expert server, so it has no log of it).
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

AGENT_PORT = 7840          # tools/stage_node.py's default "agent_port"
AGENT_EVERY_S = 15.0       # an agent's /info runs nvidia-smi on the worker PC: not more often than this
AGENT_TIMEOUT_S = 4.0
LOG_TAIL = 300             # the worker log's lines read each time; the whole log once per worker start
LOG_OVERLAP = 5            # the last lines of a read that the next tail must hold to be read on from them
CHAIN_MAX = 8              # relays followed at most (a loop in a broken config ends here)
READ_MAX = 1 << 20         # at most the last MiB of the main log's new lines per read ...
HEAD_MAX = 256 << 10       # ... and the first 256 KiB of the engine's current run (its link's start-up lines)
SCAN_MAX = 256 << 20       # the lines before those: at most this much scanned per look (the rest at the next)
WAIT_STALE_S = 25.0        # a request's wait for its worker says so every 10 s (INFO remote_stage_wait=)


def remote_args(args) -> tuple[str | None, int | None]:
    """The engine's --remote-stage HOST:PORT and the worker's first layer, or None: its --layer-split K, or DeepSeek-V4.1's
    own --stage-end K (this PC runs layers 0 to K - 1; it wins when both are there, as the engine reads only that one -
    the server may have added a --layer-split auto)."""
    args = list(args or [])
    addr = k = end = None
    for i, a in enumerate(args[:-1]):
        if a == "--remote-stage":
            addr = str(args[i + 1])
        elif a in ("--layer-split", "--stage-end"):
            try:
                v = int(str(args[i + 1]).split(",")[0])
            except ValueError:
                continue
            if a == "--stage-end":
                end = v
            else:
                k = v
    return addr, end if end is not None and addr is not None else k


def remote_experts_arg(args) -> str | None:
    """DeepSeek-V4.1's --remote-experts HOST:PORT (where this PC listens for its expert server), or None.  A
    --remote-stage beside it wins (remote_args): the card shows the stage, as before."""
    args = list(args or [])
    for i, a in enumerate(args[:-1]):
        if a == "--remote-experts":
            return str(args[i + 1])
    return None


def host_of(addr: str) -> str:
    """"10.0.0.2:7841" -> "10.0.0.2"; "[fe80::1]:7841" -> "fe80::1"."""
    h = addr.rsplit(":", 1)[0] if ":" in addr else addr
    return h[1:-1] if h.startswith("[") and h.endswith("]") else h


def port_of(addr: str) -> int | None:
    try:
        return int(addr.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


class MainLog:
    """The main process's link as its log tells it, read on from where it was last read (as ConvCacheLog does).
    Unlike the conversation cache, the link's history spans engine starts: each start that connects after the first
    is a reconnect.  The server's own "[strata] ... engine started:" line marks where a start begins.  When more than
    READ_MAX was written since the last read, the lines before the last READ_MAX are scanned, not read one by one:
    the starts, connects and relinks in them counted, and the last line of each kind that changes the state read, in
    the log's order (the state reading every line would leave), SCAN_MAX of them per look; "behind": what is left
    to scan (the counts are provisional until then).  The current run's first lines are read whole (its link), from
    where it began even when the line before never ended (an engine that died mid-line).  A log found shorter than
    what was read, or with other first bytes (rotated in place, then grown past it before a look), was rotated: the
    current run began in the old log, unless its start line is at its place in the new one (started after it)."""
    STARTED = " engine started: "
    # (", rows FMT" since the row formats; "; " and the worker's certificate and TLS version after it since TLS)
    CONNECT = re.compile(r"strata serve: remote stage (\S+): layers (\d+)-(\d+) there \(prompt chunk (\d+), "
                         r"context (\d+)(?:, rows (\w+)(?:; .*)?)?\)")
    WAITING = re.compile(r"strata serve: remote stage: waiting for the worker at (\S+) \((.*?)\)(?:, up to |$)")
    RELINK = re.compile(r"strata serve: remote stage: the link to (\S+) was lost \((.*)\): connected again")
    WINDOWS = re.compile(r"strata remote: (\d+) windows: mean ([\d.]+) ms a round trip = worker ([\d.]+) \+ "
                         r"link ([\d.]+)")
    GAP = re.compile(r"strata remote: main process between windows ([\d.]+) ms each")
    # A failed link as the start prints it: "strata serve: " + the link's error.  Each of src/net/stage_link.cpp's
    # errors by its start, and the worker's own ("remote stage (worker): ..."); every other "strata serve: remote
    # stage: ..." line says how it goes (a checkpoint saved or restored, a session continued or dropped after
    # connecting again, a K/V check an earlier worker cannot take part in) and leaves the link as it is.  And the
    # key folder's errors (src/net/stage_tls.cpp's TlsCredentials::load, or a build without OpenSSL) as the start
    # prints them before the model loads: "strata generate: remote stage: " + the error; that prefix's other lines say
    # how the start goes.
    ERROR = re.compile(r"strata (?:serve: |generate: (?=remote stage: (?:no credentials folder|cannot read |TLS: |"
                       r"this build has no TLS)))"
                       r"(remote stage \(worker\): .*|"
                       r"remote stage: (?:no credentials folder|cannot read |TLS: |this build has no TLS|"
                       r"cannot connect to |cannot resolve |expected HOST:PORT|"
                       r"the sockets did not start|not connected|receive failed|send failed|the link failed earlier|"
                       r"the peer closed the connection|the two sides differ|"
                       r"the wait was cancelled|the worker refused|the worker was started without --vision|"
                       r"refused the worker at|no TLS credentials|"
                       r"unexpected reply|bad message magic|a window's rows|an empty position table).*)")
    # a scanned part of the log: the lines counted, and the lines looked for (the last of each kind; a start: a line
    # that begins "[strata]", as event() takes it; an error: the last line ERROR takes)
    COUNTED = {"connects": b" there (prompt chunk ", "relinks": b"): connected again"}
    MARKS = {"connect": b" there (prompt chunk ", "relink": b"): connected again",
             "wait": b"strata serve: remote stage: waiting for the worker at ", "windows": b" windows: mean ",
             "gap": b"strata remote: main process between windows "}
    STARTS = re.compile(rb"\n\[strata\][^\n]*" + re.escape(STARTED.encode()))
    ERRORS = re.compile(ERROR.pattern.encode())
    # DeepSeek-V4.1's expert server (--remote-experts): the engine's own lines (src/dsv41/ds4_model.cu, rx_setup /
    # rx_lost), each after "ds4: remote experts: "
    RX = "ds4: remote experts: "
    RX_WAIT = re.compile(r"waiting up to (\d+) s for an expert server on (\S+)")
    RX_SERVER = re.compile(r"server (\S+) \((.*), ([\d.]+) GB of VRAM for experts\): sending (\d+) of the CPU lane's "
                           r"(\d+) experts \(([\d.]+) GB, ([\d.]+)% of their routings")
    RX_UP = re.compile(r"(\d+) experts \(([\d.]+) GB\) in the server's VRAM, uploaded in ([\d.]+) s")
    RX_WARM = re.compile(r"(\d+) warm-up requests of 1 row: round trip median ([\d.]+) ms")
    RX_FILLS = re.compile(r"(fills on|fills off|no fills) \((.*?)[;)]")
    RX_LOST = re.compile(r"(.*) - the CPU lane computes all its experts( from now on)?\s*$")
    RX_NONE = re.compile(r"none held")
    RX_LATE = re.compile(r"(a reply past its deadline|8 of the last 64 replies past the link's bound)")
    PROMPT = "maya ds4 serve: prompt "     # one request served (the engine's line after each)
    COUNTED["rx_connects"] = b"GB of VRAM for experts): sending "
    MARKS.update({"rx_wait": b"ds4: remote experts: waiting up to ", "rx_server": b"GB of VRAM for experts): sending ",
                  "rx_up": b" in the server's VRAM, uploaded in ", "rx_warm": b" warm-up requests of 1 row: round trip ",
                  "rx_fon": b"ds4: remote experts: fills on (", "rx_foff": b"ds4: remote experts: fills off (",
                  "rx_nofill": b"ds4: remote experts: no fills (", "rx_lost": b" - the CPU lane computes all its experts",
                  "rx_none": b"ds4: remote experts: none held"})
    FINGERPRINT = 64       # the log's first bytes: other ones at a later look, a log rotated (though not shorter)

    def __init__(self):
        self.key, self.pos = None, 0
        self.read_at = 0.0             # when the log was last read
        self.lock = threading.Lock()   # two pages (or a scraper) reading at once must not count a line twice
        self.head = b""                # the log's first FINGERPRINT bytes at the last read
        self.old_start = None          # a run's start in a log rotated since (no place in the new one)
        self.err_seen = None           # the engine's ERR as last seen (a new one: placed in this log)
        # Each event carries where its line is in the log ("pos"), which orders it against the engine's stdout ERR
        # (the server notes the log's size when it reads one), and "at", when it was read (a lazy read sees it late:
        # shown, never compared).  "link": the current run's last connect or relink; "run_pos": where that run began;
        # "stage_err": the engine's ERR with its place; "size": the log's size at this read; "rotated": the last
        # rotation seen (the log emptied, cut short or begun again) - "by": the places read moved that much, before
        # the new log's start, "at": the read before it
        self.state = {"connect": None, "connects": 0, "relinks": 0, "waiting": None, "error": None, "link": None,
                      "windows": None, "gap_ms": None, "runs": 0, "run_pos": None, "run_connected": False,
                      "stage_err": None, "behind": 0, "size": 0, "rotated": None,
                      # the expert server's link: "rx" this run's facts (rx_new), "rx_connects" the servers taken so far
                      "rx": None, "rx_connects": 0}

    def poll(self, path, first, start, err=None) -> dict:
        """The state after the log's new lines.  `first`: where the engine's first run in this server began (read
        from there once), `start`: where its current run began, `err`: the engine's last ERR (its text, when, the
        log's size then)."""
        if not path or start is None:
            return dict(self.state)
        with self.lock:
            st = self.state
            if self.key != path:
                self.key, self.pos, self.head, self.old_start = path, start if first is None else first, b"", None
            run_start = start
            if start == self.old_start:                 # this run began in a log rotated since: no place in this one
                start = -1
            else:                                       # (a later run: the rotated one's place is of no use now)
                self.old_start = None
            rot, size = None, None
            try:
                size = os.path.getsize(path)
                with open(path, "rb") as f:
                    head = f.read(min(self.FINGERPRINT, size))
                    n = min(len(head), len(self.head))
                    if size < self.pos or head[:n] != self.head[:n]:   # emptied, cut short or begun again: rotated
                        rot = self.rotated()
                        self.old_start, start = run_start, -1
                    if start == -1 and self.starts_at(f, run_start, size):   # unless the run's start line is here:
                        self.old_start, start = None, run_start              # it began after the rotation
                    self.head = head
                    budget = SCAN_MAX
                    while size - self.pos > READ_MAX and budget > 0:
                        if self.pos == start:           # the current run's first lines: read
                            f.seek(start)
                            head = f.read(min(HEAD_MAX, size - READ_MAX - start))
                            n = head.rfind(b"\n") + 1
                            if n:
                                self.feed(head[:n], start)
                                self.pos, budget = self.pos + n, budget - n
                                continue
                        stop = min(size - READ_MAX, self.pos + budget)
                        if self.pos < start < stop:
                            stop = start
                        n = self.scan(f, self.pos, stop)
                        if not n:
                            if stop == start:           # the run's first line, after a line that never ended
                                self.pos = start
                                continue
                            # no whole line in it: the budget is spent, or a line runs into the end (read below)
                            budget = 0 if stop < size - READ_MAX else budget
                            break
                        self.pos, budget = self.pos + n, budget - n
                    behind = size - self.pos > READ_MAX and budget <= 0
                    st["behind"] = size - READ_MAX - self.pos if behind else 0
                    f.seek(self.pos)
                    data = b"" if behind else f.read(size - self.pos)
            except OSError:
                data = b""
            if self.pos < start <= self.pos + len(data):   # the same, read line by line: the lines before it
                n = start - self.pos
                self.feed(data[:n], self.pos)
                data, self.pos = data[n:], start
            end = data.rfind(b"\n") + 1                 # whole lines only: the rest is read next time
            self.feed(data[:end], self.pos)
            self.pos += end
            if err != self.err_seen:                    # a new ERR (an older one moved at each rotation): of the old
                self.err_seen = err                     # log, if one was rotated since it came, past the new end
                pos = err[2] if err and len(err) > 2 and err[2] is not None else float("-inf")   # (unknown: first)
                if err and rot and size is not None and pos > size:
                    pos = min(pos - rot["by"], -1)
                st["stage_err"] = {"text": err[0], "at": err[1], "pos": pos} if err else None
            if size is not None:
                self.read_at, st["size"] = time.time(), size
            return dict(st)

    def rotated(self) -> dict:
        """The log was rotated: what was read keeps its order, before the new log's first line, and is read from
        its start.  An ERR of the old log moves with it once, here."""
        st, by = self.state, self.pos
        for k in ("link", "error", "waiting", "stage_err"):
            if st[k]:
                st[k] = {**st[k], "pos": min(st[k]["pos"] - by, -1)}
        if st["run_pos"] is not None:
            st["run_pos"] -= by
        st["rotated"], self.pos = {"by": by, "at": self.read_at}, 0
        return st["rotated"]

    def starts_at(self, f, at: int, size: int) -> bool:
        """The server's start line at byte `at` of the log (as event() takes it)."""
        if not 0 <= at < size:
            return False
        f.seek(at)
        line = f.read(128).split(b"\n")[0]   # ("[strata] YYYY-MM-DD HH:MM:SS engine started: " is 46 bytes)
        return line.startswith(b"[strata]") and self.STARTED.encode() in line

    @staticmethod
    def line_at(buf: bytes, i: int, base: int) -> tuple[int, str]:
        """The line of buf that holds byte i: where it begins in the log (buf begins at `base`), and its text."""
        a = buf.rfind(b"\n", 0, i) + 1
        b = buf.find(b"\n", i)
        return base + a, buf[a:b].decode("utf-8", "replace").rstrip("\r")

    def scan(self, f, a: int, b: int) -> int:
        """The whole lines from byte a to b, scanned (see the class): the bytes scanned, up to the last line end."""
        f.seek(a)
        pos, left, carry, last = a, b - a, b"", {}
        served = 0                                      # the requests after the last start scanned (the expert server's)
        prompt = self.PROMPT.encode()
        while left > 0:
            chunk = f.read(min(left, 8 << 20))
            if not chunk:
                break
            left -= len(chunk)
            buf = b"\n" + carry + chunk                 # (a line end before its first line: STARTS takes it too)
            end = buf.rfind(b"\n") + 1
            carry = buf[end:]
            if end == 1:
                continue
            base = pos - 1
            for k, pat in self.COUNTED.items():
                self.state[k] += buf.count(pat, 0, end)
            for kind, mark in self.MARKS.items():
                i = buf.rfind(mark, 0, end)
                if i >= 0:
                    last[kind] = self.line_at(buf, i, base)
            m = None
            for m in self.STARTS.finditer(buf, 0, end):
                self.state["runs"] += 1
            if m:
                last["start"] = self.line_at(buf, m.start() + 1, base)
                served = buf.count(prompt, m.end(), end)
            else:
                served += buf.count(prompt, 0, end)
            m = None
            for m in self.ERRORS.finditer(buf, 0, end):
                pass
            if m:
                last["error"] = self.line_at(buf, m.start(), base)
            pos += end - 1
        now = time.time()
        for at, line in sorted(last.values()):         # in the log's order
            self.event(line, at, now, count=False)
        rx = self.state["rx"]
        if rx and rx["server"] and not rx["lost"]:     # (a scanned part: all of its requests, none when the link is lost by its end)
            rx["requests"] += served
        return pos - a

    def feed(self, data: bytes, base: int) -> None:
        """Whole lines that begin at byte `base` of the log."""
        now = time.time()
        pos = base
        for raw in data.split(b"\n")[:-1]:
            at, pos = pos, pos + len(raw) + 1
            self.event(raw.decode("utf-8", "replace").rstrip("\r"), at, now)

    def event(self, line: str, at: int, now: float, count: bool = True) -> None:
        """One line at byte `at` (`count`: a start, connect or relink is counted here, not by a scan)."""
        st = self.state
        if line.startswith("[strata]") and self.STARTED in line:
            st["runs"] += count
            st.update(run_pos=at, run_connected=False, waiting=None, link=None, rx=None)
            st["windows"] = st["gap_ms"] = None             # the timing of this run's link only
            return
        if line.startswith(self.PROMPT):
            rx = st["rx"]
            if count and rx and rx["server"] and not rx["lost"]:
                rx["requests"] += 1
            return
        if "remote" not in line:
            return
        i = line.find(self.RX)
        if i >= 0:
            self.rx_event(line[i + len(self.RX):], at, now, count)
            return
        m = self.CONNECT.search(line)
        if m:
            st["connects"] += count
            st["run_connected"], st["waiting"] = True, None
            st["link"] = {"pos": at, "at": now}
            st["connect"] = {"address": m[1], "layers": [int(m[2]), int(m[3])], "chunk": int(m[4]),
                             "context": int(m[5]), "rows": m[6], "at": now, "relinked": False}
            return
        m = self.RELINK.search(line)
        if m:
            st["relinks"] += count
            st["run_connected"], st["waiting"] = True, None
            st["link"] = {"pos": at, "at": now}
            st["error"] = {"text": m[2], "at": now, "pos": at}
            if st["connect"]:
                st["connect"] = {**st["connect"], "at": now, "relinked": True}
            return
        m = self.WAITING.search(line)
        if m:
            st["waiting"] = {"address": m[1], "why": m[2], "at": now, "pos": at}
            return
        m = self.WINDOWS.search(line)
        if m:
            st["windows"] = {"n": int(m[1]), "round_trip_ms": float(m[2]), "worker_ms": float(m[3]),
                             "link_ms": float(m[4]), "at": now}
            return
        m = self.GAP.search(line)
        if m:
            st["gap_ms"] = float(m[1])
            return
        m = self.ERROR.search(line)
        if m:
            st["error"] = {"text": m[1], "at": now, "pos": at}
            st["run_connected"], st["waiting"] = False, None   # (a start that waited and gave up ends)

    @staticmethod
    def rx_new() -> dict:
        """An engine run's expert-server facts: "wait" (listening for one), "server" (the one taken: its address, GPU,
        the experts sent), "held" (the upload done), "warm_ms" (the warm-up's median round trip), "fills" (True / False,
        and why not), "lost" (the link given up, or never made: its reason), "none" (it holds none of them), "late"
        (the late-reply lines) and "requests" (served while the link was up)."""
        return {"wait": None, "server": None, "held": None, "warm_ms": None, "fills": None, "fills_note": None,
                "lost": None, "none": False, "late": 0, "late_note": None, "requests": 0}

    def rx_event(self, text: str, at: int, now: float, count: bool) -> None:
        """One "ds4: remote experts: " line (`text`: what follows it)."""
        st = self.state
        rx = st["rx"]
        if rx is None:
            rx = st["rx"] = self.rx_new()
        if m := self.RX_SERVER.match(text):
            st["rx_connects"] += count
            rx.update(server={"address": m[1], "gpu": m[2], "vram_gb": float(m[3]), "experts": int(m[4]),
                              "lane": int(m[5]), "gb": float(m[6]), "routing_pct": float(m[7]), "at": now, "pos": at},
                      wait=None, lost=None, none=False)
        elif m := self.RX_WAIT.match(text):
            rx.update(wait={"seconds": int(m[1]), "listen": m[2], "at": now, "pos": at}, lost=None)
        elif m := self.RX_UP.match(text):
            rx["held"] = {"experts": int(m[1]), "gb": float(m[2]), "upload_s": float(m[3]), "at": now}
        elif m := self.RX_WARM.match(text):
            rx["warm_ms"] = float(m[2])
        elif m := self.RX_FILLS.match(text):
            rx["fills"], rx["fills_note"] = m[1] == "fills on", None if m[1] == "fills on" else m[2]
        elif m := self.RX_LOST.match(text):
            rx.update(lost={"text": m[1], "at": now, "pos": at, "after_link": bool(m[2])}, wait=None)
        elif self.RX_NONE.match(text):
            rx["none"] = True
        elif m := self.RX_LATE.match(text):
            rx["late"] += count
            rx["late_note"] = {"text": text.strip(), "at": now}


class WorkerLog:
    """A worker's facts from its log (the node agent's GET /log): its engine, expert cache, layers, the next worker
    of a relay, its own time per window, and the main process's (or the previous relay's) connection to it."""
    VERSION = re.compile(r"session is up \(engine ([^)\s]+)\)")
    CACHE = re.compile(r"strata generate: expert cache (\d+) slots, ([\d.]+) GiB of VRAM")
    LAYERS = re.compile(r"strata serve: stage worker: layers (\d+)-(\d+), (?:relaying to ([^,\s]+)|no head)")
    LISTEN = re.compile(r"strata stage worker: listening on ")
    OWN = re.compile(r"strata stage worker: windows: own layers ([\d.]+) ms(?:, next worker ([\d.]+) ms)? each")
    MAIN = re.compile(r"strata stage worker: main process (\S+)(?: \(.*\))?: layers")   # (its certificate, since TLS)
    GONE = re.compile(r"strata stage worker: (\S+) disconnected(?:: (.*))?")
    RELAY = re.compile(r"strata serve: relay: layers (\d+)-(\d+) on (\S+)")
    RELAY_ERR = re.compile(r"strata serve: relay: (?!layers \d)(.*)")

    @classmethod
    def parse(cls, lines: list[str], facts: dict | None = None) -> dict:
        """`facts` updated by `lines` (a log's whole text, or its tail after it was read whole once)."""
        f = dict(facts or {})
        for line in lines:
            if cls.LISTEN.search(line):
                f["listening"] = True
            elif m := cls.VERSION.search(line):
                f["version"] = m[1]
            elif m := cls.CACHE.search(line):
                f["cache_slots"], f["cache_gib"] = int(m[1]), float(m[2])
            elif m := cls.LAYERS.search(line):
                f["layers"], f["next"] = [int(m[1]), int(m[2])], m[3]
            elif m := cls.OWN.search(line):
                f["own_ms"] = float(m[1])
                f["next_ms"] = float(m[2]) if m[2] else None
            elif m := cls.MAIN.search(line):
                f["peer"], f["peer_error"] = m[1], None
            elif m := cls.GONE.search(line):
                f["peer"], f["peer_error"] = None, m[2]
                # a relay whose next worker failed in a window: the error names it when it could not reach it
                if m[2] and f.get("next") and f["next"] in m[2]:
                    f["next_ok"], f["next_error"] = False, m[2]
            elif m := cls.RELAY.search(line):
                f["next_layers"], f["next_ok"], f["next_error"] = [int(m[1]), int(m[2])], True, None
            elif m := cls.RELAY_ERR.search(line):
                f["next_ok"], f["next_error"] = False, m[1]
        return f


class AgentSaid(Exception):
    """An agent's own error answer ({"ok": false, "error": ...})."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):   # the agent named, never on to another address
        return None


_stage_node = None
_contexts: dict[str, object] = {}
_contexts_lock = threading.Lock()


def stage_node():
    """tools/stage_node.py, loaded from this tree once: the agent's own TLS (the same pinned certificates)."""
    global _stage_node
    if _stage_node is None:
        p = Path(__file__).resolve().parent.parent / "tools" / "stage_node.py"
        spec = importlib.util.spec_from_file_location("strata_stage_node", p)
        if spec is None or spec.loader is None:
            raise AgentSaid(f"no node agent's code at {p}")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _stage_node = m
    return _stage_node


def _opener(keys: str):
    """Straight to the agent over HTTPS with this PC's stage key (`keys`: the engine's STRATA_STAGE_TLS): its
    certificate one this PC accepts, no plain http, no redirect, and no proxy (HTTP_PROXY, the system's).  The folder
    is read once per server start, as the engine reads it once per start."""
    if not keys:
        raise AgentSaid("no key folder: the agents speak HTTPS with this PC's stage key (STRATA_STAGE_TLS in the "
                        "engine's environment)")
    with _contexts_lock:
        ctx = _contexts.get(keys)
        if ctx is None:
            sn = stage_node()
            try:
                ctx = _contexts[keys] = sn.tls_context(Path(keys).resolve(), server=False)
            except (OSError, ssl.SSLError, ValueError) as e:
                raise AgentSaid(f"the key folder (STRATA_STAGE_TLS): {e}") from None
    o = urllib.request.OpenerDirector()
    for h in (urllib.request.ProxyHandler({}), urllib.request.UnknownHandler(), urllib.request.HTTPSHandler(context=ctx),
              urllib.request.HTTPDefaultErrorHandler(), _NoRedirect(), urllib.request.HTTPErrorProcessor()):
        o.add_handler(h)
    return o


def agent_url(url: str) -> str:
    """An agent's address as https:// (an http:// one from an earlier run config is the same agent)."""
    url = url.rstrip("/")
    scheme, sep, rest = url.partition("://")
    return "https://" + rest if sep and scheme.lower() in ("http", "https") else url


class Agent:
    """One node agent: its last /info and the facts of one worker's log, refreshed on a thread."""

    def __init__(self, url: str, port: int | None):
        self.url, self.port = agent_url(url), port
        self.info = None          # the agent's /info, trimmed to what the card shows
        self.facts: dict = {}
        self.read_whole = False   # the worker's log was read since its start (whole, or its tail when that failed)
        self.last_lines: list[str] = []   # the end of the last read: the next tail is read on from them
        self.seen_running = False  # the agent ran this worker at some look (else its port's log may be an old one)
        self.ok = None            # None: not asked yet
        self.error = None
        self.at = None            # when it last answered (or failed)
        self.log_lines = 0

    def get(self, path: str, keys: str):
        with _opener(keys).open(self.url + path, timeout=AGENT_TIMEOUT_S) as r:
            out = json.loads(r.read().decode("utf-8", "replace"))
        if not isinstance(out, dict):
            raise AgentSaid("not an agent's answer")
        return out

    def log(self, keys: str, whole: bool) -> list[str]:
        """The worker's log: its last LOG_TAIL lines, or all of them (once per worker start: its start-up lines)."""
        log = self.get(f"/log?port={self.port}&lines={10 ** 7 if whole else LOG_TAIL}", keys)
        return [str(x) for x in log.get("log") or []]

    def whole_log(self, keys: str, tail: list[str] | None) -> tuple[list[str], bool]:
        """All of the worker's log, and True; when that read fails (a log too long to send in time), its tail and
        False - read on from there next time, with the facts known so far, instead of failing at every look."""
        try:
            return self.log(keys, whole=True), True
        except (AgentSaid, ValueError, OSError, http.client.HTTPException):
            return (tail if tail is not None else self.log(keys, whole=False)), False

    def new_lines(self, lines: list[str]) -> list[str] | None:
        """The lines of a tail after the last read's end, or None when it does not hold that end (a new worker's
        log, or more than a tail written since)."""
        k = len(self.last_lines)
        if not k:
            return None
        for i in range(len(lines) - k, -1, -1):
            if lines[i:i + k] == self.last_lines:
                return lines[i + k:]
        return None

    def scrub(self, text: str) -> str:
        """`text` without the URL's user:password - an error goes to the browser."""
        user = urlsplit(self.url).netloc.rpartition("@")[0]
        return text.replace(user, "...") if user else text

    def refresh(self, keys: str) -> None:
        try:
            info = self.get("/info", keys)
            if not info.get("ok", True):
                raise AgentSaid(str(info.get("error") or "the agent said no"))
            gpus = info.get("gpus") if isinstance(info.get("gpus"), list) else []
            running = [int(p) for p in info.get("workers") or [] if str(p).lstrip("-").isdigit()]
            run = self.port in running if self.port is not None else None
            self.seen_running = self.seen_running or bool(run)
            self.info = {"host": info.get("host"), "gpus": [{k: g.get(k) for k in ("name", "vram_mib", "used_mib")}
                                                             for g in gpus if isinstance(g, dict)], "running": run}
            if self.port is not None:
                # the agent starts each worker's log anew (tools/stage_node.py): a tail that holds the last read's
                # end goes on from it, any other is a new worker's log (or more than a tail was written since)
                lines = self.log(keys, whole=False) if self.read_whole else None
                new = self.new_lines(lines) if lines is not None and len(lines) >= LOG_TAIL else None
                if new is not None:
                    self.facts = WorkerLog.parse(new, self.facts)
                elif lines is not None and len(lines) < LOG_TAIL:
                    self.facts = WorkerLog.parse(lines)             # a tail shorter than LOG_TAIL is all of it
                else:
                    lines, whole = self.whole_log(keys, lines)
                    self.facts = WorkerLog.parse(lines, None if whole else self.facts)
                self.read_whole = bool(lines)
                self.last_lines = lines[-LOG_OVERLAP:]
                self.log_lines = len(lines)
            self.ok, self.error = True, None
        except urllib.error.HTTPError as e:
            self.ok = False
            self.error = f"HTTP {e.code}"
            e.close()
        except (AgentSaid, urllib.error.URLError, OSError) as e:   # the network's, TLS's or the agent's own words
            self.ok = False
            reason = getattr(e, "reason", None) or e
            self.error = self.scrub(str(reason)) or type(e).__name__
        except Exception as e:   # noqa: BLE001 - any failure is this agent's state, never the thread's end
            # its type only: a message may quote the request
            self.ok = False
            self.error = f"the agent's request failed ({type(e).__name__})"
        self.at = time.time()

    def view(self) -> dict:
        u = urlsplit(self.url)
        shown = u._replace(netloc=u.netloc.rsplit("@", 1)[-1]).geturl()   # no user:password of the URL
        return {"url": shown, "ok": self.ok, "error": self.error, "at": self.at}


class StageMonitor:
    """The Workers card's data (GET /metrics "stages"): None when the engine runs no remote stage."""

    def __init__(self):
        self.main_log = MainLog()
        # the run config's "stage_agents" (opt-in): true asks each worker's agent at HOST:7840, a map
        # {"HOST:PORT": "https://agent:7840", ...} asks the agents it names and no other (http:// is taken as https://)
        self.agents_cfg = None
        self.agents: dict[str, Agent] = {}
        self.lock = threading.Lock()
        self.busy = False
        self.asked_at = 0.0

    def agents_on(self) -> bool:
        return self.agents_cfg is True or isinstance(self.agents_cfg, dict)

    def agent_for(self, addr: str, expert_server: bool = False) -> Agent | None:
        cfg = self.agents_cfg
        if not self.agents_on():
            return None
        # a map names every agent that is asked; true: each worker's own host (its agent's certificate must be one
        # this PC accepts, as the worker's is).  `expert_server`: DeepSeek-V4.1's expert server at `addr` (its address and
        # this PC's listening port): a map's entry for another port of that host names the same PC's agent, and only its
        # /info is asked (the agent does not run the expert server: no log of it)
        url = cfg.get(addr) if isinstance(cfg, dict) else None
        if isinstance(cfg, dict) and not url and expert_server:
            url = next((u for k, u in cfg.items() if isinstance(k, str) and host_of(k) == host_of(addr)), None)
        if isinstance(cfg, dict) and not (isinstance(url, str) and url):
            return None
        if not url:
            h = host_of(addr)
            url = f"https://{'[' + h + ']' if ':' in h else h}:{AGENT_PORT}"
        if urlsplit(url).scheme not in ("http", "https"):
            return None
        with self.lock:
            a = self.agents.get(addr)
            if a is None or a.url != agent_url(url):
                a = self.agents[addr] = Agent(url, None if expert_server else port_of(addr))
            return a

    def ask_agents(self, chain: list[str], keys: str, expert_server: bool = False) -> None:
        """Every agent of the chain, on a thread, at most every AGENT_EVERY_S; a call while one runs does nothing."""
        now = time.time()
        with self.lock:
            if self.busy or now - self.asked_at < AGENT_EVERY_S:
                return
            self.busy, self.asked_at = True, now

        def run():
            try:
                for a in [a for a in (self.agent_for(x, expert_server) for x in chain) if a is not None]:
                    a.refresh(keys)
            finally:
                with self.lock:
                    self.busy = False
        threading.Thread(target=run, daemon=True, name="stage-agents").start()

    def view(self, engine, ask: bool = False) -> dict | None:
        spawn = getattr(engine, "spawn", None)
        args = spawn[1] if isinstance(spawn, tuple) and len(spawn) > 1 else []
        first, k = remote_args(args)
        listen = None if first else remote_experts_arg(args)
        if not first and not listen:
            return None
        env = spawn[4] if len(spawn) > 4 and isinstance(spawn[4], dict) else None
        keys = (env or os.environ).get("STRATA_STAGE_TLS", "")   # the engine's key folder: this PC's
        log = self.main_log.poll(getattr(engine, "log_path", None), getattr(engine, "first_log_start", None),
                                 getattr(engine, "log_start", None), getattr(engine, "stage_err", None))
        if listen:
            return self.experts_view(engine, listen, log, keys, ask)
        # the chain: the main process's worker, then each relay's next worker as its log names it
        chain = [first]
        while len(chain) < CHAIN_MAX:
            a = self.agents.get(chain[-1])
            nxt = a.facts.get("next") if a is not None else None
            if not nxt or nxt in chain:
                break
            chain.append(nxt)
        on = self.agents_on()
        if ask and on:
            self.ask_agents(chain, keys)
        workers = [self.worker_view(i, addr, chain, engine, log, on) for i, addr in enumerate(chain)]
        c = log["connect"]
        rows = c.get("rows") if c else None
        # the round trips are in the log only with STRATA_REMOTE_TIMING=1 in the engine's environment (as it reads it)
        timing = str((env or os.environ).get("STRATA_REMOTE_TIMING", "")).startswith("1")
        return {"local_layers": [0, k - 1] if k else None, "rows": rows, "chunk": c["chunk"] if c else None,
                "context": c["context"] if c else None, "gap_ms": log["gap_ms"], "agents": on, "timing": timing,
                "provisional": log["behind"] > 0, "workers": workers}

    def experts_view(self, engine, listen: str, log: dict, keys: str, ask: bool) -> dict:
        """The card for DeepSeek-V4.1's expert server (--remote-experts `listen`): one row, "kind": "experts"."""
        rx = dict(log.get("rx") or MainLog.rx_new())
        srv = rx["server"]
        on = self.agents_on()
        a = None
        if srv:
            # the server's address as the engine names it (no port: the server connected to this PC's `listen`)
            peer = srv["address"]
            key = f"{'[' + peer + ']' if ':' in peer else peer}:{port_of(listen) or 0}"
            if on:
                if ask:
                    self.ask_agents([key], keys, expert_server=True)
                a = self.agent_for(key, expert_server=True)
        info = a.info if a is not None and a.ok else None
        gpu = (info.get("gpus") or [None])[0] if info else None
        if gpu is None and srv:
            gpu = {"name": srv["gpu"], "vram_mib": None, "used_mib": None}   # (the engine's word for it, no VRAM)
        state, since, err = self.experts_state(engine, rx)
        held = rx["held"]
        w = {"address": srv["address"] if srv else None, "listen": listen, "host": info.get("host") if info else None,
             "agent": a.view() if a else None, "gpu": gpu, "version": None, "cache_slots": None, "cache_gib": None,
             "own_ms": None, "relay_to": None, "layers": None, "round_trip_ms": rx["warm_ms"], "worker_ms": None,
             "link_ms": None, "reconnects": max(0, log.get("rx_connects", 0) - 1), "since": since, "error": err,
             "state": state,
             "experts": {"held": held["experts"] if held else (srv["experts"] if srv else None),
                         "gb": held["gb"] if held else (srv["gb"] if srv else None),
                         "lane": srv["lane"] if srv else None, "routing_pct": srv["routing_pct"] if srv else None,
                         "vram_gb": srv["vram_gb"] if srv else None, "upload_s": held["upload_s"] if held else None,
                         "fills": rx["fills"], "fills_note": rx["fills_note"], "late": rx["late"],
                         "late_note": rx["late_note"], "requests": rx["requests"] if srv else None}}
        return {"kind": "experts", "listen": listen, "local_layers": None, "rows": None, "chunk": None, "context": None,
                "gap_ms": None, "agents": on, "timing": True, "provisional": log["behind"] > 0, "workers": [w]}

    def experts_state(self, engine, rx: dict):
        """The expert server's link: (state, since, error), from this run's lines (MainLog.rx_event) and the engine."""
        if getattr(engine, "unloaded", False):
            return "unloaded", None, None
        alive = engine.alive() if hasattr(engine, "alive") else True
        proc = getattr(engine, "proc", None)
        starting = not alive and proc is not None and proc.poll() is None
        if not getattr(engine, "log_path", None):
            return ("nolog" if alive else "connecting" if starting else "down"), None, None
        lost, srv, wait = rx["lost"], rx["server"], rx["wait"]
        if lost:   # given up (or never made): the CPU lane computes every expert until the engine starts again
            return "down", lost["at"], {"text": lost["text"], "at": lost["at"]}
        if not alive and not starting:
            return "down", None, None
        if rx["none"]:
            return "idle", None, {"text": "the expert server holds none of this PC's experts", "at": None}
        if srv:
            return "connected", getattr(engine, "ready_at", None) or srv["at"], None
        if wait:
            return "connecting", wait["at"], {"text": f"waiting up to {wait['seconds']} s for an expert server on "
                                                      f"{wait['listen']}", "at": wait["at"]}
        return ("connecting" if starting else "unknown"), None, None

    def worker_view(self, i: int, addr: str, chain: list[str], engine, log: dict, on: bool) -> dict:
        a = self.agents.get(addr) if on else None
        f = a.facts if a is not None and a.ok else {}
        info = a.info if a is not None and a.ok else None
        gpu = (info.get("gpus") or [None])[0] if info else None
        w = {"address": addr, "host": info.get("host") if info else None, "agent": a.view() if a else None,
             "gpu": gpu, "version": f.get("version"), "cache_slots": f.get("cache_slots"),
             "cache_gib": f.get("cache_gib"), "own_ms": f.get("own_ms"), "relay_to": f.get("next"),
             "layers": f.get("layers"), "round_trip_ms": None, "worker_ms": None, "link_ms": None,
             "reconnects": None, "since": None, "error": None}
        if i == 0:
            state, since, err = self.main_state(engine, log)
            c = log["connect"]
            if c and not w["layers"]:
                w["layers"] = c["layers"]
            win = log["windows"]
            if win:
                w.update(round_trip_ms=win["round_trip_ms"], worker_ms=win["worker_ms"], link_ms=win["link_ms"])
            w["reconnects"] = max(0, log["connects"] - 1) + log["relinks"]
        else:
            prev = self.agents.get(chain[i - 1]) if on else None
            pf = prev.facts if prev is not None and prev.ok else {}
            since, err = None, None
            if not w["layers"] and pf.get("next_layers"):
                w["layers"] = pf["next_layers"]
            w["round_trip_ms"] = pf.get("next_ms")
            if pf.get("next_ok") is True:
                state = "connected"
            elif pf.get("next_ok") is False:
                state, err = "down", {"text": pf.get("next_error"), "at": prev.at}
            else:
                state = "unknown"
        # the worker's agent: a worker process it ran that ended is down, whatever the link last said
        if info is not None and info.get("running") is False and a.seen_running and a.log_lines:
            if state in ("connected", "unknown"):
                state = "down"
                why = f.get("peer_error")
                err = {"text": "the worker process ended" + (f" ({why})" if why else ""), "at": a.at}
        w.update(state=state, since=since, error=err)
        return w

    def main_state(self, engine, log: dict):
        """The main process's link: (state, since, error).  Its events are ordered by where they are in the engine's
        log (an ERR on stdout by the log's size when the server read it), never by when the log was read."""
        if getattr(engine, "unloaded", False):
            return "unloaded", None, None
        alive = engine.alive() if hasattr(engine, "alive") else True
        proc = getattr(engine, "proc", None)
        starting = not alive and proc is not None and proc.poll() is None
        if not getattr(engine, "log_path", None):              # the run config has no "log": nothing says how it goes
            return ("nolog" if alive else "connecting" if starting else "down"), None, None
        # this run's errors only: the log's (a failed start), the ERR of a request (from this run's start on)
        # (an ERR at run_pos came before the start; its place as MainLog.poll keeps it)
        run_pos = log["run_pos"]
        errs = [e for e in (log["error"], log["stage_err"]) if e and (run_pos is None or e["pos"] > run_pos)]
        err = max(errs, key=lambda e: e["pos"]) if errs else None
        link, wait = log["link"], log["waiting"]
        closed = False         # the link was closed after it was made, the engine kept: the next request connects again
        if wait:
            # a request's wait for the worker (main-next) is over at its ERR, or when no heartbeat came for
            # WAIT_STALE_S (an ERR the server did not see).  A start's wait lasts while the process starts.
            beat = getattr(engine, "stage_wait_at", None) or 0
            stale = not starting and time.time() - max(wait["at"], beat) > WAIT_STALE_S
            if (err and err["pos"] > wait["pos"]) or stale:   # (an ERR at wait["pos"] came before it)
                wait, closed = None, True
        again = log["connects"] > 0 or log["relinks"] > 0
        if wait:                                                # why it waits: the last try's error
            return ("reconnecting" if again else "connecting"), wait["at"], {"text": wait["why"], "at": wait["at"]}
        if starting:
            return ("reconnecting" if again else "connecting"), None, err if again else None
        if err and link and err["pos"] <= link["pos"]:          # history: the link was made again after it
            err = None
        if not alive:
            return "down", err["at"] if err else None, err
        if link is None:                                        # up, but its log has not said so (yet)
            return "unknown", None, err
        # the worker's own failure of a step leaves the link as it is; any other error after the link closed it
        if closed or (err and not err["text"].startswith("remote stage (worker)")):
            return "idle", err["at"] if err else None, err
        c = log["connect"]
        since = link["at"] if c and c["relinked"] else getattr(engine, "ready_at", None) or link["at"]
        return "connected", since, err
