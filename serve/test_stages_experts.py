"""The Workers card for DeepSeek-V4.1's expert server (strata-dsv41 --remote-experts; serve/stages.py "kind": "experts"):
the engine log's "ds4: remote experts: ..." lines, the server PC's node agent (GET /info only), and the page's own code
(serve/web/app.js, run in node with stand-ins for its helpers).  No GPU, no engine, no network beyond 127.0.0.1.

    python -m unittest serve.test_stages_experts
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from serve import stages
from serve.server import ByteTokenizer, MockEngine
from serve.test_stages import (MAIN_RUN, STARTED, FakeAgent, LogCase, RemoteEngine, setUpModule,  # noqa: F401
                               tearDownModule, view_now)  # noqa: F401  (the key folders the agents speak with)

DS_ARGS = ["--pack", "/p/deepseek-v4.1-flash-iq1m", "--max-context", "262144", "--vision", "on",
           "--remote-experts", "0.0.0.0:27861", "--verbose"]
DS_STARTED = ("[strata] 2026-11-03 15:35:31 engine started: /srv/strata/src/build/strata-dsv41 --serve --pack /p "
              "--remote-experts 0.0.0.0:27861 --verbose\n")
# the engine's own lines (CT 580's engine.log, 2026-11-03 15:35-15:43 UTC, cut where they ran past 200 characters)
DS_LOAD = """ds4: build 77611561158f4e50363c8884ad9937bb50f78cbc (0.1.56)
ds4: remote experts: waiting up to 300 s for an expert server on 0.0.0.0:27861
"""
DS_SERVER = """ds4: remote experts: server 10.10.10.2 (NVIDIA GeForce RTX 3080, 9.57 GB of VRAM for experts): sending 1278 of the CPU lane's 15039 experts (9.57 GB, 23.0% of their routings in the profile)
ds4: expert parallelism H': k 6 - an EP window's CPU-lane picks that go to the expert server split at n_ff block 6 of 9
ds4: remote experts: the upload read 6.41 of its 9.57 GB from the disk (not in the page cache before it)
ds4: remote experts: 1278 experts (9.57 GB) in the server's VRAM, uploaded in 18.9 s (507 MB/s); a reply not in once the lane's own picks are done
ds4: remote experts: 16 warm-up requests of 1 row: round trip median 0.155 ms (0.150 - 0.495): the gate's estimate at 1 row starts there
ds4: remote experts: fills on (port 27862, 0.90 GB/s, 6 a token, 6 pending, the main's [0, 6) of a split layer; the fill thread on CPU 18)
maya ds4 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <pictures> <ids>; STOP, QUIT
"""
DS_PROMPT = ("maya ds4 serve: prompt 53 tokens = 0 reused + 53 read in 5123 ms (10.3 tok/s), 286 generated in 11045 ms "
             "(25.89 tok/s); CPU lane 4.3% of the experts\n")
DS_LATE = ("ds4: remote experts: a reply past its deadline (0.88 ms after the lane's part, 1.29 ms after its send; layer "
           "35): its 1 picks computed here (said once; the count is in the stats)\n")
DS_LOST = "ds4: remote experts: receive failed (the peer closed the connection) - the CPU lane computes all its experts from now on\n"
DS_NEVER = "ds4: remote experts: no expert server within 300 s - the CPU lane computes all its experts\n"
GPU_3080 = [{"name": "NVIDIA GeForce RTX 3080", "vram_mib": 10240, "used_mib": 9364, "util": 3, "temp_c": 49}]


def ds_engine(tok, log: str, args=None, env=None) -> RemoteEngine:
    return RemoteEngine(tok, log, args=args or list(DS_ARGS), env=env)


class ExpertsLogTest(LogCase):
    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()

    def test_no_link_of_either_kind(self):
        mon = stages.StageMonitor()
        self.assertIsNone(mon.view(MockEngine(self.tok, "x")))
        e = ds_engine(self.tok, self.log, args=["--pack", "/p", "--vision", "off"])
        self.assertIsNone(mon.view(e, ask=True))
        self.assertEqual(stages.remote_experts_arg(DS_ARGS), "0.0.0.0:27861")
        self.assertIsNone(stages.remote_experts_arg(["--remote-experts"]))      # (no value: not a listen address)

    def test_a_remote_stage_beside_it_keeps_the_stage_card(self):
        self.append(STARTED + MAIN_RUN)
        mon = stages.StageMonitor()
        mon.agents_cfg = False
        args = ["--stage-end", "26", "--remote-stage", "10.10.10.2:7841", "--remote-experts", "0.0.0.0:27861"]
        v = mon.view(ds_engine(self.tok, self.log, args=args))
        self.assertNotIn("kind", v)
        self.assertEqual((v["local_layers"], v["workers"][0]["address"]), ([0, 25], "10.10.10.2:7841"))

    def test_waiting_then_connected_then_requests(self):
        mon = stages.StageMonitor()
        mon.agents_cfg = False                                      # no agent is asked
        e = ds_engine(self.tok, self.log)
        e.running = False                                           # loading: the process runs, no READY yet
        self.append(DS_STARTED + DS_LOAD)
        v = mon.view(e, ask=True)
        self.assertEqual((v["kind"], v["listen"], v["agents"]), ("experts", "0.0.0.0:27861", False))
        (w,) = v["workers"]
        self.assertEqual((w["state"], w["address"], w["listen"]), ("connecting", None, "0.0.0.0:27861"))
        self.assertIn("waiting up to 300 s", w["error"]["text"])
        self.assertIsNone(w["experts"]["requests"])
        self.append(DS_SERVER)
        e.running = True
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["since"], w["error"], w["reconnects"]), ("connected", 1_000_000.0, None, 0))
        self.assertEqual((w["address"], w["gpu"]), ("10.10.10.2", {"name": "NVIDIA GeForce RTX 3080", "vram_mib": None,
                                                                    "used_mib": None}))
        x = w["experts"]
        self.assertEqual((x["held"], x["gb"], x["lane"], x["routing_pct"], x["vram_gb"], x["upload_s"]),
                         (1278, 9.57, 15039, 23.0, 9.57, 18.9))
        self.assertEqual((w["round_trip_ms"], x["fills"], x["fills_note"], x["requests"], x["late"]),
                         (0.155, True, None, 0, 0))
        self.append(DS_PROMPT * 3 + DS_LATE + DS_PROMPT)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["experts"]["requests"], w["experts"]["late"]), (4, 1))
        self.assertIn("layer 35", w["experts"]["late_note"]["text"])

    def test_lost_never_made_and_a_new_start(self):
        mon = stages.StageMonitor()
        mon.agents_cfg = False
        e = ds_engine(self.tok, self.log)
        self.append(DS_STARTED + DS_LOAD + DS_SERVER + DS_PROMPT + DS_LOST + DS_PROMPT)
        w = mon.view(e)["workers"][0]
        self.assertEqual(w["state"], "down")
        self.assertEqual(w["error"]["text"], "receive failed (the peer closed the connection)")
        self.assertEqual(w["experts"]["requests"], 1)               # the one after the link was given up: not its
        # the engine starts again and its expert server comes back: connected, one reconnect, a new count
        self.append(DS_STARTED + DS_LOAD + DS_SERVER.replace("1278", "1200") + DS_PROMPT)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["error"], w["reconnects"]), ("connected", None, 1))
        self.assertEqual((w["experts"]["held"], w["experts"]["requests"]), (1200, 1))
        # a start whose wait ran out: down with the engine's reason, nothing held
        self.append(DS_STARTED + DS_LOAD + DS_NEVER)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["address"], w["experts"]["held"]), ("down", None, None))
        self.assertEqual(w["error"]["text"], "no expert server within 300 s")
        e.running, e.proc = False, None                             # the engine ended
        self.assertEqual(mon.view(e)["workers"][0]["state"], "down")
        e.unloaded = True
        self.assertEqual(mon.view(e)["workers"][0]["state"], "unloaded")

    def test_fills_off_and_none_held(self):
        mon = stages.StageMonitor()
        mon.agents_cfg = False
        e = ds_engine(self.tok, self.log)
        self.append(DS_STARTED + DS_LOAD + DS_SERVER.replace(
            "fills on (port 27862, 0.90 GB/s, 6 a token, 6 pending, the main's [0, 6) of a split layer; the fill thread on CPU 18)",
            "no fills (the main's port 27862 is taken) - the server keeps the load's experts"))
        x = mon.view(e)["workers"][0]["experts"]
        self.assertEqual((x["fills"], x["fills_note"]), (False, "the main's port 27862 is taken"))
        self.append("ds4: remote experts: none held - the link core (CPU 18) goes back to the CPU lane\n")
        w = mon.view(e)["workers"][0]
        self.assertEqual(w["state"], "idle")

    def test_no_log_in_the_run_config(self):
        mon = stages.StageMonitor()
        e = ds_engine(self.tok, self.log)
        e.log_path = None
        self.assertEqual(mon.view(e)["workers"][0]["state"], "nolog")

    def test_a_long_log_is_scanned(self):
        """More than READ_MAX written since the last look: the lines scanned, the last of each kind taken, the requests
        after the last start counted."""
        mon = stages.StageMonitor()
        mon.agents_cfg = False
        e = ds_engine(self.tok, self.log)
        filler = "ds4: expert cache: settle 1 (one a prompt with the expert server)\n"
        n = stages.READ_MAX // len(filler) + 100
        self.append(DS_STARTED + DS_LOAD + DS_SERVER + DS_PROMPT * 7 + DS_STARTED + DS_LOAD + DS_SERVER + DS_PROMPT * 5
                    + filler * n + DS_PROMPT)
        w = mon.view(e)["workers"][0]
        self.assertEqual((w["state"], w["reconnects"], w["experts"]["held"]), ("connected", 1, 1278))
        self.assertEqual(w["experts"]["requests"], 6)               # the second run's: 5 scanned and 1 read

    def test_a_scan_counts_in_the_log_s_order(self):
        """A scanned part with the server taken, requests, late replies and then the link given up: the requests before
        the loss count, the ones after it do not, and the late-reply lines are counted with the last one's text."""
        mon = stages.StageMonitor()
        mon.agents_cfg = False
        e = ds_engine(self.tok, self.log)
        filler = "ds4: expert cache: settle 1 (one a prompt with the expert server)\n"
        n = stages.READ_MAX // len(filler) + 100
        self.append(DS_STARTED + DS_LOAD + DS_SERVER + DS_PROMPT * 3 + DS_LATE * 2 + DS_LOST + DS_PROMPT * 4 + filler * n)
        w = mon.view(e)["workers"][0]
        self.assertEqual(w["state"], "down")
        self.assertEqual((w["experts"]["requests"], w["experts"]["late"]), (3, 2))
        self.assertIn("layer 35", w["experts"]["late_note"]["text"])


class ExpertsAgentTest(LogCase):
    """The server PC's node agent: its /info gives the VRAM; its /log is never asked (it does not run the server)."""

    def setUp(self):
        super().setUp()
        self.tok = ByteTokenizer()
        self.ag = None

    def tearDown(self):
        if self.ag is not None:
            self.ag.close()
        super().tearDown()

    def test_the_stage_agents_map_of_the_same_pc(self):
        self.ag = FakeAgent(None, running=False, gpus=GPU_3080)
        asked = []
        real = self.ag.workers.log_path
        self.ag.workers.log_path = lambda port, prev=False: asked.append(port) or real(port, prev)
        self.append(DS_STARTED + DS_LOAD + DS_SERVER)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": self.ag.url}           # the remote stage's entry for this PC (Qwen's)
        v = view_now(mon, ds_engine(self.tok, self.log))
        w = v["workers"][0]
        self.assertTrue(v["agents"])
        self.assertTrue(w["agent"]["ok"], w["agent"])
        self.assertEqual(w["gpu"], {"name": "NVIDIA GeForce RTX 3080", "vram_mib": 10240, "used_mib": 9364})
        self.assertEqual(w["state"], "connected")                   # (the agent runs no worker: not "down" for it)
        self.assertEqual(asked, [])                                 # no /log of a port
        self.assertNotIn("PRIVATE KEY", json.dumps(v))

    def test_one_pc_as_a_worker_then_as_an_expert_server_and_back(self):
        """The same address asked as a stage worker (its log by port) and as an expert server (no log): each its own
        agent, whichever came first."""
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:7841": "https://127.0.0.1:9"}
        a = mon.agent_for("10.10.10.2:7841")
        self.assertEqual(a.port, 7841)
        b = mon.agent_for("10.10.10.2:7841", expert_server=True)
        self.assertIsNone(b.port)
        c = mon.agent_for("10.10.10.2:7841")
        self.assertEqual(c.port, 7841)
        self.assertIsNot(c, b)

    def test_an_agent_that_does_not_answer(self):
        self.append(DS_STARTED + DS_LOAD + DS_SERVER)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.2:27861": "https://127.0.0.1:9"}  # (nothing listens there)
        w = view_now(mon, ds_engine(self.tok, self.log, env={"STRATA_STAGE_TLS": ""}))["workers"][0]
        self.assertIs(w["agent"]["ok"], False)
        self.assertEqual((w["state"], w["gpu"]["name"], w["gpu"]["vram_mib"]), ("connected", "NVIDIA GeForce RTX 3080", None))

    def test_no_entry_for_it_in_a_map(self):
        self.append(DS_STARTED + DS_LOAD + DS_SERVER)
        mon = stages.StageMonitor()
        mon.agents_cfg = {"10.10.10.9:7841": "https://127.0.0.1:9"}  # another PC's: not asked
        w = view_now(mon, ds_engine(self.tok, self.log))["workers"][0]
        self.assertIsNone(w["agent"])
        self.assertEqual(mon.agents, {})


WEB = Path(__file__).resolve().parent / "web"
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "node is needed to run the page's own code")
class ExpertsCardPage(unittest.TestCase):
    """serve/web/app.js's renderWorkers with stand-ins for the page's helpers: the expert server's own columns, and the
    remote stage's header back when a stage comes after it."""

    def render(self, views: list) -> list:
        js = (WEB / "app.js").read_text(encoding="utf-8")
        block = js[js.index("const WK_STATE = "):js.index("// #596")]
        html = (WEB / "index.html").read_text(encoding="utf-8")
        head = html[html.index('<thead id="wk-head">') + len('<thead id="wk-head">'):html.index("</thead>", html.index('id="wk-head"'))]
        script = (
            "const els = {};\n"
            "const el = (id) => els[id] || (els[id] = {id, hidden: false, innerHTML: '', textContent: '', dataset: {}});\n"
            "const $ = el;\n"
            f"el('wk-head').innerHTML = {json.dumps(head)};\n"
            # the page's own esc / fmt / gb (its first lines), in the en-US locale (fmt's toLocaleString)
            + "".join(line + "\n" for line in js.splitlines()
                      if line.startswith(("const esc = ", "const fmt = ", "const gb = "))) +
            "const icon = () => '';\n"
            f"{block}\n"
            "const out = [];\n"
            f"for (const v of {json.dumps(views)}) {{ renderWorkers(v); out.push({{hidden: el('wk-card').hidden, head: el('wk-head').innerHTML, "
            "body: el('wk-body').innerHTML, sum: el('wk-sum').textContent, note: el('wk-note').textContent}); }\n"
            "console.log(JSON.stringify(out));")
        done = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True,
                              env={**os.environ, "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"})
        return json.loads(done.stdout)

    def test_untrusted_strings_are_escaped(self):
        """The GPU's name and the engine's reasons come from another PC and the log: never markup on the page."""
        tok = ByteTokenizer()
        bad = '<img src=x onerror="alert(1)">'
        with LogDir() as d:
            Path(d.log).write_text(DS_STARTED + DS_LOAD + DS_SERVER.replace("NVIDIA GeForce RTX 3080", bad)
                                   + DS_LOST.replace("receive failed", "<script>alert(2)</script>"), encoding="utf-8")
            mon = stages.StageMonitor()
            mon.agents_cfg = False
            (a,) = self.render([mon.view(ds_engine(tok, d.log))])
        self.assertNotIn("<img", a["body"])
        self.assertNotIn("<script", a["body"])
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", a["body"])
        self.assertIn("&lt;script&gt;alert(2)&lt;/script&gt;", a["body"])
        self.assertIn("Down", a["body"])

    def test_a_card_still_counting_says_at_least(self):
        v = {"kind": "experts", "listen": "0.0.0.0:27861", "provisional": True, "agents": False, "workers": [
            {"address": "10.10.10.2", "listen": "0.0.0.0:27861", "state": "connected", "gpu": {"name": "GPU"},
             "reconnects": 2, "round_trip_ms": None, "since": None, "error": None,
             "experts": {"held": 5, "lane": 9, "requests": 1234, "late": 3, "fills": True}}]}
        (a,) = self.render([v])
        for want in ("≥ 2", "≥ 1,234", "≥ 3", "catching up"):
            self.assertIn(want, a["body"], want)

    def test_experts_then_a_stage_then_nothing(self):
        tok = ByteTokenizer()
        with LogDir() as d:
            Path(d.log).write_text(DS_STARTED + DS_LOAD + DS_SERVER + DS_PROMPT * 2 + DS_LATE, encoding="utf-8")
            mon = stages.StageMonitor()
            mon.agents_cfg = False
            experts = mon.view(ds_engine(tok, d.log))
            Path(d.log2).write_text(STARTED + MAIN_RUN, encoding="utf-8")
            stage = stages.StageMonitor().view(RemoteEngine(tok, d.log2))
        a, b, c = self.render([experts, stage, None])
        self.assertFalse(a["hidden"])
        self.assertIn("Expert server", a["head"])
        self.assertIn("Experts held", a["head"])
        for want in ("NVIDIA GeForce RTX 3080", "10.10.10.2", "1,278", "9.57 GB", "23.0% of routings", "Connected",
                     "0.16 ms", ">2<", ">On<", ">1<"):
            self.assertIn(want, a["body"], want)
        self.assertNotIn("≥", a["body"])                           # (a log read whole: the counts are not lower bounds)
        self.assertIn("listening on 0.0.0.0:27861", a["sum"])
        self.assertIn("1,278 of the lane's 15,039 experts", a["sum"])
        self.assertIn("stage_agents", a["note"])
        # the remote stage's card as it was: its own header, layers, link
        self.assertIn("<th>Layers</th>", b["head"])
        self.assertNotIn("Expert server", b["head"])
        self.assertIn("26–47", b["body"])
        self.assertTrue(c["hidden"])


class LogDir:
    def __enter__(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.dir.name, "engine.log")
        self.log2 = os.path.join(self.dir.name, "strata.log")
        return self

    def __exit__(self, *a):
        self.dir.cleanup()


if __name__ == "__main__":
    unittest.main()
