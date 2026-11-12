"""serve/test_model_control.py - the after_unload hook and the Monitor's model card ("model_control"), both opt-in.

after_unload runs once the engine stopped on an unload (POST /unload, the idle unload), not when nothing was loaded or
a request runs; a failure or a timeout is kept for the card and the model stays unloaded.  model_control: the config is
checked at start; GET /model-control lists the models; POST /model-control runs the switch command with the target id
in a session of its own (202), refuses this server's own model (400), an unknown id (404), a running request or a
second switch (409), a form or a foreign page (415 / 403); while the switch runs a load is refused with 503 and
Retry-After; a switch command that ends while the server still runs clears the switch.

    python -m unittest serve.test_model_control -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, after_unload_setting, model_control_setting, serve  # noqa: E402
from serve.test_server import CTX, ROOT, UnloadableEngine  # noqa: E402

PY = sys.executable


def wait_for(pred, s=10.0):
    deadline = time.time() + s
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


class Base(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = UnloadableEngine(tok, "</think>\n\nok", max_context=CTX)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def req(self, path, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        r = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers=h, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, dict(resp.headers), json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, dict(e.headers), json.loads(e.read())

    def chat(self):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})


class AfterUnload(Base):
    def test_off_by_default(self):
        self.assertIsNone(self.svc.after_unload)
        self.assertEqual(self.req("/unload", {})[2], {"status": "unloaded"})
        self.assertIsNone(self.svc.after_unload_result)

    def test_runs_after_the_engine_stopped(self):
        mark = self.tmp / "ran"
        self.svc.after_unload = [PY, "-c", f"open({str(mark)!r}, 'w').close()"]
        self.assertEqual(self.req("/unload", {})[2], {"status": "unloaded"})
        self.assertFalse(self.engine.alive())
        self.assertTrue(mark.exists())
        self.assertEqual(self.svc.after_unload_result, "ok")

    def test_not_run_when_nothing_was_unloaded(self):
        mark = self.tmp / "ran"
        self.svc.after_unload = [PY, "-c", f"open({str(mark)!r}, 'w').close()"]
        self.svc.unload()
        mark.unlink()
        self.assertEqual(self.req("/unload", {})[2], {"status": "not loaded"})
        self.assertFalse(mark.exists())
        with self.svc.fifo:                                 # a request runs: busy, nothing stopped, nothing run
            self.svc.ensure_loaded()
            self.assertEqual(self.svc.unload(), "busy")
        self.assertFalse(mark.exists())

    def test_a_failure_is_kept_and_the_model_stays_unloaded(self):
        self.svc.after_unload = [PY, "-c", "import sys; sys.exit(3)"]
        self.assertEqual(self.req("/unload", {})[2], {"status": "unloaded"})
        self.assertFalse(self.engine.alive())
        self.assertIn("code 3", self.svc.after_unload_result)
        self.assertEqual(self.chat()[0], 200)               # the next request loads as usual

    def test_a_timeout_is_a_failure(self):
        self.svc.after_unload = [PY, "-c", "import time; time.sleep(30)"]
        self.svc.after_unload_timeout_s = 0.5
        t0 = time.time()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertLess(time.time() - t0, 10)
        self.assertIn("after_unload_timeout_s", self.svc.after_unload_result)

    def test_the_idle_unload_runs_it_too(self):
        mark = self.tmp / "ran"
        self.svc.after_unload = [PY, "-c", f"open({str(mark)!r}, 'w').close()"]
        self.svc.idle_unload_s = 1
        self.svc.last_request_at = time.time() - 5
        self.assertEqual(self.svc.unload(idle_for=1), "unloaded")
        self.assertTrue(mark.exists())

    def test_setting(self):
        self.assertIsNone(after_unload_setting(None))
        self.assertIsNone(after_unload_setting(""))
        self.assertEqual(after_unload_setting("echo hi"), "echo hi")
        self.assertEqual(after_unload_setting(["a", "b"]), ["a", "b"])
        for bad in (["a", 1], 5, {"cmd": "x"}):
            with self.assertRaises(ValueError):
                after_unload_setting(bad)


MODELS = [{"id": "qwen-sc117", "label": "Qwen3.8 Next Flash SC117"}, {"id": "deepseek", "label": "DeepSeek V4.1 Flash"}]


class ModelControlSetting(unittest.TestCase):
    def test_absent_is_off(self):
        self.assertIsNone(model_control_setting(None))

    def test_good(self):
        mc = model_control_setting({"this": "qwen-sc117", "models": MODELS, "switch": ["bash", "s.sh", "to"],
                                    "log": "/tmp/x"})
        self.assertEqual((mc["this"], [m["id"] for m in mc["models"]], mc["switch"], mc["log"]),
                         ("qwen-sc117", ["qwen-sc117", "deepseek"], ["bash", "s.sh", "to"], "/tmp/x"))
        one = model_control_setting({"this": "a", "models": [{"id": "a"}]})   # one model: no switch needed
        self.assertEqual((one["models"][0]["label"], one["switch"]), ("a", []))

    def test_bad(self):
        good = {"this": "qwen-sc117", "models": MODELS, "switch": ["s"]}
        for bad in ([], {"models": MODELS, "switch": ["s"]}, {**good, "this": "other"}, {**good, "models": []},
                    {**good, "models": [{"id": "a b"}, {"id": "qwen-sc117"}]},
                    {**good, "models": [{"id": "qwen-sc117"}, {"id": "qwen-sc117"}]},
                    {**good, "models": [{"id": "qwen-sc117", "label": 3}]},
                    {**good, "switch": []}, {**good, "switch": "s.sh"}, {**good, "switch": ["s", 1]},
                    {**good, "log": 5}, {**good, "models": [{"id": "../x"}, {"id": "qwen-sc117"}]}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                model_control_setting(bad)


class ModelControl(Base):
    def configure(self, script):
        self.args = self.tmp / "args.json"
        self.svc.model_control = model_control_setting(
            {"this": "qwen-sc117", "models": MODELS, "switch": [PY, "-c", script, str(self.args)],
             "log": str(self.tmp / "switch.out")})

    def test_off_by_default(self):
        self.assertEqual(self.req("/model-control")[2], {"enabled": False})
        s, _, b = self.req("/model-control", {"model": "deepseek"})
        self.assertEqual(s, 404)
        self.assertIn("model_control", b["error"]["message"])

    def test_view(self):
        self.configure("pass")
        s, _, b = self.req("/model-control")
        self.assertEqual(s, 200)
        self.assertEqual((b["enabled"], b["this"], [m["id"] for m in b["models"]], b["loaded"], b["switching"]),
                         (True, "qwen-sc117", ["qwen-sc117", "deepseek"], True, None))

    def test_switch_runs_the_command_with_the_target_and_clears_when_it_ends_here(self):
        # the command writes its argv and prints a line (to the log), then ends while this server still runs
        self.configure("import json, sys; json.dump(sys.argv[1:], open(sys.argv[1], 'w')); print('switch says hi')")
        s, _, b = self.req("/model-control", {"model": "deepseek"})
        self.assertEqual((s, b), (202, {"status": "switching", "to": "deepseek"}))
        self.assertTrue(wait_for(lambda: self.args.exists() and self.args.stat().st_size > 0))
        self.assertEqual(json.loads(self.args.read_text())[-1], "deepseek")
        self.assertTrue(wait_for(lambda: self.svc.switching is None), "a command that ended here: the switch is over")
        self.assertIn("switch says hi", (self.tmp / "switch.out").read_text())
        self.assertIsNone(self.req("/model-control")[2]["switching"])

    def test_while_switching_a_second_switch_and_a_load_are_refused(self):
        self.configure("import time; time.sleep(30)")
        self.svc.unload()
        self.assertEqual(self.req("/model-control", {"model": "deepseek"})[0], 202)
        view = self.req("/model-control")[2]
        self.assertEqual(view["switching"]["to"], "deepseek")
        self.assertEqual(self.req("/model-control", {"model": "deepseek"})[0], 409)
        s, h, b = self.chat()
        self.assertEqual(s, 503)
        self.assertIn("switching to deepseek", b["error"]["message"])
        self.assertTrue(int(h["Retry-After"]) > 0)
        self.assertEqual(self.engine.starts, 0)
        self.assertEqual(self.req("/load", {})[0], 503)
        import os
        import signal
        os.kill(self.svc.switching["pid"], signal.SIGTERM if hasattr(signal, "SIGTERM") else 15)
        self.assertTrue(wait_for(lambda: self.svc.switching is None))

    def test_refusals(self):
        self.configure("pass")
        self.assertEqual(self.req("/model-control", {"model": "qwen-sc117"})[0], 400)    # this server's: POST /load
        self.assertEqual(self.req("/model-control", {"model": "nope"})[0], 404)
        self.assertEqual(self.req("/model-control", {"model": 5})[0], 404)
        self.assertEqual(self.req("/model-control", ["deepseek"])[0], 400)
        with self.svc.fifo:                                                               # a request runs
            self.assertEqual(self.req("/model-control", {"model": "deepseek"})[0], 409)
        self.assertEqual(self.req("/model-control", {"model": "deepseek"}, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.req("/model-control", {"model": "deepseek"}, {"Origin": "https://example.com"})[0], 403)
        self.assertIsNone(self.svc.switching)
        self.assertFalse(self.args.exists())

    def test_a_command_that_cannot_run(self):
        self.svc.model_control = model_control_setting(
            {"this": "qwen-sc117", "models": MODELS, "switch": [str(self.tmp / "missing-binary")]})
        s, _, b = self.req("/model-control", {"model": "deepseek"})
        self.assertEqual(s, 500)
        self.assertIn("could not run", b["error"]["message"])
        self.assertIsNone(self.svc.switching)

    def test_the_api_key_guards_it(self):
        self.configure("pass")
        self.svc.api_key = "k"
        self.assertEqual(self.req("/model-control")[0], 401)
        self.assertEqual(self.req("/model-control", {"model": "deepseek"})[0], 401)
        self.assertEqual(self.req("/model-control", None, {"Authorization": "Bearer k"})[0], 200)


if __name__ == "__main__":
    unittest.main()
