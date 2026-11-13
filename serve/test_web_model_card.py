"""serve/test_web_model_card.py - the Monitor's Model card (opt-in "model_control", serve/test_model_control.py has the
server side).

Pins, as text: index.html holds the card hidden in the Monitor view; app.js reads GET model-control, shows the card only
when the server says it is enabled, leaves Unload and this server's Load to the header button (its checks and confirm),
POSTs model-control for another model only after a confirm, and reloads once the port answers again.

    python -m unittest serve.test_web_model_card -v
"""
from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"


class WebModelCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (WEB / "index.html").read_text(encoding="utf-8")
        cls.js = (WEB / "app.js").read_text(encoding="utf-8")
        cls.css = (WEB / "app.css").read_text(encoding="utf-8")

    def test_card_is_in_the_monitor_and_hidden_until_enabled(self):
        monitor = self.html[self.html.index('id="view-monitor"'):]
        self.assertIn('<div class="st-card mc-card" id="mc-card" hidden>', monitor)
        for el in ('id="mc-model"', 'id="mc-load"', 'id="mc-unload"', 'id="mc-note"', 'id="mc-state"'):
            self.assertIn(el, monitor)
        self.assertIn('card.hidden = !(mc && mc.enabled);', self.js)
        self.assertIn('fetch("model-control", {headers: headers()})', self.js)
        self.assertIn(".mc-row", self.css)

    def test_this_servers_load_and_unload_go_through_the_header_button(self):
        self.assertIn('$("mc-unload").onclick', self.js)
        self.assertIn('$("unload-btn").click()', self.js)

    def test_a_switch_asks_first_and_posts_the_id(self):
        i = self.js.index('$("mc-load").onclick')
        body = self.js[i:i + 2500]
        self.assertLess(body.index("confirm("), body.index('fetch("model-control", {method: "POST"'))
        self.assertIn("JSON.stringify({model: target})", body)
        self.assertIn("r.status !== 202", body)
        self.assertIn("location.reload()", body)

    def test_the_card_polls_only_on_the_monitor(self):
        self.assertIn('if (tab === "monitor" && mcpTick % 5 === 0) loadModelControl();', self.js)

    def test_a_server_without_the_card_is_asked_once(self):
        i = self.js.index("async function loadModelControl()")
        body = self.js[i:i + 400]
        self.assertLess(body.index("if (mcOff) return;"), body.index('fetch("model-control"'))
        self.assertIn("if (r.status === 404) { mc = null; mcOff = true;", body)


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class WebModelCardParses(unittest.TestCase):
    def test_app_js_parses(self):
        r = subprocess.run(["node", "--check", str(WEB / "app.js")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
