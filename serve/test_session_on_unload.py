"""serve/test_session_on_unload.py - "session_on_unload" (opt-in): an unload saves the conversation to
slot_save_path/unload-session.bin before the engine stops, and the next load restores it (restore_after_load).

Off by default; on, the SAVE goes out before the engine stops and before after_unload runs; the load after it restores
the file and deletes it.  A save that fails, or that has not answered in session_on_unload_wait_s (the engine is ended
then), never stops the unload and leaves nothing to restore.  No save when the unload does not happen (a request runs,
a model_control switch), with parallel requests, or with an engine that has no session files.  (The engine's real
SAVE / RESTORE exchange over fake engine lines: serve/test_slots.py's FakeEngine.)

    python -m unittest serve.test_session_on_unload -v
"""
from __future__ import annotations

import os
import queue
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import UNLOAD_SESSION_FILE, ByteTokenizer, Service  # noqa: E402
from serve.test_server import CTX, UnloadableEngine  # noqa: E402
from serve.test_slots import FakeEngine, FakeProc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


class Engine(FakeEngine):
    """test_slots' engine (the server's own SAVE / RESTORE exchange), stopped and started like a real one."""

    def __init__(self):
        super().__init__()
        self.events: list = []                       # ("unload", the lines the stopped process was sent)

    def unload(self):
        self.events.append(("unload", list(self.proc.sent) if self.proc is not None else None))
        self.proc, self.unloaded = None, True

    def restart(self):
        self.ended, self.unloaded = False, False
        self.lines = queue.Queue()
        self.proc = FakeProc(self)


class SessionOnUnload(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.engine = Engine()
        self.engine.silence_s = 300.0
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.slot_save_path = self.dir.name
        self.svc.session_on_unload = True
        self.path = os.path.join(self.dir.name, UNLOAD_SESSION_FILE)

    def tearDown(self):
        p = getattr(self.engine.proc, "player", None) if self.engine.proc is not None else None
        if p is not None:
            self.engine.proc.killed = True
            p.join(timeout=5)
        self.dir.cleanup()

    def sent(self):
        """What the process the unload stopped was sent."""
        unloads = [e for e in self.engine.events if e[0] == "unload"]
        self.assertEqual(len(unloads), 1)
        return unloads[0][1]

    def test_off_by_default(self):
        svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.slot_save_path = self.dir.name               # (slot save/restore on, session_on_unload not)
        self.assertFalse(svc.session_on_unload)
        self.assertEqual(svc.unload(), "unloaded")
        self.assertEqual(self.sent(), [])
        self.assertIsNone(svc.restore_after_load)

    def test_saved_before_the_engine_stops_and_restored_after_the_next_load(self):
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.sent(), [f"SAVE {self.path}\n"])
        self.assertTrue(os.path.exists(self.path))
        self.assertEqual(self.svc.restore_after_load, self.path)
        self.assertFalse(self.svc.status["busy"])
        with self.svc.fifo:
            self.svc.ensure_loaded()
        self.assertEqual(self.engine.proc.sent, [f"RESTORE {self.path}\n"])
        self.assertIsNone(self.svc.restore_after_load)
        self.assertFalse(os.path.exists(self.path), "the file is deleted once restored")

    def test_a_restore_that_ends_the_engine_is_followed_by_a_fresh_start(self):
        # a remote main whose restore failed after the device state was changed: that engine is ended, and the
        # request that loaded it gets a new one (without the old cache), not the ended one
        self.assertEqual(self.svc.unload(), "unloaded")
        starts = []
        real = self.engine.restart

        def restart():
            real()
            starts.append(self.engine.proc)
            self.engine.script = None if len(starts) > 1 else [
                "SESSION 268435456 1198691396\n",
                "FATAL restoring the session file failed after the device state was changed: copy\n"]
        self.engine.restart = restart
        with self.svc.fifo:
            self.svc.ensure_loaded()
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0].sent, [f"RESTORE {self.path}\n"])
        self.assertTrue(starts[0].killed, "the engine the restore ended")
        self.assertEqual(starts[1].sent, [], "nothing restored into the new one")
        self.assertTrue(self.svc.loaded())
        self.assertIsNone(self.svc.restore_after_load)
        self.assertFalse(os.path.exists(self.path))

    def test_a_new_start_that_fails_after_a_fatal_restore_is_engine_died(self):
        from serve.server import EngineDied
        self.assertEqual(self.svc.unload(), "unloaded")
        real = self.engine.restart
        starts = []

        def restart():
            starts.append(1)
            if len(starts) > 1:
                raise OSError("no such program")
            real()
            self.engine.script = [
                "SESSION 268435456 1198691396\n",
                "FATAL restoring the session file failed after the device state was changed: copy\n"]
        self.engine.restart = restart
        with self.svc.fifo, self.assertRaises(EngineDied):
            self.svc.ensure_loaded()
        self.assertEqual(len(starts), 2)
        self.assertIsNone(self.svc.restore_after_load)

    def test_an_older_file_is_replaced(self):
        Path(self.path).write_bytes(b"old")
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(Path(self.path).read_bytes(), b"x" * 1234)

    def test_after_unload_runs_after_the_save(self):
        mark = Path(self.dir.name) / "mark"
        self.svc.after_unload = [PY, "-c", f"import os; open({str(mark)!r}, 'w').write(str(os.path.exists({self.path!r})))"]
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(mark.read_text(), "True")

    def test_a_failed_save_does_not_stop_the_unload(self):
        Path(self.path).write_bytes(b"old")              # an older file is not restored either
        self.engine.fail = True                          # SERR invalid: nothing to save
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.sent(), [f"SAVE {self.path}\n"])
        self.assertTrue(self.engine.unloaded)
        self.assertIsNone(self.svc.restore_after_load)
        self.assertFalse(os.path.exists(self.path))
        self.assertFalse(self.svc.status["busy"])
        self.engine.fail = False
        with self.svc.fifo:
            self.svc.ensure_loaded()
        self.assertEqual(self.engine.proc.sent, [], "nothing to restore after the load")

    def test_a_save_that_does_not_answer_in_time_is_given_up(self):
        # a step announced for an hour (a remote stage's worker writing its half), and then nothing
        self.engine.script = ["SWAIT worker 3600\n"]
        self.svc.session_on_unload_timeout_s = 0.5
        t0 = time.monotonic()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertLess(time.monotonic() - t0, 10)
        self.assertTrue(self.engine.ended, "the engine was ended (it cannot be put back in step mid-file)")
        self.assertIsNone(self.svc.restore_after_load)
        self.assertFalse(self.svc.status["busy"])

    def test_with_session_save_reclaim_the_save_is_retried_within_the_timeout(self):
        from unittest import mock
        import serve.session_save_retry as retry
        seen = []
        real = retry.session_file

        def spy(service, action, path, refused_type, deadline=None):
            seen.append((action, path, deadline))
            return real(service, action, path, refused_type, deadline=deadline)
        self.engine.session_save_reclaim = True
        self.svc.session_on_unload_timeout_s = 77
        t0 = time.monotonic()
        with mock.patch.object(retry, "session_file", spy):
            self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][:2], ("save", self.path))
        self.assertTrue(t0 + 77 <= seen[0][2] <= time.monotonic() + 77)
        self.assertEqual(self.svc.restore_after_load, self.path)

    def test_a_running_request_or_a_switch_saves_nothing(self):
        with self.svc.fifo:                              # a request holds the FIFO: busy, nothing sent
            self.assertEqual(self.svc.unload(), "busy")
        self.svc.switching = {"to": "other", "at": time.time(), "pid": 0}
        self.assertEqual(self.svc.unload(), "busy")      # model_control: another model, its own way
        self.assertEqual(self.engine.proc.sent, [])
        self.assertEqual(self.engine.events, [])

    def test_parallel_requests_save_nothing(self):
        self.engine.batch = 2
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.sent(), [])
        self.assertTrue(self.svc.session_on_unload_said)
        self.assertIsNone(self.svc.restore_after_load)

    def test_no_slot_save_path_saves_nothing(self):
        self.svc.slot_save_path = None
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertEqual(self.sent(), [])

    def test_an_engine_without_session_files_unloads_as_before(self):
        eng = UnloadableEngine(ByteTokenizer(), "ok", max_context=CTX)
        svc = Service(eng, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.slot_save_path, svc.session_on_unload = self.dir.name, True
        self.assertEqual(svc.unload(), "unloaded")
        self.assertFalse(eng.alive())
        self.assertIsNone(svc.restore_after_load)


if __name__ == "__main__":
    unittest.main()
