"""remote_sessions_test.py's reading of its sessions, without a GPU: which difference between sessions is a finding.

    python -m unittest tools.dsv41.test_remote_sessions

The rows are the ones measured on CT 580 (RTX 2080 Ti, IQ1_M, one worker K 20 over the loopback, 2026-11-03): with
--vram-experts 0.3 the main placed 4 experts on the GPU in session 1 and 5 from session 2 on, and its answer changed at
token 6; with the main's placement fixed every session gave session 1's tokens.  A difference with the placement moved
is not settled until a run with the placement fixed (exit 3); one with session 1's placement counts is a finding about
what the processes kept across sessions (exit 1); a session without its placement line is no evidence either way (2).
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import remote_sessions_test as T  # noqa: E402

S1 = [881, 21002, 362, 12070, 305, 8049, 23914, 14, 98576, 10087]
S2 = [881, 21002, 362, 12070, 305, 8049, 20341, 12227, 14, 98576]


def row(i, tokens, cpu):
    return {"session": i, "tokens": tokens, "placement": {"expert_slots": cpu - 8440, "cpu_experts": cpu, "ram_experts": 0}}


class Verdict(unittest.TestCase):
    def test_the_share_on_one_gpu_moves_the_placement_not_a_finding(self):
        code, words = T.verdict([row(1, S1, 8060), row(2, S2, 8059), row(3, S2, 8059), row(4, S2, 8059)])
        self.assertEqual(code, 3)
        self.assertIn("s2, s3, s4", words)

    def test_the_placement_fixed_every_session_alike(self):
        self.assertEqual(T.verdict([row(1, S1, 8064), row(2, S1, 8064), row(3, S1, 8064)])[0], 0)

    def test_a_second_session_that_differs_with_the_same_placement_is_a_finding(self):
        code, words = T.verdict([row(1, S1, 8064), row(2, S2, 8064)])
        self.assertEqual(code, 1)
        self.assertIn("s2", words)

    def test_one_moved_and_one_not_is_still_a_finding(self):
        self.assertEqual(T.verdict([row(1, S1, 8060), row(2, S2, 8059), row(3, S2, 8060)])[0], 1)

    def test_no_session(self):
        self.assertEqual(T.verdict([])[0], 2)

    def test_a_session_without_its_placement_line_is_not_read_as_the_same_placement(self):
        blind = {"session": 2, "tokens": S2, "placement": {}}
        self.assertEqual(T.verdict([{"session": 1, "tokens": S1, "placement": {}}, blind])[0], 2)
        self.assertEqual(T.verdict([row(1, S1, 8064), blind])[0], 2)


class FirstDiff(unittest.TestCase):
    def test_positions(self):
        self.assertEqual(T.first_diff(S2, S1), 6)
        self.assertIsNone(T.first_diff(S1, list(S1)))
        self.assertEqual(T.first_diff(S1[:4], S1), 4)   # (one ended early: where it stopped)


class Placement(unittest.TestCase):
    def test_the_info_line_is_read(self):
        m = T.Main("x", [], {})
        m.info = ["INFO context=20000 kv=fp4 kv_resident=20000 expert_slots=-380 engine_kind=ds4 cpu_experts=8060 "
                  "ram_experts=0 engine=ds4-0.1"]
        self.assertEqual(m.placement(), {"expert_slots": -380, "cpu_experts": 8060, "ram_experts": 0})

    def test_the_load_line_s_free_vram_is_read(self):
        m = T.Main("x", [], {})
        m.stderr_lines = lambda: [
            "ds4: experts per word: 5 (faster mode; full quality: --experts 6)",
            "ds4: CUDA0 layers [0, 42] + head: weights 3.99 GB, experts 0.08 GB (4 of 8064 in VRAM), caches 0.01 GB, "
            "784 MiB free"]
        self.assertEqual(m.left_free(), 784)


if __name__ == "__main__":
    unittest.main()
