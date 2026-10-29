// Strata. DS4_RX_PREWAKE (src/dsv41/ds4_rx.hpp): the wake request a prompt sends the expert server shortly before its
// end, so the server's GPU has woken when the decode's first requests come.
//   - rx::PreWake, the prompt's side: once an arm, when the last unit starts, or the last step's pace leaves lead_ms or
//     less, at the latest at the run's end;
//   - rx::Main against a stand-in server over loopback: the wake goes out as an ordinary 1-row request under a seq of
//     its own (kPrewakeSeq); its reply - or its Stale - read during a request's wait or by poll_wake is never that
//     wait's result, never in take_late (rx_finish's only way into rx::Backoff, rx::LateBook and the gate's EWMAs) and
//     never in stales (the Stale count of the stats); take_woke has it, with the server's time.  Replies of the
//     requests, late ones too, are as without it.
// No GPU, no model.
//   ds4_rx_prewake_test [PORT (27896)]
#include "../../src/dsv41/ds4_rx.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace rx = strata::ds4::rx;

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}
static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

int main(int argc, char** argv) {
    const std::string port = argc > 1 ? argv[1] : "27896";
    {
        rx::PreWake p;
        check(!p.due(0.0, 1.0, 1.0, true), "not armed: never due");
        p.arm(0.0);
        check(p.due(5.0, 0.0, 10.0, true), "a prompt of one window: due as it starts (its last unit)");
        check(!p.due(6.0, 0.0, 10.0, true), "and once an arm");
        p.arm(100.0);
        check(p.due(200.0, 1.0, 1.0, false), "a run's end: due, with no pace known");
    }
    {   // a layer-major run of 43 layers, 2 s a layer: due when its last layer starts
        rx::PreWake p;
        p.arm(0.0);
        bool any = false;
        for (int nd = 1; nd <= 41; ++nd) any = any || p.due(2000.0 * nd, nd, 43, nd >= 42);
        check(!any, "43 layers at 2 s a layer: not due before the last one (4 s left at 41)");
        check(p.due(2000.0 * 42, 42, 43, true), "due as the last layer starts");
    }
    {   // the pace: 2 s a layer for 20 layers, then 100 ms (the YOCO layers' few rows)
        rx::PreWake p;
        p.arm(0.0);
        double t = 0.0;
        int at = -1;
        for (int nd = 1; nd <= 43 && at < 0; ++nd) {
            t += nd <= 20 ? 2000.0 : 100.0;
            if (p.due(t, nd, 43, nd >= 42)) at = nd;
        }
        check(at == 33, "faster layers: due once the last step's pace leaves 1 s (10 layers of 100 ms at 33)");
    }
    {   // time within a step counts: 3 layers left at 500 ms a layer, 600 ms on
        rx::PreWake p;
        p.arm(0.0);
        check(!p.due(500.0 * 40, 40, 43, false), "1.5 s left: not due");
        check(!p.due(500.0 * 40 + 300.0, 40, 43, false), "1.2 s left: not due");
        check(p.due(500.0 * 40 + 600.0, 40, 43, false), "0.9 s left (600 ms into the step): due");
    }
    {   // lead_ms 0: only the last unit or the end
        rx::PreWake p;
        p.lead_ms = 0.0;
        p.arm(0.0);
        check(!p.due(1000.0, 1, 2, false), "lead 0: not due on the pace alone");
        check(p.due(1500.0, 2, 2, false), "lead 0: due at the end");
    }
    {   // rx::WakeSeries: at once, then each gap until the lane's next request, kReps a go() at most
        rx::WakeSeries w;
        w.gap_ms = 1000.0;
        check(!w.send_now(0.0), "no go(): nothing to send");
        w.go();
        check(w.send_now(0.0), "go(): one at once");
        w.sent_at(0.0);
        check(!w.send_now(999.0) && w.send_now(1000.0), "then the next a gap later");
        for (int i = 1; i < rx::WakeSeries::kReps; ++i) w.sent_at(1000.0 * i);
        check(!w.send_now(1e6), "kReps of them, then none");
        w.go();
        check(!w.send_now(7500.0) && w.send_now(8000.0), "a go() soon after the last one waits for its gap");
        w.stop();
        check(!w.send_now(9000.0), "the lane's request stops it");
    }
    {   // a chunked run as forward_chunks and serve_lane drive them (a 10 ms clock): the chunks' times, the settle
        // after the run, then the decode's first request - the last wake within a gap before that request
        struct Run {
            double first = -1.0, last = -1.0, req = 0.0, min_gap = 1e30;
            int n = 0;
        };
        auto run = [](std::vector<double> chunk_ms, double settle_ms) {
            rx::PreWake p;
            rx::WakeSeries w;
            w.gap_ms = 1000.0;
            Run r;
            double t = 0.0, end = 0.0;
            for (double c : chunk_ms) end += c;
            p.arm(0.0);
            auto lane = [&](double now) {
                if (!w.send_now(now)) return;
                if (r.last >= 0.0) r.min_gap = std::min(r.min_gap, now - r.last);
                if (r.first < 0.0) r.first = now;
                r.last = now;
                ++r.n;
                w.sent_at(now);
            };
            for (; t < end; t += 10.0) {   // (the run's wait loop: the chunks done by t)
                double s = 0.0;
                int done = 0;
                for (double c : chunk_ms)
                    if ((s += c) <= t) ++done;
                if (p.chunk_due(t, done, (double) chunk_ms.size())) w.go();
                lane(t);
            }
            if (p.run_end()) w.go();
            r.req = end + settle_ms;
            for (; t < r.req; t += 10.0) lane(t);
            return r;
        };
        const Run a = run({15000.0}, 3000.0);
        check(a.first >= 15000.0 && a.req - a.last <= 1000.0,
              "one chunk of 15 s, a settle of 3 s: no wake before its end, the last one within 1 s of the decode");
        const Run b = run({3000.0, 3000.0, 3000.0, 15000.0}, 500.0);
        check(b.first >= 10990.0 && b.first <= 11010.0 && b.req - b.last <= 1000.0 && b.min_gap >= 1000.0,
              "chunks of 3 s then a last one of 15 s: due on the pace (1 s left at 11 s), again at the run's end, the "
              "last within 1 s of the decode, a gap apart");
        const Run c = run({200.0, 200.0}, 6000.0);
        check(c.req - c.last <= 1000.0 && c.min_gap >= 1000.0 && c.n <= 2 * rx::WakeSeries::kReps,
              "a short run and the settle's 6 s: the series covers it, a gap apart");
    }
    // ------------------------------------------------------------------------------------------- rx::Main, loopback
    rx::Main m;
    rx::Conn s;
    std::string em, es;
    rx::Hdr h;
    std::vector<uint8_t> q;
    bool lm = false;
    std::thread tm([&] { lm = m.accept("127.0.0.1:" + port, "tok", 10, em); });
    sleep_ms(200);
    rx::HelloMsg hm;
    std::snprintf(hm.token, sizeof hm.token, "tok");
    bool ls = s.dial("127.0.0.1:" + port, 5, es) && s.send(rx::Hello, 0, 0, 0, 0, &hm, sizeof hm, es);
    tm.join();
    ls = ls && s.recv(h, q, rx::now_ms() + 1000.0, es) == 1 && h.type == rx::Hello;
    const int D = 8;
    rx::GeoMsg g;
    g.n_embd = D;
    ls = ls && lm && m.send_geo(g, {}, em) && s.recv(h, q, rx::now_ms() + 1000.0, es) == 1 && h.type == rx::Geo;
    check(ls, ("a main and a stand-in server (" + em + es + ")").c_str());
    if (!ls) return 1;
    std::vector<float> x(D, 0.01f);
    const float* xs[1] = {x.data()};
    std::vector<uint8_t> y(2 * D, 0);
    const uint16_t* yy = nullptr;
    int su = 0;
    rx::ReqRow wr;
    wr.n = 1;
    wr.e[0] = 17;
    wr.w[0] = 1.0f;
    {   // the wake on the wire: an ordinary request of 1 row, its seq kPrewakeSeq | 1
        const double t0 = rx::now_ms();
        const bool sent = m.prewake(3, wr, x.data(), em);
        const bool got = s.recv(h, q, rx::now_ms() + 1000.0, es) == 1;
        rx::ReqRow rr;
        if (got && q.size() >= sizeof rr) std::memcpy(&rr, q.data(), sizeof rr);
        check(sent && got && h.type == rx::Req && h.seq == (rx::kPrewakeSeq | 1u) && h.a == 3 && h.b == 1 && h.c == 0 &&
                  q.size() == sizeof(rx::ReqRow) + (size_t) D * 2 && rr.n == 1 && rr.e[0] == 17 && rr.slice == 0 &&
                  rx::row_refusal(rr, h.c, false).empty(),
              "the wake: a Req of 1 row (layer 3 expert 17, no slice bits: what any server takes), seq kPrewakeSeq | 1");
        check(m.prewake_out() && m.prewakes == 1, "it is out, counted as a wake");
        // a request (seq 5) after it: the server answers the wake first (22 ms of its time), then 5
        const bool s5 = m.send_req(5, 0, 1, &wr, xs, em);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        const bool r5 = h.type == rx::Req && h.seq == 5;
        s.send(rx::Resp, rx::kPrewakeSeq | 1u, 1, 0, 22000, y.data(), y.size(), es);
        s.send(rx::Resp, 5, 1, 0, 400, y.data(), y.size(), es);
        const int w5 = m.wait_resp(5, 1, rx::now_ms() + 1000.0, &yy, su, em);
        check(s5 && r5 && w5 == 1 && su == 400, "request 5 after it: its own seq, and its wait gets its own reply");
        check(m.take_late().empty(), "the wake's reply read during that wait: not in take_late (not late, nobody's picks)");
        check(m.stales == 0 && !m.prewake_out(), "no Stale counted; the wake is in");
        const auto wk = m.take_woke();
        check(wk.size() == 1 && wk[0].srv_us == 22000 && wk[0].layer == 3 && wk[0].e == 17 && wk[0].sent >= t0 &&
                  wk[0].read >= wk[0].sent,
              "take_woke: the wake's reply, the server's 22 ms");
#if defined(__linux__)
        check(!wk.empty() && wk[0].in >= wk[0].sent && wk[0].in <= wk[0].read,
              "Linux: its kernel arrival, between its send and its read (the round trip the line says)");
#endif
        check(m.take_woke().empty(), "and only once");
    }
    {   // a wake's Stale during a request's wait: not the wait's Stale, not counted
        m.prewake(3, wr, x.data(), em);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        const uint32_t ws = h.seq;
        m.send_req(6, 0, 1, &wr, xs, em);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        s.send(rx::Stale, ws, 1, 0, 0, nullptr, 0, es);
        s.send(rx::Resp, 6, 1, 0, 0, y.data(), y.size(), es);
        const int w6 = m.wait_resp(6, 1, rx::now_ms() + 1000.0, &yy, su, em);
        check(ws == (rx::kPrewakeSeq | 2u) && w6 == 1 && !m.last_stale() && m.stales == 0,
              "a wake's Stale: request 6 gets its reply, no Stale for it, stales 0");
        const auto wk = m.take_woke();
        check(wk.size() == 1 && wk[0].srv_us == -1 && m.take_late().empty(), "take_woke: the Stale (no GPU work)");
        // the request's own Stale is as before
        m.send_req(7, 0, 1, &wr, xs, em);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        s.send(rx::Stale, 7, 1, 0, 0, nullptr, 0, es);
        const int w7 = m.wait_resp(7, 1, rx::now_ms() + 1000.0, &yy, su, em);
        check(w7 == 0 && m.last_stale() && m.stales == 1, "a request's own Stale: 0, last_stale, counted (as before)");
    }
    {   // poll_wake, no request waiting: the wake's reply into take_woke, a late request's into take_late
        m.prewake(4, wr, x.data(), em);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        const uint32_t ws = h.seq;
        s.send(rx::Resp, 4, 1, 0, 0, y.data(), y.size(), es);   // (request 4's, late)
        s.send(rx::Resp, ws, 1, 0, 1500, y.data(), y.size(), es);
        int pr = 0;
        for (int i = 0; i < 200 && m.prewake_out() && pr == 0; ++i) {
            pr = m.poll_wake(em);
            if (m.prewake_out()) sleep_ms(5);
        }
        const auto lt = m.take_late();
        const auto wk = m.take_woke();
        check(pr == 0 && !m.prewake_out() && wk.size() == 1 && wk[0].srv_us == 1500 && wk[0].layer == 4,
              "poll_wake: the wake's reply read without a wait");
        check(lt.size() == 1 && lt[0].first == 4, "and request 4's late reply into take_late, as a wait would skip it");
        // nothing out: poll_wake reads nothing (a late reply stays for the next wait, as without it)
        s.send(rx::Resp, 9, 1, 0, 0, y.data(), y.size(), es);
        sleep_ms(20);
        const int p0 = m.poll_wake(em);
        check(p0 == 0 && m.take_late().empty(), "no wake out: poll_wake reads nothing");
        const int w10 = m.wait_resp(10, 1, rx::now_ms() + 5.0, &yy, su, em);
        const auto l9 = m.take_late();
        check(w10 == 0 && l9.size() == 1 && l9[0].first == 9, "the next wait skips 9 into take_late, as before");
    }
    {   // the speed path's poll (rx_poll: wait_resp with its deadline past) while only the wake's reply is in: not in
        // yet, not a Stale - its request's reply then as before
        uint32_t ws = 0;
        m.prewake(6, wr, x.data(), em, &ws);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        const bool same = h.seq == ws;
        m.send_req(12, 0, 1, &wr, xs, em);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        s.send(rx::Resp, ws, 1, 0, 300, y.data(), y.size(), es);
        sleep_ms(20);
        const int p12 = m.wait_resp(12, 1, 0.0, &yy, su, em, 1);
        const bool st12 = m.last_stale();
        s.send(rx::Resp, 12, 1, 0, 0, y.data(), y.size(), es);
        const int w12 = m.wait_resp(12, 1, rx::now_ms() + 1000.0, &yy, su, em);
        const auto wk = m.take_woke();
        check(same && ws == (rx::kPrewakeSeq | 4u), "prewake's seq out: the one on the wire");
        check(p12 == 0 && !st12 && w12 == 1 && m.take_late().empty() && wk.size() == 1 && wk[0].seq == ws,
              "a poll with only the wake's reply in: 0, no Stale; then request 12's own reply; the wake's in take_woke");
    }
    {   // only a wake still out matches - with one out, the warm-up's seqs (0xFFFF0000 + i) and a wake's seq again are
        // late, as before
        m.prewake(5, wr, x.data(), em);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        const uint32_t ws = h.seq;
        s.send(rx::Resp, 0xFFFF0003u, 1, 0, 0, y.data(), y.size(), es);
        s.send(rx::Resp, rx::kPrewakeSeq | 1u, 1, 0, 0, y.data(), y.size(), es);
        s.send(rx::Stale, rx::kPrewakeSeq | 2u, 1, 0, 0, nullptr, 0, es);
        s.send(rx::Resp, ws, 1, 0, 700, y.data(), y.size(), es);
        s.send(rx::Resp, 11, 1, 0, 0, y.data(), y.size(), es);
        const int w11 = m.wait_resp(11, 1, rx::now_ms() + 1000.0, &yy, su, em);
        const auto lt = m.take_late();
        const auto wk = m.take_woke();
        check(w11 == 1 && lt.size() == 2 && lt[0].first == 0xFFFF0003u && lt[1].first == (rx::kPrewakeSeq | 1u) &&
                  m.stales == 2,
              "a wake out, and high-bit seqs that are not it: late (their Stale counted), as without DS4_RX_PREWAKE");
        check(ws == (rx::kPrewakeSeq | 5u) && wk.size() == 1 && wk[0].srv_us == 700 && wk[0].layer == 5,
              "and the one out: take_woke's");
    }
    {   // the server's Error while a wake is out: poll_wake -1, the link shut
        m.prewake(3, wr, x.data(), em);
        s.recv(h, q, rx::now_ms() + 1000.0, es);
        const char why[] = "layer 3 expert 17 is not here";
        s.send(rx::Error, h.seq, 0, 0, 0, why, sizeof why - 1, es);
        int pr = 0;
        for (int i = 0; i < 200 && pr == 0; ++i) {
            pr = m.poll_wake(em);
            if (pr == 0) sleep_ms(5);
        }
        check(pr == -1 && !m.ok() && em.find("not here") != std::string::npos,
              "the server's Error: poll_wake -1 with it, the link shut");
        check(m.prewakes == 6, "6 wakes sent in all");
    }
    std::printf("%s\n", fails ? "FAILED" : "ALL OK");
    return fails ? 1 : 0;
}
