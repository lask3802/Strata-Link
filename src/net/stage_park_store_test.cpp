// src/net/stage_park_store_test.cpp - a stage worker's half of the main process's parked conversations
// (docs/remote-stage/PARKING.md), walked by hand, no GPU:
//
//   1. the store: a park is found under its id with its length; put counts its bytes, the same id again replaces it
//      (counted once), prune drops what keep[] does not name, clear (a disconnect) empties it;
//   2. stage_park_fits: the parks' bytes and the checkpoint parts beyond the hello's count go against the budget; the
//      new allocation must leave the RAM floor; an unknown RAM reading and a size that overflows are refused, each
//      with its reason;
//   3. stage_park_auto_budget: half of the RAM available less the checkpoints' share, at most the cap, 0 unknown;
//   4. StageCkptStore::size_after: the parts held after a put with keep[] (the same id once, the others pruned);
//      stage_keep_roots: the main process's keep[] is the chain's as before, then each parked root once;
//   5. a main process's turns walked through both stores, as generate.cpp's worker runs them: a long conversation
//      parked, a side request, the long one restored (its park taken, its checkpoints kept), a park that does not
//      fit (refused, the store as it was), and the parks the main process evicted dropped at its next keep[].
#include "strata/net/stage_ckpt_store.hpp"
#include "strata/net/stage_park_store.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using strata::net::StageCkptStore;
using strata::net::StageParkBudget;
using strata::net::StageParkStore;
using strata::net::stage_park_auto_budget;
using strata::net::stage_park_fits;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
constexpr uint64_t MiB = 1ull << 20, GiB = 1ull << 30;
struct Half {   // what a park would carry: its K/V's tag and length
    int tag = 0;
};
StageParkStore<Half>::Park park(int64_t L, uint64_t bytes, int tag) {
    StageParkStore<Half>::Park p;
    p.L = L;
    p.bytes = bytes;
    p.payload.tag = tag;
    return p;
}
StageCkptStore::Part part(int64_t L) {
    StageCkptStore::Part p;
    p.L = L;
    return p;
}
}  // namespace

int main() {
    std::printf("stage_park_store_test\n");
    // 1. the store
    {
        StageParkStore<Half> s;
        check(s.size() == 0 && s.bytes() == 0 && s.find(1) == nullptr, "an empty store holds nothing");
        s.put(1, park(247093, 2 * GiB, 1));
        s.put(2, park(4000, 30 * MiB, 2));
        const auto* p = s.find(1);
        check(p != nullptr && p->L == 247093 && p->payload.tag == 1 && s.size() == 2 &&
                  s.bytes() == 2 * GiB + 30 * MiB,
              "a park is found under its id, with its length; the bytes add up");
        s.put(2, park(5000, 40 * MiB, 3));
        check(s.size() == 2 && s.find(2)->L == 5000 && s.find(2)->payload.tag == 3 && s.bytes() == 2 * GiB + 40 * MiB,
              "the same id again replaces its park, counted once");
        check(s.bytes_kept({2, 9}) == 40 * MiB && s.bytes_kept({}) == 0 && s.bytes_kept({1, 2}) == s.bytes(),
              "bytes_kept counts only what a keep[] names");
        s.prune({2, 7, 9});
        check(s.size() == 1 && s.find(1) == nullptr && s.bytes() == 40 * MiB,
              "prune drops what keep[] does not name (ids it names but the store lacks: nothing)");
        s.prune({2});
        check(s.size() == 1 && s.bytes() == 40 * MiB, "a keep[] that names it keeps it");
        s.clear();
        check(s.size() == 0 && s.bytes() == 0 && s.find(2) == nullptr, "a disconnect clears the store");
    }
    // 2. the admission
    {
        StageParkBudget b;
        b.budget = 8 * GiB;
        b.floor = 2 * GiB;
        b.ckpt_max = 7;
        b.part_bytes = 66 * MiB;
        std::string why;
        check(stage_park_fits(b, 2 * GiB, 7, 2 * GiB, 24 * GiB, why), "a park within the budget and above the floor");
        check(!stage_park_fits(b, 8 * GiB + 1, 0, 1, 24 * GiB, why) && why.find("budget") != std::string::npos,
              "parks over the budget are refused, the reason names --stage-park-mib");
        check(stage_park_fits(b, 8 * GiB - 7 * 66 * MiB, 14, 66 * MiB, 24 * GiB, why),
              "the parts beyond the hello's count go against the budget: exactly full fits");
        check(!stage_park_fits(b, 8 * GiB - 7 * 66 * MiB, 15, 66 * MiB, 24 * GiB, why) &&
                  why.find("budget") != std::string::npos,
              "... and one part more does not");
        check(!stage_park_fits(b, 2 * GiB, 7, 2 * GiB, 4 * GiB - 1, why) && why.find("floor") != std::string::npos,
              "an allocation that would leave less than the floor is refused");
        check(stage_park_fits(b, 2 * GiB, 7, 2 * GiB, 4 * GiB, why), "... one that leaves exactly the floor is not");
        check(!stage_park_fits(b, 2 * GiB, 7, 3 * GiB, 2 * GiB, why) && why.find("floor") != std::string::npos,
              "an allocation larger than the RAM available is refused");
        check(!stage_park_fits(b, 1, 0, 1, std::nullopt, why) && why.find("cannot be read") != std::string::npos,
              "an unknown RAM reading is refused");
        StageParkBudget huge = b;
        huge.budget = std::numeric_limits<uint64_t>::max();
        huge.part_bytes = std::numeric_limits<uint64_t>::max() / 2;
        check(!stage_park_fits(huge, 1, 10, 1, 24 * GiB, why) && why.find("overflows") != std::string::npos,
              "parts whose size overflows are refused");
        huge.part_bytes = 1;
        check(!stage_park_fits(huge, std::numeric_limits<uint64_t>::max(), 10, 1, 24 * GiB, why) &&
                  why.find("overflows") != std::string::npos,
              "parks plus parts that overflow are refused");
        StageParkBudget none = b;
        none.ckpt_max = -1;   // (a hello without checkpoints: every part counts)
        check(!stage_park_fits(none, 8 * GiB - 66 * MiB, 2, 66 * MiB, 24 * GiB, why),
              "a negative ckpt_max counts every part");
    }
    // 3. the default budget
    {
        check(stage_park_auto_budget(24 * GiB, 7 * 66 * MiB) == 12 * GiB - 7 * 66 * MiB,
              "half of the RAM available less the checkpoints' share (pve-5900x: 24 GiB)");
        check(stage_park_auto_budget(64 * GiB, 0) == 16 * GiB, "at most 16 GiB");
        check(stage_park_auto_budget(64 * GiB, 0, 4 * GiB) == 4 * GiB, "at most the cap given");
        check(stage_park_auto_budget(1 * GiB, 1 * GiB) == 0, "nothing when the checkpoints take the half");
        check(stage_park_auto_budget(std::nullopt, 0) == 0, "nothing when the RAM cannot be read");
    }
    // 4. the checkpoint store's count after a put
    {
        StageCkptStore c;
        check(c.size_after(1, {1}) == 1, "an empty store holds one after a put");
        c.put(1, part(10), {1});
        c.put(2, part(20), {1, 2});
        check(c.size_after(3, {1, 2, 3}) == 3 && c.size_after(3, {2, 3}) == 2 && c.size_after(2, {1, 2}) == 2,
              "the same id counts once, the parts keep[] does not name do not count");
        check(c.size_after(3, {}) == 1, "a keep[] without the others: the new one alone");
    }
    // 4b. the main process's keep[]: the chain's ids as before, then each parked root once
    {
        using strata::net::stage_keep_roots;
        check((stage_keep_roots({3, 1, 2}, {}) == std::vector<int64_t>{3, 1, 2}),
              "no parks: the chain's keep[] exactly as an older build sends it");
        check((stage_keep_roots({3, 1}, {5, 1, 6, 5, 3}) == std::vector<int64_t>{3, 1, 5, 6}),
              "parked roots appended once each, after the chain, none repeated");
        check((stage_keep_roots({}, {7, 7, 8}) == std::vector<int64_t>{7, 8}),
              "a reset that drops the chain keeps the parked roots only");
    }
    // 5. a main process's turns through both stores (what generate.cpp's worker handlers do with them)
    {
        StageCkptStore cks;
        StageParkStore<Half> parks;
        StageParkBudget b;
        b.budget = 5 * GiB;
        b.floor = 2 * GiB;
        b.ckpt_max = 7;
        b.part_bytes = 66 * MiB;
        const std::optional<uint64_t> ram = 24 * GiB;
        std::string why;
        const uint64_t long_half = (uint64_t) (1.971 * (double) GiB);   // 247,093 tokens (PARKING.md)
        // CkptSave: admitted, then put, then the parks keep[] does not name go
        auto ckpt_save = [&](int64_t id, int64_t L, const std::vector<int64_t>& keep) {
            if (!cks.admits(id, keep)) return 0;
            if (!stage_park_fits(b, parks.bytes_kept(keep), cks.size_after(id, keep), b.part_bytes, ram, why)) return 0;
            cks.put(id, part(L), keep);
            parks.prune(keep);
            return 1;
        };
        // ParkSave: prune first, whatever comes of it; then the budget
        auto park_save = [&](int64_t id, int64_t L, uint64_t bytes, const std::vector<int64_t>& keep) {
            cks.prune(keep);
            parks.prune(keep);
            if (!stage_park_fits(b, parks.bytes() + bytes, cks.size(), bytes, ram, why)) return 0;
            parks.put(id, park(L, bytes, (int) id));
            return 1;
        };
        // ParkRestore: found at its length, then pruned to keep[] (the park itself not in it)
        auto park_restore = [&](int64_t id, int64_t L, const std::vector<int64_t>& keep) {
            const auto* p = parks.find(id);
            if (p == nullptr || p->L != L) return false;
            cks.prune(keep);
            parks.prune(keep);
            return true;
        };
        // the long conversation: three checkpoints on its path, then parked when a side request comes
        check(ckpt_save(1, 100000, {1}) && ckpt_save(2, 200000, {1, 2}) && ckpt_save(3, 245570, {1, 2, 3}),
              "the long conversation's checkpoints are stored");
        check(park_save(4, 247093, long_half, {1, 2, 3, 4}) && parks.size() == 1 && cks.size() == 3,
              "the long conversation parked: its park and its checkpoints kept");
        // the side request reads from 0 (a Reset keeping the parked roots) and saves its own checkpoint
        cks.prune({1, 2, 3, 4});
        parks.prune({1, 2, 3, 4});
        check(ckpt_save(5, 158895, {5, 1, 2, 3, 4}) && cks.size() == 4 && parks.size() == 1,
              "a side request's checkpoint beside the parked roots");
        // the next turn of the long conversation: the side request parked (its own half), the long one restored
        check(park_save(6, 159200, (uint64_t) (1.3 * (double) GiB), {5, 6, 1, 2, 3, 4}) && parks.size() == 2,
              "the side request parked too");
        check(park_restore(4, 247093, {1, 2, 3, 5, 6}) && parks.size() == 1 && parks.find(4) == nullptr &&
                  cks.size() == 4,
              "the long conversation restored whole: its park taken, its checkpoints and the side request's kept");
        // a park that does not fit: refused, the stores as they were (the prune to its keep[] stands)
        check(!park_save(7, 247200, 4 * GiB, {1, 2, 3, 5, 6, 7}) && parks.size() == 1 && cks.size() == 4 &&
                  why.find("budget") != std::string::npos,
              "a park over the budget is refused; the stores as they were");
        // the main process evicted the side request's copy: its next keep[] drops its park and its part
        check(ckpt_save(8, 247300, {1, 2, 3, 8}) && parks.size() == 0 && cks.size() == 4 && cks.find(5) == nullptr,
              "the next keep[] without the evicted conversation drops its park and its checkpoint part");
        check(!park_restore(6, 159200, {}), "an evicted park is missing (the main process reads from 0)");
    }
    if (g_fail == 0) std::printf("stage_park_store_test: all passed\n");
    else std::printf("stage_park_store_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
