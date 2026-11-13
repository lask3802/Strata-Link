// include/strata/net/stage_park_store.hpp - a stage worker's half of the main process's parked conversations
// (docs/remote-stage/PARKING.md): its own layers' K/V of a conversation's first L cells and its running state at L,
// under the main process's id.
//
// As with the checkpoint parts (stage_ckpt_store.hpp), the main process alone decides what stays: every message with
// a keep[] names the ids it holds, and the store drops the others; it never evicts on its own.  A ParkSave prunes
// before it admits (the main process applied its evictions before it sent), so a park the main process dropped never
// makes the worker refuse a new one.  Header-only and templated on the payload, so stage_park_store_test walks it
// without a GPU (the worker's payload is its layers' ConversationKv and a ConversationCheckpoint).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::net {

/// What the worker may hold for a parking main process.
struct StageParkBudget {
    uint64_t budget = 0;      ///< bytes for the parks plus the checkpoint parts beyond ckpt_max (--stage-park-mib)
    uint64_t floor = 0;       ///< RAM that must stay available after an allocation (--conversation-cache-min-free-mib)
    int32_t ckpt_max = 0;     ///< the checkpoint parts the main process's hello asked for (its RAM was checked then)
    uint64_t part_bytes = 0;  ///< one checkpoint part
};

/// The admission of a new park or checkpoint part.  After the save the worker holds `parks_after` bytes of parks and
/// `parts_after` checkpoint parts; the parts beyond the hello's ckpt_max count against the budget with the parks (the
/// parked conversations' checkpoints, which that hello did not count).  The allocation itself adds `new_bytes` to what
/// is held now (a CkptSave's doomed parts included: they go after it), so the RAM `available` now must cover it and
/// keep `floor` beside it.  An unknown reading, or a size that overflows, is refused.  `why`: the refusal.
inline bool stage_park_fits(const StageParkBudget& b, uint64_t parks_after, size_t parts_after, uint64_t new_bytes,
                            std::optional<uint64_t> available, std::string& why) {
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    const uint64_t over = parts_after > (size_t) std::max<int32_t>(b.ckpt_max, 0)
                              ? (uint64_t) (parts_after - (size_t) std::max<int32_t>(b.ckpt_max, 0))
                              : 0;
    if (b.part_bytes != 0 && over > kMax / b.part_bytes) {
        why = "the size overflows";
        return false;
    }
    const uint64_t extra = over * b.part_bytes;
    if (parks_after > kMax - extra) {
        why = "the size overflows";
        return false;
    }
    if (parks_after + extra > b.budget) {
        why = "the budget (" + std::to_string((parks_after + extra) >> 20) + " MiB needed of " +
              std::to_string(b.budget >> 20) + " MiB, --stage-park-mib)";
        return false;
    }
    if (!available) {
        why = "the RAM available cannot be read";
        return false;
    }
    if (new_bytes > *available || *available - new_bytes < b.floor) {
        why = "the RAM floor (" + std::to_string(new_bytes >> 20) + " MiB more with " + std::to_string(*available >> 20) +
              " MiB available would leave less than " + std::to_string(b.floor >> 20) + " MiB)";
        return false;
    }
    return true;
}

template<class Payload>
class StageParkStore {
public:
    struct Park {
        int64_t L = 0;        ///< the position it was saved at (its length)
        uint64_t bytes = 0;   ///< what it holds, for the budget
        Payload payload;
    };
    const Park* find(int64_t id) const {
        const auto it = parks_.find(id);
        return it == parks_.end() ? nullptr : &it->second;
    }
    /// stores `id` (in place of one of the same id); the caller pruned and admitted first
    void put(int64_t id, Park park) {
        bytes_ -= find(id) != nullptr ? find(id)->bytes : 0;
        bytes_ += park.bytes;
        parks_.insert_or_assign(id, std::move(park));
    }
    /// drops every park `keep` does not name
    void prune(const std::vector<int64_t>& keep) {
        for (auto it = parks_.begin(); it != parks_.end();) {
            if (std::find(keep.begin(), keep.end(), it->first) != keep.end()) { ++it; continue; }
            bytes_ -= it->second.bytes;
            it = parks_.erase(it);
        }
    }
    void clear() {   ///< the main process went away: a new one is a new id space
        parks_.clear();
        bytes_ = 0;
    }
    size_t size() const { return parks_.size(); }
    uint64_t bytes() const { return bytes_; }
    /// the bytes held once pruned to `keep` (a CkptSave's admission counts what stays after it)
    uint64_t bytes_kept(const std::vector<int64_t>& keep) const {
        uint64_t n = 0;
        for (const auto& kv : parks_)
            if (std::find(keep.begin(), keep.end(), kv.first) != keep.end()) n += kv.second.bytes;
        return n;
    }

private:
    std::map<int64_t, Park> parks_;
    uint64_t bytes_ = 0;
};

/// A parking main process's keep[]: `v` (a chain's checkpoint parts, in its order, as a main process that does not park
/// sends it) followed by each id of `more` (the parked conversations' parks and parts, a conversation taken out) that is
/// not in it yet, once.  Without parks `more` is empty and keep[] is `v` unchanged - the bytes an older build sends.
inline std::vector<int64_t> stage_keep_roots(std::vector<int64_t> v, const std::vector<int64_t>& more) {
    for (const int64_t id : more)
        if (std::find(v.begin(), v.end(), id) == v.end()) v.push_back(id);
    return v;
}

/// The worker's default budget (--stage-park-mib -1): half of the RAM available at the main process's hello, less the
/// checkpoint share that hello checks (`ckpt_bytes`), at most `cap` (16 GiB).  0 when the RAM cannot be read.
inline uint64_t stage_park_auto_budget(std::optional<uint64_t> available, uint64_t ckpt_bytes,
                                       uint64_t cap = (uint64_t) 16 << 30) {
    if (!available) return 0;
    const uint64_t half = *available / 2;
    return std::min(cap, half > ckpt_bytes ? half - ckpt_bytes : 0);
}

}  // namespace strata::net
