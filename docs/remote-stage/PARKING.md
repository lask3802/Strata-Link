# Parked conversations across a remote stage

Status: design (2026-11-11), reviewed before it was built ([Review](#review)); [As built](#as-built) is filled in
with the code. Opt-in: nothing changes without `--conversation-cache-mib N` given explicitly to the main process.

## The problem

A remote main keeps one branch of conversation state. `CHECKPOINTS.md` says why: a checkpoint holds only running state,
so it is valid only while the K/V cells below it still hold its tokens, and every checkpoint that is not a prefix of
the next prompt is erased at the request's start. `generate.cpp` forces `--conversation-cache-mib` to 0 on a remote
main, so the outgoing conversation's K/V cannot be set aside either.

An agent client alternates one long conversation with side requests (a compaction summary, a title, a sub-agent).
Each side request starts from another system prompt, so it erases the long conversation's chain on both PCs, and the
long conversation's next turn reads its whole prompt again. Measured on 2026-11-10 (RTX 2080 Ti main, layers 0-19 and
the head; RTX 3080 worker on another PC, layers 20-47; 10 GbE; `--prefill 5888 --kv int8 --kv-resident 32768
--prompt-cache 6`, Qwen3.8 IQ3_S): before a compaction request each turn read `245570 tokens = 242141 reused + 3429
read`; after it, `247093 tokens = 0 reused`, 180.1 s before the first token (176.0 s and 178.2 s in the two repeats).

On one GPU the conversation cache ("parking") solves this: before a request overwrites the session, the outgoing
conversation (its K/V, its running state and its checkpoints) is copied to host RAM, and a later request that
continues it copies it back. This design does the same with the worker's half kept on the worker.

## What changes, in one paragraph

A parked conversation on a remote main has two halves. The main process's half is the image it builds today
(`SavedConversation`: its own layers' K/V and running state, its checkpoints, the drafter's K/V). The worker's half is
new: the K/V of the worker's own attention layers for the conversation's first L cells and the worker's running state
at L, stored on the worker under an id the main process gives it (as a checkpoint part is). The checkpoints inside a
parked conversation keep their worker parts: the main process goes on listing their ids as held. Restoring a parked
conversation copies both halves back whole; the request then goes on exactly as after a one-GPU restore (from the
live end, or through a checkpoint inside the image with `CkptRestore`).

## Opt-in and compatibility

- Main process: parking is on when `--conversation-cache-mib N` (N > 0) is given explicitly, `--prompt-cache N` is
  given explicitly (checkpoints on), and the worker of the current connection says it can park (below). The automatic
  sizing a one-GPU server does when the flag is absent stays off on a remote main.
- Worker: says so with a capability bit in its hello, `kStageCapPark = 1u << 30` in `row_formats` (bits 24-31 are
  capability bits; bit 31 is `kStageCapKvElem`). The hello compares only the K/V element bits (16-23) and the row
  format bits it knows (`stage_link.cpp` `hello_matches`, `stage_rows_agreed`), so an older main process ignores the
  bit. A worker lists it only when it can park: not a relay, not a worker whose own layers are split over several of
  its GPUs (its parts would need every local stage; not built), not with `--stage-park-mib 0`. The main process reads
  the bit again at every connection; without it parking stays off (one line) and nothing else changes.
- The protocol number stays 8: the new messages go only to a worker that listed the bit.
- Default path: a main process without the flag sends the same messages as before (the Reset payload, keep[], every
  other field). The only change on the wire is the worker's hello bit.

## Protocol

Two new messages, synchronous with a reply, carrying `int64 keep[]` as Reset / CkptSave / CkptRestore do:

| message | header | payload | reply |
|---|---|---|---|
| `ParkSave` (16) | a = id, b = L | `int64 keep[]` | `ParkSaveOk` (17: a = 1 stored, 0 not stored) or `Error` |
| `ParkRestore` (18) | a = id, b = L | `int64 keep[]` | `ParkRestoreOk` (19) or `Error` |

- Ids come from the main process's checkpoint counter, so checkpoint parts and parks share one id space and one keep[].
  The worker tells them apart by the message that created them. The counter is 32-bit and never wraps: at its end the
  main process makes no more checkpoints or parks (one line) instead of reusing an id.
- keep[] is the set of ids the main process holds anywhere, its "ownership roots": the live chain's checkpoints, every
  entry of the conversation cache (its park id and its checkpoints' ids), and the request's transient ones - a parked
  conversation already taken out of the cache to be restored (`incoming`), and the chain while the outgoing
  conversation is being captured. An id is held while any of them names it (a checkpoint can be in the chain and in a
  parked image at once: the same immutable part).
- With parking the list can pass 64 ids, so a worker that lists the bit reads up to `kStageKeepMax` = 256 ids in any
  keep[] (a main process sends more than 64 only when parks exist, which needs the bit). 256 bounds the payload; the
  checkpoint store itself still holds at most 64 parts, so a main process with many slots and checkpoints can meet that
  limit first (a CkptSave then answers "not stored", as today).
- `ParkSave(id, L, keep)`: the worker must be at L (a state error otherwise, as for CkptSave). It first drops every
  park and checkpoint part not in keep[], whatever the outcome (unlike CkptSave, whose refusal leaves the store as it
  was: here the main process has applied its evictions before it sends), then admits or refuses the new park
  ([Worker](#worker)), then saves: a device sync, its layers' running state at L (`conversation_checkpoint_save` with
  the explicit length), and the K/V of each owned attention layer for cells [0, L) (`conversation_kv_save`, which
  reads the authoritative host pool of a streamed K/V). a = 0: the park was not created; the prune happened.
- `ParkRestore(id, L, keep)`: a pending Commit error is logged and cleared, and a failed position table is reported
  (the request's own, as at a CkptRestore: the state class). The worker validates the whole park first (L, the layer
  carve, every layer's geometry) without writing anything, then restores with the existing helpers
  (`conversation_kv_restore`, which also refreshes the resident window of a streamed K/V, then
  `conversation_checkpoint_restore` with the explicit length). The session is at L. Then it prunes to keep[] (the park
  itself is not in it: the main process took it).
- There is no partial restore: a new conversation that starts with a checkpoint inside a parked one takes the parked
  conversation whole instead of borrowing its prefix (on one GPU it borrows; with a remote stage it behaves as a layer
  split does). So the worker's position is always known.
- Errors. A park the worker does not hold, or one that fails its validation before any write, answers with the
  `kStageCkptMissing` prefix: the main process reads the prompt from token 0 (the fallback of a missing checkpoint
  part). A failed copy after the first write, a position mismatch, a pending error on ParkSave, a failed position
  table: the state class, the engine ends as for a failed CkptSave (the server starts it again, the worker clears its
  store when the link closes).

## Main process

- Every message that carries keep[] (Reset, CkptSave, CkptRestore, ParkSave, ParkRestore) sends the ownership roots
  above. Every fallback that sent `session_from0({})` (keep nothing) keeps the roots still held instead, so a missing
  checkpoint part does not throw away the other parked conversations' parts.
- Parking the outgoing conversation (`park_source`, at the same call sites as on one GPU). After the main process's own
  admission (its budget and RAM floor, as today), before its capture and before STRATA_PARK_ASYNC moves any checkpoint
  out of the chain: `ParkSave(new id, live.size(), roots + new id)`. a = 0: one line, the conversation is not parked.
  a = 1: the main process captures its image and records the id in it (`SavedConversation::remote_park`); the chain's
  checkpoints are copied into the image with their `remote_id` (the pooled copy, `copy_checkpoint_into`, did not copy
  it: fixed). If the capture or `put` then does not keep the image, the worker's park is an orphan until the next
  keep[]. A remote main skips the park when its last pipelined window failed (`pl_drain_err`), as `checkpoint_at` does.
- Taking a parked conversation (`incoming`). After the main process's restore and after the image's checkpoints become
  the chain: `ParkRestore(id, L, roots)`. Missing: the main process zeroes its session and reads from token 0 with a
  Reset that keeps the roots still held.
- A link made again anywhere (`remote_relink`) drops every parked conversation together with what it drops today (the
  live session, the checkpoints): the new worker holds nothing, and the capability bit is read from its hello. A lost
  link found by a ParkSave at a request's start skips that park and abandons the request's resume decision (`incoming`
  dropped, `resume` 0): the request reads from token 0 with a Reset that keeps nothing. A ParkRestore that finds it
  lost does the same (the missing fallback).
- After non-finite logits (#879) the parked conversations are dropped already; the Reset keeps nothing.

## Worker

- `--stage-park-mib N`: the RAM the worker may give parked conversations and the checkpoint parts beyond the hello's
  `ckpt_max`. Default -1: half of the RAM available at the main process's hello, less the checkpoint share that hello
  checks, at most 16384 MiB. 0: no parking (the bit is not listed).
- The store keeps parts and parks by id with their byte counts; the main process alone decides what stays (keep[]).
  Each id is counted once, however many of the main process's images name it.
- Admission, for a ParkSave after its prune and for a CkptSave before it stores: after the save, the parks plus the
  checkpoint parts beyond `ckpt_max` must fit in the budget; and at the allocation's peak (what is held now, a CkptSave's
  doomed parts included, plus the new bytes) the RAM available (`available_host_bytes()`: MemAvailable, the cgroup
  limits, `--memory-limit-mib`) must stay above `--conversation-cache-min-free-mib` (default 2560). A RAM reading that
  is missing, a size that overflows or a host allocation that fails: not stored. A CkptSave refused keeps the store as
  it was.
- `disconnected`: the parks go with the checkpoint parts (a new main process is a new id space).

## Memory

From `conversation_snapshot.cpp` `layout()` (Qwen3.8 with `--kv int8`: 2 K/V heads of 256, an indexer key of 128
floats per 4 cells): per token and attention layer K 512 B, V 512 B, two scale rows of 16 B, the pooled indexer 128 B,
1,184 B; every 4th layer is an attention layer. At 247,093 tokens, with the page rounding and the indexer's spare row
(247,096 cells, 61,774 pooled rows), one attention layer is 292,561,664 B.

| | attention layers | K/V at 247,093 tokens | running state | one park with 6 checkpoint parts |
|---|---|---|---|---|
| Worker, layers 20-47 | 7 (23, 27, ..., 47) | 1.907 GiB | 68,655,132 B (65.5 MiB) | 1.971 GiB + 6 x 65.5 MiB = 2.355 GiB |
| Main, layers 0-19 + drafter | 5 + the draft layer (1,056 B a token, no indexer) | 1.61 GiB | about 49 MB plus PLE | about 1.93 GiB |

Computed from the layout code, not measured. A parked image on the main process holds its checkpoints too, so four
images of 247k tokens need about 7.7 GiB, and a swap holds the incoming image beside them (`held`).

Budgets on the two PCs of the measurement above (read 2026-11-11 with `free -g`, nothing changed):

- Worker PC: 62 GiB, the worker process 33.7 GiB resident, 24 GiB available. The default budget is about 11.5 GiB:
  four parks of 247k tokens with their checkpoints.
- Main PC (a container): 112 GiB, 64 GiB available. `--conversation-cache-mib 8192` holds three to four images of
  247k tokens; more needs a larger value.

## Eviction

The main process decides, with the cache it already has (`ConversationCache`): the least recently parked first when
the budget or `--conversation-cache-slots` is full, the RAM floor's own evictions, superseded copies of the same
conversation dropped before a park (#342), `--conversation-cache-min-tokens`. An entry it drops leaves the roots, and
the worker frees it at the next message that carries keep[]. A request that continues the live session sends none,
so a dropped park can stay on the worker until the next request that does; the worker's ParkSave prunes before it
admits, so a dropped park never makes it refuse a new one.

## Correctness: why a request cannot run on the wrong K/V

1. What a request may restore is decided as on one GPU: token ids, picture keys and the control-vector state must
   match (`ConversationCache::best`), and a checkpoint is mounted only after its whole parked conversation is back on
   both PCs (a park is restored whole, never as a prefix).
2. The worker's half is found by a unique id (never reused within a connection; the counter stops at its end) and
   checked against the L the main process sends; a park saved at another length is a state error, not a restore.
   Every copy of a checkpoint carries its `remote_id` (the pooled copy fixed, with a test).
3. Every park is a whole copy of cells [0, L) and the running state, taken while the worker was at L (checked), and a
   restore copies all of it back, the resident window refreshed, before anything reads it. A park shares nothing with
   the live session, so no later write changes it. (Copying only what changed since the conversation's last park
   would need the worker to track every write below it; left out.)
4. The worker validates a park before its first write; a failure after the first write ends the engine, so a half
   restored session is never decoded from.
5. A new connection (a restarted worker, a take-over) clears every part and park on the worker, and the main process
   drops every parked conversation when it connects again, so ids from two connections never meet.
6. The ownership roots include what the request holds outside the cache, so a ParkSave's prune cannot drop the
   conversation the same request is about to restore.

## Failure and fallback

| What happens | Result |
|---|---|
| The worker lacks the bit (older build, relay, split worker, `--stage-park-mib 0`) | Parking off, one line; as today |
| A park does not fit (budget, RAM floor, RAM unknown, allocation) | `ParkSaveOk a=0`: not parked, one line; as today for that conversation |
| A restore finds no park (pruned, restarted worker, invalid image) | That request reads from token 0; as today |
| The link is lost at a request's start (alive, ParkSave, ParkRestore) | Made again; parks dropped; the request reads from token 0 |
| A copy fails after its first write, a position mismatch | The engine ends, the server starts it again (CkptSave's class) |

## Cost and expected effect (estimated, not measured)

- A park copies about 1.9 GiB on the worker and up to 1.6 GiB on the main PC (less there: the main process keeps the
  K/V of the image it restored last and copies only what changed). Host to host at an assumed 4-8 GB/s with the first
  touch of new pages: 0.3-0.6 s on each side. A restore is about the same.
- A park happens when a request does not continue the live session: a side request, and a user turn whose template
  drops the earlier thinking (a tool call that extends the live session parks nothing). Such a turn pays about
  0.5-1.5 s more than today.
- The case measured above: the long conversation's turn after a compaction request read 247,093 tokens in about 178 s.
  With its park restored it resumes at its last turn boundary and reads the new part (3,429 tokens in the measured
  turn: about 2.4 s at the measured 1,438 tok/s) after the two restores: about 3-5 s instead of about 178 s.
- Not helped: the side request itself still reads its own prompt (158,895 tokens, 110 s for the compaction).

## Out of scope (not built)

- A relay worker, and a worker whose layers are split over its own GPUs: they do not list the bit.
- Borrowing a parked conversation's prefix (the remote main takes it whole instead).
- Keeping the unchanged prefix of the conversation's last park on the worker instead of copying it again.
- Overlapping the two halves' copies.
- `--batch` slots (a remote stage does not run with `--batch`).

## Checks

- Unit (no GPU): the worker's store (admission with the extra checkpoint parts and the allocation peak, a ParkSave's
  prune before admission, a refused CkptSave leaving the store as it was, ids counted once, clear); the new messages
  over a loopback link (fields, keep[] longer than 64, the bit, the position checks, missing vs other restore errors,
  a failed position table reported by a ParkRestore, a worker without the handlers); `copy_checkpoint_into` keeping
  `remote_id`; `ConversationCache::remote_ids`.
- Integration (no GPU): a loopback main and worker whose "session" is host memory written as a function of the tokens:
  a long conversation, a side request, the long conversation again, an eviction, a park the worker refuses, a lost
  park; the restored cells and state must equal what reading the prompt from 0 gives, and keep[] must free exactly the
  dropped parks.
- On GPUs (not run here): the CHECKPOINTS.md replay with a side request between turns, every repeated turn
  byte-identical and the restores in both logs; the default path byte-identical to the base build; the fallback with
  a worker debug switch that empties its parks; a worker restarted between turns.

## Review

Codex (a different model family, read-only, 2026-11-11) on the first draft: APPROVE WITH CHANGES. Taken in:

1. `copy_checkpoint_into` (the pooled checkpoint copy a park uses with STRATA_CKPT_POOL) did not copy `remote_id`.
2. A prefix restore left the worker's position unknown while Run / Prefill / Commit still ran: the borrow is not built.
3. A worker split over its own GPUs saves only its first session: it does not list the bit.
4. A link made again after the resume decision: the decision is abandoned and every park dropped (above).
5. Admission at the allocation's peak, each id counted once, unknown RAM refused.
6. keep[] must name the request's transient roots (an `incoming` already taken out of the cache): without that, a
   ParkSave of the outgoing conversation dropped the park of the one about to be restored.

Also corrected from the review: the memory table (page rounding, the checkpoints in a main image).
