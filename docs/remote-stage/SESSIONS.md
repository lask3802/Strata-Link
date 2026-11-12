# Session files across a remote stage

Status: built (2026-11-12) and checked without a GPU ([Checks](#checks)). Not run on GPUs yet: no number on this page
is a measurement of this feature. Where the code differs from the design it was built from, [As built](#as-built)
says so. Opt-in on both PCs: nothing changes without `--stage-session-dir` on the worker, and the server saves on an
unload only with `"session_on_unload": true` in its config.

## The problem

A remote stage runs one model over two PCs: the main process (layers 0 to K-1 and the head) and a worker on another
PC (layers K and up). An unload stops both (the server's `"after_unload"` stops the worker,
[DETAILS.md](../DETAILS.md#sharing-the-gpu-with-other-programs-optional)), so their RAM goes to other work, and with them goes
the conversation's K/V. The next request of a long agent conversation reads its whole prompt again. PARKING.md
measured that read on the split below: 247093 tokens, 180.1 s before the first token.

Session files (`SAVE` / `RESTORE`, `POST /slots/0?action=save|restore`, DETAILS.md "Session files (disk)") keep a
conversation across a restart on one GPU. A remote main refused them: its file would hold only its own layers.

## What changes, in one paragraph

A remote main's session file has two halves. The main process writes its half where `SAVE` names, as on one GPU (its
own layers' K/V and running state, the deepest checkpoint). The worker writes its half to its own disk, in the folder
`--stage-session-dir` names: the K/V of its layers for the conversation's L cells, its running state at L, and its part
of the checkpoint the main file keeps. A random 128-bit token joins the two: it names the worker's file and is written
in the main file's header. `RESTORE` reads the main file, asks the worker to read and check its half whole and put it
back, and only then writes its own half.

## Opt-in and compatibility

- Worker: `--stage-session-dir DIR` (created mode 0700 when missing; it must be a folder) and `--stage-session-keep N`
  (default 4: the N newest worker halves are kept, older ones are deleted after each save). The worker lists
  `kStageCapSession = 1u << 29` in its hello's `row_formats` (bits 24-31 are capability bits; 30 is parking, 31 the
  K/V element type). A relay, or a worker whose own layers are split over several of its GPUs, does not list it and
  says so. Both options on a process that is not a worker stop it at start.
- Main process: it reads the bit at every connection. Without it, `SAVE` / `RESTORE` are refused as before ("session
  files do not support --remote-stage").
- The protocol number stays 8: the new messages go only to a worker that listed the bit. An older main process
  ignores the bit.
- Default path: with neither option, the hello's bits are the only difference on the wire (none is set), the one-GPU
  file is byte for byte what it was (`conversation_file_test`'s golden file: same size, same hash), and a one-GPU
  engine reads and writes files as before.
- Files across engines: a remote main's file has a non-zero token in the header's reserved words. An engine that does
  not ask for a token (a one-GPU engine, or an older build: both check those words are zero) refuses it ("invalid
  header fields"). A remote main refuses a file with no token ("the file has no remote stage half"). A remote main's
  file is bound to its split (K: the worker runs layers K and up) as well as to the usual model and settings, so a
  file made at split K is refused at another split; the worker's half is bound to the worker's own layer range.

## Protocol

Two messages, synchronous with a reply. The payload starts with the token (two uint64) and Lc (int64), 24 bytes:

| message | header | payload | reply |
|---|---|---|---|
| `SessionPut` (20) | a = the id of the checkpoint part at Lc (0: none), b = L | token, Lc | `SessionPutOk` (21: a = bytes written) or `Error` |
| `SessionGet` (22) | a = the id the worker keeps the checkpoint under (0: none), b = L | token, Lc, `int64 keep[]` | `SessionGetOk` (23) or `Error` |

- `SessionPut` is checked as a `ParkSave` is: after every reply and window before it, only where the worker's session
  is (L), never empty, never without a token, and a checkpoint id and length go together (`0 <= Lc <= L`). A failed
  Commit or position table is its error, and stays pending (a park drops a failed Commit, because the session is
  replaced right after it; a SAVE replaces nothing, so the next request must still find it). It does not move the
  worker's session.
- `SessionGet` reads at most `kStageKeepMax` ids in keep[] from a worker that parks, `kStageCkptMax` from one that does
  not (as the other messages with a keep[]). An error that starts with `kStageCkptMissing` (the worker has no such
  half, or the half failed a check before anything was written) leaves the worker's session where it was. Any other
  failure leaves no position trusted (the next message must be a Reset or a restore).
- The worker's file errors start with their kind (`invalid: `, `storage: `, `memory: `, `io: `): the main process
  answers its `SERR` with that kind.
- The reply comes once the worker has written or read its whole file. The main process announces the wait to the
  server as one blocking step, `SWAIT worker <s>` with s = the time for 16 KiB a token at 4 MiB/s
  (`session_phase_limit_s`) plus 120 s, at most 3600, and the link waits that long for the reply's first byte instead
  of `STRATA_REMOTE_TIMEOUT_S`.

## Save

1. The main process's usual checks (a complete session, `--prompt-cache` > 0, RAM for its own half), and two more on a
   remote main: the link is alive, and the last pipelined window did not fail.
2. A new token. `SessionPut`: the worker writes `DIR/stage-session-<32 hex digits>.bin` (temporary file, then rename;
   the K/V goes from its pools into the file 16 MiB at a time, with no host copy of it), then deletes the oldest
   halves past `--stage-session-keep`. A failure here publishes nothing on either side.
3. The main process writes its own file with the token in the header. If that fails, the worker's half is an orphan;
   the keep-N rule deletes it later.

The main file keeps the deepest checkpoint only (as on one GPU). On a remote main a checkpoint without a worker part
is not kept: the file then has no checkpoint.

## Restore

1. The main process reads its whole file and checks it (as on one GPU, plus the token).
2. `SessionGet` with a new checkpoint id (when the file has a checkpoint) and keep[] = that id plus what the main
   process still holds (its parked conversations, unless the restore follows a NaN flush). The worker reads its whole
   half into RAM (admitted against `--conversation-cache-min-free-mib`, as a park is), checks it against its session
   (model, settings, layer range, L, Lc, every part's size), and only then writes it to the device. Its checkpoint
   part goes into its store under the new id; its stores keep only keep[].
3. Only after the worker's success does the main process write its own half.

| what fails | answer (`SERR` kind) | after it |
|---|---|---|
| the worker has no such half, or its half fails a check | invalid (400) | both sides as they were |
| the link is lost | io (500) | the next request connects again |
| the worker fails after it started writing | io (500) | the main process's live session is dropped; the next request reads from a checkpoint or from token 0 |
| the main file is a one-GPU file, or has more than one checkpoint | invalid (400) | as it was |

## The server: `"session_on_unload"`

`"session_on_unload": true` (needs `"slot_save_path"`; the server does not start without it) makes an unload (`POST
/unload`, the Monitor's Unload, the idle unload) save the conversation before the engine stops, to
`<slot_save_path>/unload-session.bin` (an older one is replaced). The next load restores it and then deletes the main
file, as it does for an image job's file. It works on one GPU too.

- It runs under the request queue, after the unload's own checks (a running or waiting request, a model switch: no
  unload, no save) and before the engine stops and `"after_unload"` runs. Meanwhile the server's status is busy, in
  the phase "saving the session".
- It never stops the unload. A failed save (nothing to save, short of RAM or disk, a worker without
  `--stage-session-dir`) is one log line, and any older `unload-session.bin` is deleted so it is not restored. A save
  that has not answered in 120 s ends the engine (the two sides cannot be put back in step in the middle of a file,
  and the unload was about to stop it) and the unload goes on.
- With parallel requests (`"parallel"` / `--batch`) it is off, said once at start.
- A failed restore after the load is one log line; the request reads its prompt as without the file.
- The model_control switch to another model does not save (it does not go through the unload).

## Files on each PC

| PC | file | written by | deleted by |
|---|---|---|---|
| main | `<slot_save_path>/unload-session.bin`, or the `/slots` file name | `SAVE` | the restore after the load (`session_on_unload` only); `/slots` files: nobody |
| worker | `<--stage-session-dir>/stage-session-<token>.bin` | `SessionPut` | the keep-N rule at the next save |

Both are mode 0600 on Linux; on Windows they inherit the folder's permissions. The worker's file has its own magic
(`STRWSES`, version 1) and end marker, and the same 64-byte header layout and hash as the main file: a file of one kind
is refused as the other. The keep-N rule deletes only names of the exact pattern `stage-session-<32 lower-case hex
digits>.bin`, never the file just written.

A worker half is deleted only by the keep-N rule, so a `/slots` file saved more than N saves ago cannot be restored
on a remote main any more (its worker half is gone; the restore answers 400 and changes nothing).

## What is not kept

- Parked conversations other than the live one, and every checkpoint but the deepest (as on one GPU).
- The worker's checkpoint parts of other chains.
- A file across a change of split, of the worker's layer range, of model files or of the settings that change what the
  cells mean, on either PC.

## Memory and time (not measured)

- Save: the worker copies its running state at L and its checkpoint part (host copies; their size is the worker's
  share of the recurrent state) and streams its K/V to disk. The main process's half costs what it costs on one GPU.
- Restore: the worker reads its whole half into host RAM before it writes anything (so a bad file changes nothing):
  the file's size, admitted against MemAvailable less `--conversation-cache-min-free-mib`.
- Neither the file sizes at 240K tokens nor the save and restore times have been measured.

## Checks

Run (no GPU, Windows, MSVC 2022 with CUDA 13.0 headers, `CUDA_VISIBLE_DEVICES=-1`):

- `conversation_file_test` (added to): the token's hex and file name; a main file with a token (the token under the
  header hash, the rest of the file as without one; a read that asks for no token refuses it, one that does returns
  it); the worker half's round trip with and without a checkpoint; every rejection before the caller's image is touched
  (another token, no token, another model or configuration, another layer range, L or Lc, another geometry, more K/V
  layers or a larger K/V part than the worker holds, the RAM preflight, torn files at every boundary, trailing bytes,
  flipped bytes in the payload and the header, a main file read as a worker half and back, a missing file); writes
  refused before any disk access (no token, empty, Lc past L); temp + rename (a failed rename or a full disk keeps the
  old file, no temporary left; the free-space preflight); mode 0600; the keep-N rule (the newest stay, the one just
  written always, other names untouched, a link of that name not followed, a missing folder); `--stage-session-dir` made absolute and created (0700 on
  Linux), a file refused. The one-GPU golden file is unchanged.
- `stage_link_test` section 19 (new): SessionPut / SessionGet over a TLS loopback link: the bit; the position, empty,
  token and checkpoint checks, none reaching the handler; the token (high bits set), Lc, the id and a 33-bit byte count
  carried whole; SessionPut does not move the session; a write error comes back with the link and session kept; a
  keep[] of 201 ids, one of 257 refused with the link in step; the session at L after a SessionGet, its checkpoint
  under the new id; missing (no such half, another length) leaves the session where it was, any other failure leaves no
  position; a failed Positions and a failed Commit on both messages, the Commit's still pending after a SessionPut
  and reaching the next prompt chunk; a reply allowed longer than
  `STRATA_REMOTE_TIMEOUT_S` read as any other; a worker without the handlers (no bit, errors that are not the missing
  one, at most 64 ids).
- `conversation_validation_test` (added to): a state's check at an explicit length (one without ids at any length
  within the cells, one with ids only at its own); a worker half's read limits for this session (the caller's own
  fields kept), the half of this session read back within them and validated, another length refused, and the limits
  of another carve, or a half with one K/V layer more than the carve owns, refused.
- Server (`python -m unittest serve.test_session_on_unload serve.test_model_control`): off by default; saved before the
  engine stops and before `after_unload`, restored after the next load and deleted; an older file replaced; a failed
  save does not stop the unload and leaves nothing to restore; a save with no answer in time is given up and the engine
  ended; no save while a request runs, during a switch, with parallel requests, without `slot_save_path`, or with an
  engine that has no session files; the config key (needs `slot_save_path`, true or false, off with `"parallel"`).
- Each new check was seen to fail on a mutated build (the reserved-word check, the token check, the keep-N rule and its
  name filter, the token in the header, the layer range check, the read limits' layer count, the explicit-length
  check, the worker's position after a missing half, the Lc check, the failed Commit kept pending by a SessionPut,
  the server's save call, its delete of an older file, its parallel check, its config check, the restore's clear, its
  deadline: without it the test waits out the announced 3600 s).
- The tests that passed before pass after it: `conv_cache_test`, `conversation_cache_test`, `conversation_memory_test`,
  `conversation_split_failure_test`, `session_save_reclaim_test`, `split_rules_test`, `split_walk_test`,
  `stage_ckpt_store_test`, `stage_park_store_test`, `stage_reserve_test`, and the serve tests that touch unloading and
  session files (`test_slots`, `test_images`, `test_session_save_retry`, `test_lifecycle`, `test_model_control`).
  `conversation_copy_streams_test` and `conversation_snapshot_test` need a GPU and skip.

Not run:

- Anything on GPUs: a save and a restore on the two-PC split, the file sizes, the times, the worker's RAM peak at
  restore, the `SWAIT worker` allowance against a real save.
- The main process's SAVE / RESTORE code in `generate.cpp` (it cannot run without a model): checked by review only.
- `conversation_part_sources` (the worker's state and K/V gathered for the file) needs a device: not run.
- A Linux (GCC) build of this change, and so the checks that run only there (the files' mode 0600, the folder's 0700).
- The keep-N rule's link check: this Windows PC may not make symbolic links, so it was skipped, and the fix it guards
  (links not followed) was not seen to fail without it.

## As built

Where the code differs from the design it was built from, or says more than it:

- A lost link answers `SERR io` (500), not a new kind: the kinds are fixed, and an older server ends an engine that
  answers an unknown one as out of step.
- A `SessionGet` that fails after the worker started writing answers `SERR io` and drops the main process's live
  session, instead of ending the engine: the main process has written nothing, and the worker's stores are intact.
- A `SessionGet` that fails before any write (the missing case) leaves the worker's position where it was; a
  `ParkRestore` sets it unknown in every case.
- A failed Commit before a `SessionGet` is its error, not the missing one (a `ParkRestore` drops it with a log
  line), so the main process drops its live session as after any other failure; the next request reads from a
  checkpoint or from token 0.
- A failed Commit or position table before a `SessionPut` stays pending after it (a `ParkSave` keeps only the
  position table's).
- The worker streams its K/V to disk on a save (no host copy) but reads its whole half into RAM on a restore.
- A temporary file left by a crash in the middle of a write is not deleted (two workers might share a folder); a
  worker half whose main file was never written is deleted by the keep-N rule. The same holds on the main PC when the
  server's 120 s bound ends the engine during its write: its hidden temporary file stays in `slot_save_path`, up to the
  size of a session file, until someone deletes it.
- The server's 120 s bound on the save ends the engine at the bound instead of waiting for it; it is a constant
  (`UNLOAD_SESSION_WAIT_S`), not a config key.
- A failed save before an unload deletes an older `unload-session.bin`, also one a failed save had already published.

## Open risks for the GPU run

- The `SWAIT worker` allowance is a guess (16 KiB a token plus 120 s for the worker's first fingerprint of its model
  files). Too short and the server ends the engine during a save.
- The server's 120 s bound covers both halves and the main process's own fingerprint: a slow disk at 240K tokens may
  pass it.
- The worker's RAM at restore: the whole half in host RAM while its model is loaded.
- The worker's take-over timer (`STRATA_REMOTE_TAKEOVER_S`) counts idle time between messages; a long `SessionPut` or
  `SessionGet` is one message, so a second main process waiting to take over is not served during it.
