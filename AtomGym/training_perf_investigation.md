# Training Throughput Investigation (Solo, on AC)

Findings from a sequence of empirical sweeps, run on the user's laptop
(AMD Ryzen 9, 16 cores / 16 GB RAM, AC power, performance EPP). All
numbers below are from the on-AC runs only — battery-power results
were discarded since they don't reflect the actual training condition.

The TL;DR: **we got the win we were going to get**, the throughput
ceiling for solo training appears to be around 10000-10500 FPS, and
several follow-up ideas that sounded promising in theory (shared-memory
vec envs, EnvPool, Sample Factory) are not worth the engineering cost
because the bottleneck isn't IPC — it's somewhere in the sim+update
cycle balance.

---

## Headline result

```
              Before              After                  Δ
              ------              -----                  -
solo train    ~7800 FPS           ~10100 FPS            +30%
              n_envs=15           n_envs=20             same machine
              GPU/auto device     CPU device pin        same hyperparams
```

Two changes, in the order they landed, with reproducible measurements
on a clean system:

1. **Force `device='cpu'` for PPO.** The big lever. ~+1500 FPS.
2. **Bump `--n-envs` from 15 to 20.** ~+9% on top of the CPU win.

Both are now baked into `train.py`. Other knobs we tested (BLAS thread
caps, n_envs reductions, alternative vec env transports) either showed
no benefit or actively regressed throughput.

---

## What we tested

### 1. PPO `device='cpu'` ✅ (kept)

SB3 was raising:
```
UserWarning: You are trying to run PPO on the GPU, but it is primarily
intended to run on the CPU when not using a CNN policy ...
```

Our policy is a 128×128 MLP on 18-d obs → ~100k FLOPs per forward
pass. At that scale, GPU kernel-launch overhead dwarfs the actual
compute, and the cost of shipping rollout buffers across the PCIe bus
each PPO update is significant. Forcing CPU eliminates both costs.

**Result: +1500 FPS** observed live on the user's existing training
runs. Now hardcoded in `train.py` and `train_team.py` PPO construction
(both initial and `PPO.load()` resume paths).

This was by far the biggest single improvement of the entire
investigation, and it cost effectively nothing — one keyword argument.

### 2. n_envs sweep with SubprocVec ✅ (n_envs=20 adopted)

Sweep on AC with all other hparams matching production
(`--batch-size 1024 --n-steps 1024 --max-episode-steps 400 --use-subproc
 --manipulator default_pusher`):

| n_envs | Median FPS | Max FPS | Wall-clock (500k steps) |
|---:|---:|---:|---:|
| 8 | 5658 | 6613 | 94s |
| 15 | 9305 | 9826 | 62s |
| **20** | **10115** | **10632** | **59s** |
| 24 | (skipped — memory cap) | — | — |
| 28 | (skipped — memory cap) | — | — |

`n_envs=20` is the empirical sweet spot. Bumping from 15 to 20 gives
**+9%** with no memory drama.

We couldn't test n_envs ≥ 24 with SubprocVec because the worker memory
footprint (~375 MB PSS each) made the projected total exceed the 60%
RAM safety cap. The shared-memory experiment below addressed that.

### 3. Shared-memory vec env (`gymnasium.vector.AsyncVectorEnv`) ⚠️ (built, kept on standby, did NOT improve solo throughput)

Built `train_async.py` + a thin SB3 adapter (`_async_vec_env.py`) on
top of gymnasium's `AsyncVectorEnv(shared_memory=True)`. Hypothesis:
SubprocVec's pickle-over-pipe IPC was a real per-step bottleneck, and
shared memory would eliminate it.

**Head-to-head at n_envs=15:**

| Vec env | Median FPS | Max | Wall-clock |
|---|---:|---:|---:|
| SubprocVec | 9329 | 9726 | 61s |
| AsyncVec | 8806 | 9376 | 61s |
| Δ | **-5.6%** | -3.6% | tied |

AsyncVec was actually **slower** at n_envs=15. Two reasons:

1. **IPC wasn't a meaningful fraction of step time.** Our sim_py step
   is ~hundreds of µs (Box2D + reward eval + obs build). Pickle IPC
   per step is in the tens of µs. Eliminating it just isn't a big
   lever when the env itself is dominant.
2. **The adapter has its own overhead.** Gymnasium 1.x returns a
   recursively-vectorized info dict (e.g.,
   `info['episode'] = {'r': array(N), '_r': mask, 'l': array(N), ...}`).
   We have to un-vectorize that per-step into SB3's list-of-dicts
   format, which is Python loop overhead that SB3's native pickle
   path avoids.

**AsyncVec n_envs sweep** (using its memory savings to push higher):

| n_envs | Median FPS | Max | Notes |
|---:|---:|---:|---|
| 15 | 8919 | 9351 | -4% vs SubprocVec at same config |
| 20 | 9517 | 10070 | |
| 24 | 9766 | 10340 | |
| **28** | **9977** | **10471** | local peak |
| 32 | 9021 | 10632 | curve collapses |

AsyncVec peaks at n_envs=28 with median 9977 FPS — **statistically
tied** with SubprocVec at n_envs=20 (10115 FPS, ~1% gap). Not a win,
but not the regression we feared either. The shape of the curve
(steady climb 15-28 then collapse at 32) is also informative: **at
n_envs=32 the PPO update phase becomes a substantial fraction of the
cycle and eats the rollout-phase parallelism gains**, regardless of
which vec env transport is used.

**Why we kept the code anyway**: the per-worker memory footprint was
**5× lower** (see next section). For team training — where each
worker carries `torch + sim_py + numpy + OpponentRunner shadow policy`
≈ 600+ MB and we're more memory-constrained — that advantage may
unlock memory-blocked configurations even if it doesn't move the FPS
curve. Treat `train_async.py` as standby for that case.

### 4. Memory probe — the only place AsyncVec wins

| Metric | SubprocVec | AsyncVec |
|---|---:|---:|
| Tree PSS @ n_envs=4 | 2021 MB | **810 MB** |
| Per-worker PSS | 375 MB | **76 MB** |
| System delta | 1765 MB | **502 MB** |

AsyncVec workers use **5× less memory each**. They share the obs/action
buffers via `multiprocessing.shared_memory` and inherit a much smaller
import set (gymnasium's worker is leaner than SB3's). On a 16 GB box
this is the difference between fitting n_envs=20 and fitting n_envs=32
under the 60% safety cap.

For solo this didn't translate to FPS (curve collapses at high n_envs
anyway). For team it's the strongest reason to test the AsyncVec path.

### 5. BLAS thread caps ❌ (regression)

Tested various combinations of `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`. Results on AC:

| Setting | vs Baseline |
|---|---:|
| `OPENBLAS_NUM_THREADS=1` | tied (-0.7%) |
| `OMP_NUM_THREADS=4` | -10.7% |
| `OMP_NUM_THREADS=8` | -8.8% |
| `OMP_NUM_THREADS=1` | -10.3% |

Capping `OMP_NUM_THREADS` regressed throughput because it ALSO caps
torch's BLAS in the main process — the PPO update phase needs that
multi-threading. `OPENBLAS_NUM_THREADS=1` (which only affects numpy in
workers) was tied with baseline; it helped slightly on battery (where
power budget is constrained) but provided no measurable benefit on AC.

**Conclusion**: don't set BLAS thread caps. Workers' numpy contention
isn't a real problem on AC; the kernel scheduler handles 16 cores ×
small-pool-sizes fine.

### 6. Reduced n_envs ❌ (regression)

`n_envs=8` (half of cores) was tested as a hypothesis that SubprocVec's
IPC overhead grew with n_envs and fewer-but-fatter envs would win.
Result: -27% FPS. Confirms 15-20 is correctly sized.

---

## What is NOT worth pursuing (data-driven veto)

These are written down so future Claude sessions don't re-suggest them.

1. **EnvPool** — promised 5-10× via shared-memory + C++ batched env
   executor. The shared-memory part is what AsyncVec tested; that
   showed no benefit for our workload. The C++-batched-env part might
   help, but we'd need to re-bind sim_py against EnvPool's API
   (1-2 weeks of work). Cost-benefit ratio is bad given the data.

2. **Sample Factory / async PPO** — promised 5-10× via async (no
   rollout/update barrier). Worth flagging that even if this DID
   deliver, our PPO update phase already fits in ~1s on n_envs=20 —
   eliminating the barrier saves at most that fraction of cycle time.
   The bigger value of async PPO would be enabling truly large
   n_envs scaling, but the AsyncVec sweep showed our throughput
   collapses at n_envs > 28 from update-phase tax, not from the
   barrier cost.

3. **Per-worker BLAS thread reduction** — already tested. Doesn't
   help on AC.

4. **`n_envs` reduction (fewer-but-fatter)** — already tested.
   Worse.

5. **GPU device for PPO** — already vetoed; the CPU pin is the
   biggest win we got.

---

## Where the bottleneck likely lives

Based on the curve shape (peak at n_envs=20-28, collapse beyond), the
constraints appear to be:

- **Sim step cost** (sim_py + reward eval + obs build) — not dominant
  per-step but multiplied by every env step.
- **PPO update phase scaling** with rollout buffer size. At
  `n_envs=32, n_steps=1024, n_epochs=10, batch_size=1024`, that's
  320 gradient steps per update. Each step is fast (BLAS multi-thread,
  small batch) but the count grows linearly with n_envs.

The cycle balance: rollout phase scales roughly inversely with n_envs
(more parallelism = faster); update phase scales linearly with n_envs
(more samples = more grad steps). Their sum has a minimum around
n_envs=20-28. Past that, adding more envs slows total wall-clock.

If we wanted to push the ceiling higher, the levers are:
- **Reduce n_steps** (smaller rollout per update; risks PPO learning
  dynamics shift).
- **Increase batch_size** (faster update phase from BLAS; same).
- **Optimize sim_py** (always good, requires C++ work).
- **Reduce reward composite cost** (probably already lean).

None of these are obvious wins; they all involve learning-dynamics
tradeoffs. **Stopping here is the right call.**

---

## Production settings (post-investigation)

Use these for solo training:

```bash
.venv/bin/python -m AtomGym.training.train \
    --run-name <name> \
    --total-timesteps 16_000_000 \
    --n-envs 20 \
    --use-subproc \
    --batch-size 1024 \
    --n-steps 1024 \
    --max-episode-steps 400 \
    --ent-coef 0.01 \
    --log-std-init 0.3 \
    --manipulator default_pusher \
    [other flags...]
```

Expected throughput: 9500-10500 FPS on this machine.

The `train_async.py` script is preserved for the standby team-training
case but should NOT be the default for solo — it's slower at our
config.

---

## Bugs caught during the investigation (don't regress)

1. **Killed-the-wrong-PID in early benchmark scripts.** `bash`
   backgrounding (`cmd &`) returns the subshell PID, not the python
   PID nested inside. Killing PROBE_PID killed only the bash shell;
   the python kept running. We had a 7-process zombie tree at 750%
   CPU running for 2 hours before we noticed during a sweep that
   showed 2.6× regressed FPS.
   - **Fix**: launch via `PYTHONPATH=$REPO python -m ... &` so `$!`
     is the python directly. Walk descendants recursively for
     teardown — SubprocVecEnv's forkserver and workers are
     **grandchildren** of the main python, not direct children.

2. **`pkill -KILL -f "<pattern>"` matches the calling shell script
   if the pattern matches its filename.** Self-killed our cleanup
   step once. Use specific patterns like `--run-name perf_*` that
   only appear in python argvs, not in shell script filenames.

3. **`kill PROBE_PID` doesn't propagate to descendants.** Workers
   and forkservers survive the parent's termination. Always walk the
   tree (PID + descendants), SIGTERM, wait, then SIGKILL survivors.

4. **One probe with `n_envs=4` didn't capture worker count
   correctly.** SubprocVecEnv's tree topology is `python →
   forkserver → workers`, so `pgrep -P python_pid` returns only
   the forkserver (1 child). Recursive descendants walk is
   required. RSS-summing also overcounts shared library pages by
   2-3×; **PSS** (proportional set size, from
   `/proc/<pid>/smaps_rollup`) is the honest measurement. Cross-
   check with `MemAvailable` delta from `/proc/meminfo` for system-
   level ground truth.

These all live in the safe-sweep template scripts at `/tmp/perf_*.sh`
(don't ship the raw scripts; the lessons are now baked into this doc).

---

## Methodology

- Each sweep run: 500k env steps, batch_size=1024, n_steps=1024,
  max_episode_steps=400.
- FPS extracted from SB3's stdout `time/fps` rows. First 3 rows
  dropped as warmup-biased (cumulative since start). Median over
  remaining samples reported as the steady-state number.
- Memory probe: 60-second steady-state window at n_envs=4. Both PSS
  (proportional set size) and `MemAvailable` delta measured; larger
  used for safe pre-flight estimation.
- Pre-flight skip: if `main + n_envs × per_worker + 300 MB` exceeded
  60% of total RAM, the run was skipped without attempt.
- Live abort: between each run, `MemAvailable` checked; if dropped
  below 30% of total, sweep aborted to prevent freeze.
- Crash-safe journal at `~/perf_*_journal_<timestamp>.log`. `STARTING
  run X` lines are `fsync`'d before the heavy work. If the system
  freezes, last journal line identifies the in-flight test.

The journals + per-run logs from the final on-AC sweeps are at:
- `/tmp/perf_safe_20260502_162056/` — SubprocVec sweep
- `/tmp/perf_async_sweep_20260502_165528/` — AsyncVec sweep
- `/tmp/perf_async_20260502_164916/` — head-to-head at n_envs=15

These get cleaned up on reboot (`/tmp` is tmpfs on this system).
