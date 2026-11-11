# One-job release processing

Current run: Slurm **18438738**, run root `/orion/u/howardji/realdata/realpc_run_1111`,
dataset `/orion/u/howardji/realdata/agentwm_v2`. Job submission is initially held
until staging, checksums and the code snapshot are ready. This is a code release;
full-dataset validation is pending until `status.json` says `complete`.

## Stage once

Use bounded rsync via the cluster transfer host, not many parallel cloud downloads.
The layout is:

```
code/                         checkout of this repository; immutable during run
packages/myenv.tar.gz          conda-pack of the tested Python/CUDA environment
packages/SHA256SUMS            checksum for the packed environment
runtime/bin/ffmpeg             portable ffmpeg
runtime/lib/                  compatible libstdc++.so.6 and libgcc_s.so.1
inputs/hf/{sweep,hang,greycloth}/episode_*/  52 RGB-D recordings
models/                       SAM2 checkpoint and CoTracker hub checkout/weights
references/cloth_inputs/       current-frame RGB-D/mask packets; QA only
references/masks/             optional independent evaluation masks; QA only
references/roundtrip_v4.json   published v1 encoder numbers
checkpoints/ae_wmstats.pt      FB learned SE(3) AE, zero-shot
plwm_tag/                     matching se3patch.py / se3latent.py sources
```

The v1 dataset stays at `/orion/u/howardji/realdata/agentwm_v1` for the matched
pre-fold cloth comparison. Link `code/.model_cache` to `../models`. Verify 52 input
episodes and checksum the environment/code assets before creating `READY`.
The wrapper unpacks the environment inside the allocation and compiles the small
cloth constraint kernel on demand. It never installs packages on a login node.

## Submit, monitor, resume

```bash
# Use only node names returned by sinfo; stale names can invalidate submission.
# Current exclusions: sphinx*, jagupard27/28/29/31/39, viscam3/5/9/10/15,
# tiger7, oriong13, iliad-hgx-1 (where present).
sbatch --hold --exclude=<comma-separated-existing-bad-nodes> scripts/cluster/run.sbatch
scontrol release <job-id>
squeue -j <job-id> -o '%.18i %.10T %.10M %.30R'
cat /orion/u/howardji/realdata/realpc_run_1111/status.json
```

Do not submit a second conversion. If preempted, the same requeued job reuses
fingerprinted successful stages. An ordinary failure is explicit in `status.json`
and its per-stage log; fix the cause before `scontrol requeue <job-id>`. Never mark
a failed stage complete manually. Conversion checks config/code/assets provenance
before reuse. Code changes invalidate stage markers, so preserve the frozen checkout.

Order: environment/import/native-op checks → tests → five sweep + hang + cloth
validation conversions and videos → three live parity checks → physical cloth
audit → remaining episodes (with QA/video immediately after each) → file load and
manifest checks → aggregate QA → encoder evaluation of every episode → three
round-trip videos and v1 comparison. Cloth visibility/shape is approximate; hard
physical violations stop release, while per-frame support/contact quality flags
are retained rather than hidden or replaced by a frame-index cutoff.

## Space and delivery

Inputs are ~10 GB; the packed/unpacked environment is ~15 GB combined. Small
pose/sample datasets and compressed masks are retained, not expanded point arrays
for every frame. The driver measures the run root plus dataset before stages and
stops at 47 GB or below 8 GB filesystem free. No automatic deletion occurs.

Run the transfer-only relay on the local PC, with its archive on the external disk:

```bash
python scripts/cluster/relay.py --remote howardji@scdt.stanford.edu \
  --run-root /orion/u/howardji/realdata/realpc_run_1111 \
  --dataset /orion/u/howardji/realdata/agentwm_v2 \
  --archive /mnt/external/plwm/realpc/run_1111 \
  --recipient howards-macbook-pro:
```

Use SSH authentication configured outside the repo; if needed, `--rsh 'sshpass -e ssh'`
reads a privately supplied SSHPASS environment variable. Never log credentials.
The relay polls every five minutes, copies reports/logs, checks completed video
hashes against job markers, and Taildrops uniquely named videos as they finish.
At completion it archives dataset_v2 and verifies SHA256SUMS. It does no extraction,
GPU work, model loading or job submission. Originals remain on Orion. If space
requires removing redundant intermediates, first verify the external copy and log
the exact deleted paths; dataset_v2, manifests and source assets are retained.
