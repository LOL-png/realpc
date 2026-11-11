# Reproducible single-GPU batch processing

The runner uses one GPU allocation for environment checks, tests, validation,
conversion, QA, manifests and learned/analytic encoder evaluation. It never
submits child jobs. Historical running jobs use their own staged code snapshots;
updating this repository does not update or restart them.

## Stage inputs

Choose a staging directory (`REALPC_RUN_ROOT`) and a **separate, non-overlapping**
dataset output directory (`REALPC_DATASET`). Populate:

```
code/                         frozen checkout of this repo
packages/myenv.tar.gz          conda-pack of a tested environment
packages/SHA256SUMS            archive checksum
runtime/bin/ffmpeg             portable ffmpeg
runtime/lib/                  matching libstdc++.so.6 / libgcc_s.so.1
inputs/hf/{sweep,hang,greycloth}/episode_*/  calibrated recordings
models/                       SAM2 and CoTracker checkout/weights
references/cloth_inputs/       evaluation-only current RGB-D/mask packets
references/masks/             optional independent evaluation masks
references/roundtrip_v4.json   published baseline encoder numbers
checkpoints/ae_wmstats.pt      trained SE(3) AE checkpoint
plwm_tag/                     matching encoder source package
```

Link `code/.model_cache` to `../models`. Verify inputs/models/code before creating
`READY`. The current collection driver expects 50 sweep, one hang and one cloth
episode. Set `REALPC_V1_DATASET` for the baseline dataset used by the matched cloth
comparison. This reference dataset is evaluation-only and does not enter tracking.
The wrapper unpacks the environment and checks its GPU/native operations inside
the allocation. A preinstalled environment can instead invoke `run_all.py` directly.

## Submit a new job

```bash
export REALPC_RUN_ROOT=/path/to/staged_run
export REALPC_DATASET=/path/to/dataset_v2
export REALPC_V1_DATASET=/path/to/dataset_v1
export REALPC_PIPELINE=accurate  # or fast
mkdir -p "$REALPC_RUN_ROOT/logs"
sbatch --partition=YOUR_GPU_PARTITION --account=YOUR_ACCOUNT \
  --constraint='ampere|ada|hopper' \
  --output="$REALPC_RUN_ROOT/logs/slurm-%j.out" \
  --error="$REALPC_RUN_ROOT/logs/slurm-%j.err" scripts/cluster/run.sbatch
```

Choose valid partitions/accounts/constraints and node exclusions for your cluster.
The wrapper requests one GPU, eight CPUs, 96 GB RAM and 48 hours. Read `status.json`,
`logs/` and `markers/` under the run root. Successful stages are fingerprinted by
commit, config and checkpoint; requeue the same job after fixing an interruption.
Keep its checkout unchanged. A code tag alone does not certify dataset completion.
The driver checks combined working/output space, stopping at 47 GB or below 8 GB
filesystem free. It never automatically deletes outputs.

## Archive and video delivery (optional)

```bash
python scripts/cluster/relay.py --remote user@transfer-host \
  --run-root "$REALPC_RUN_ROOT" --dataset "$REALPC_DATASET" \
  --archive /path/to/archive --recipient YOUR_TAILSCALE_DEVICE:
```

The relay copies completed reports/logs, verifies video hashes before Taildrop,
and verifies dataset SHA256SUMS when processing finishes. It performs no model
inference. Configure SSH authentication outside the repo. Existing outputs remain
on the processing host; delete redundant intermediates only after verification.
