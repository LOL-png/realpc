#!/bin/bash
#SBATCH --job-name=realpc-all52
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
set -eo pipefail
RUN_ROOT=${REALPC_RUN_ROOT:?Set an absolute staging directory}
DATASET=${REALPC_DATASET:?Set an absolute dataset output directory}
test -f "$RUN_ROOT/READY"
mkdir -p "$RUN_ROOT/env" "$RUN_ROOT/tmp" "$RUN_ROOT/cache"
exec 9>"$RUN_ROOT/environment.lock"
flock 9
if [ ! -f "$RUN_ROOT/env/.realpc-unpacked" ]; then
  (cd "$RUN_ROOT/packages" && sha256sum -c SHA256SUMS)
  tar -xzf "$RUN_ROOT/packages/myenv.tar.gz" -C "$RUN_ROOT/env"
  "$RUN_ROOT/env/bin/python" "$RUN_ROOT/env/bin/conda-unpack"
  touch "$RUN_ROOT/env/.realpc-unpacked"
fi
flock -u 9
export PATH="$RUN_ROOT/runtime/bin:$RUN_ROOT/env/bin:$PATH"
export LD_LIBRARY_PATH="$RUN_ROOT/runtime/lib:$RUN_ROOT/env/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$RUN_ROOT/runtime/lib/libstdc++.so.6"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR="$RUN_ROOT/tmp" TORCH_EXTENSIONS_DIR="$RUN_ROOT/cache/torch"
cd "$RUN_ROOT/code"
python scripts/cluster/run_all.py --run-root "$RUN_ROOT" --dataset "$DATASET" --pipeline "${REALPC_PIPELINE:-accurate}"
