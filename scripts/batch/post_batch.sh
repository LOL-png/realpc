#!/bin/bash
set -u
cd ~/realpc
PY=~/miniconda3/envs/myenv/bin/python
O=~/data_ext/realpc_out/v1
R=~/point_tracker/reports/REALDATA_V4_1109
# 1. self-consistency QA (our SAM2 masks) for every library episode, Astra-mask check for the ep0s
qa_lane() {
  i=0
  for f in $O/sweep/episode_*.npz $O/hang/episode_*.npz; do
    i=$((i+1)); [ $(( (i-1) % 3 )) -ne $1 ] && continue
    e=$(basename $f .npz); t=$(basename $(dirname $f))
    [ -f $O/$t/${e}_selfqa.json ] && continue
    $PY scripts/eval_alignment.py --converted $f --episode ~/data_ext/agentwm_probe/hf/$t/$e --seg_dir ~/data_ext/realpc_v4/seg --every 4 --out $O/$t/${e}_selfqa.json > /dev/null 2>&1
  done
}
qa_lane 0 & qa_lane 1 & qa_lane 2 & wait
for t in sweep hang; do $PY scripts/eval_alignment.py --converted $O/$t/episode_0000.npz --episode ~/data_ext/agentwm_probe/hf/$t/episode_0000 --masks ~/data_ext/masks/$t/episode_0000 --out $O/$t/episode_0000_align.json 2>&1 | grep -v samples; done
# 2. SE(3) round trip (segment-seeded patches) on the 3 ep0 + 3 more sweep episodes
CK=/tmp/claude-1000/-home-howardji/f331bdc6-c16c-448d-961d-d4d516d8e85c/scratchpad/realdata1108/ckpt
PLWM_SE3_SRC=~/point_tracker/reports/REALDATA_SE3_1108/code/plwm_tag $PY scripts/roundtrip_v4.py --converted $O/sweep/episode_0000.npz $O/hang/episode_0000.npz $O/greycloth/episode_0000.npz $O/sweep/episode_0010.npz $O/sweep/episode_0025.npz $O/sweep/episode_0040.npz --out $R/roundtrip_v4.json --fb_ckpt $CK/fb_ae_wmstats.pt --rc_ckpt $CK/rc16k_ae_wmstats.pt 2>&1 | grep -v Warn
# 3. manifest
$PY scripts/make_manifest.py --out_dir $O --align $O/sweep/episode_0000_align.json $O/hang/episode_0000_align.json
echo POST_DONE
