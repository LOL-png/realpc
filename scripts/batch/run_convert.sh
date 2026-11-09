#!/bin/bash
# usage: run_convert.sh <worker id> <n workers>
W=$1; N=$2
cd ~/realpc
EPS="greycloth/episode_0000 hang/episode_0000 $(for i in $(seq -w 0 49); do echo sweep/episode_00$i; done)"
i=0
for e in $EPS; do
  i=$((i+1)); [ $(( (i-1) % N )) -ne $W ] && continue
  t=${e%%/*}; n=${e##*/}
  [ -f ~/data_ext/realpc_out/v1/$t/$n.npz ] && continue
  if [ $t != greycloth ]; then
    until [ -f ~/data_ext/realpc_v4/seg/$t/${n}_side.npz ]; do sleep 20; done
  fi
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/miniconda3/envs/myenv/bin/python scripts/convert_v4.py --episodes ~/data_ext/agentwm_probe/hf/$e --out_dir ~/data_ext/realpc_out/v1 2>&1 | grep -E '^\{|Error|Trace' | cut -c1-200
done
echo WORKER_DONE $W
