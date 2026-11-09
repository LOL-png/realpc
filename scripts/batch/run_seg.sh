cd ~/realpc
for e in hang/episode_0000 $(for i in $(seq -w 1 49); do echo sweep/episode_00$i; done); do
  t=${e%%/*}; n=${e##*/}
  [ -f ~/data_ext/realpc_v4/seg/$t/${n}_side.npz ] && continue
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ~/miniconda3/envs/myenv/bin/python scripts/segment_objects.py --episode ~/data_ext/agentwm_probe/hf/$e --out_dir ~/data_ext/realpc_v4/seg 2>&1 | grep -E "frames, seeds|Error|Trace"
done
echo SEG_DONE
