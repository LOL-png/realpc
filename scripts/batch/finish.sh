#!/bin/bash
until grep -q SEG_DONE ~/data_ext/realpc_v4/seg.log; do sleep 30; done
bash ~/data_ext/realpc_v4/run_seg.sh >> ~/data_ext/realpc_v4/seg_redo.log 2>&1
until [ $(cat ~/data_ext/realpc_v4/convert_w0.log ~/data_ext/realpc_v4/convert_w1.log ~/data_ext/realpc_v4/convert_w2.log | grep -c WORKER_DONE) -ge 3 ]; do sleep 30; done
# outputs older than their (re-made) segmentation are stale
for i in 03 04 05 06 10 11 12 15 17 18 23 25 28 36; do
  o=~/data_ext/realpc_out/v1/sweep/episode_00$i.npz; s=~/data_ext/realpc_v4/seg/sweep/episode_00${i}_side.npz
  if [ -f $o ] && [ $o -ot $s ]; then rm -f $o ${o%.npz}_qa.json; echo "stale $o removed"; fi
done
for w in 0 1 2; do bash ~/data_ext/realpc_v4/run_convert.sh $w 3 >> ~/data_ext/realpc_v4/convert_redo_w$w.log 2>&1 & done; wait
bash ~/data_ext/realpc_v4/post_batch.sh
