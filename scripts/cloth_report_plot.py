"""Plot full-frame cloth fit/coverage/track survival; never hide failed frames."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--metrics',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();m=json.load(open(a.metrics));t=np.arange(len(m))
fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
for c,color in [('top','#1768ac'),('side','#df6a15')]:
    axes[0].plot(t,[r['cameras'][c]['surface_to_mesh_p90_mm'] for r in m],label=c,color=color)
    axes[1].plot(t,[r['cameras'][c]['surface_coverage_10mm'] for r in m],label=c,color=color)
    if m[-1].get('cotracker'):
        axes[2].plot(t,[r['cotracker'][c]['held_median_px'] if r['cotracker'][c]['held_median_px'] is not None else np.nan for r in m],label=c,color=color)
        axes[3].plot(t,[r['cotracker'][c]['held_n'] for r in m],label=c+' CoTracker',color=color)
    else:
        axes[2].plot(t,[r.get('heldout_tracks',{}).get(c,{}).get('median_px',np.nan) for r in m],label=c,color=color)
        axes[3].plot(t,[r.get('heldout_tracks',{}).get(c,{}).get('long_n',0) for r in m],label=c+' LK age≥30',color=color)
axes[0].set_ylabel('Observed → mesh p90 (mm)');axes[1].set_ylabel('Observed coverage <10 mm');axes[2].set_ylabel('Held-out track median (px)');axes[3].set_ylabel('Visible held-out tracks');axes[3].set_xlabel('Video tick (all 453; no future smoothing)')
axes[1].set_ylim(0,1.05)
for ax in axes:
    ax.axvspan(150,240,color='gray',alpha=.10);ax.axvspan(320,390,color='red',alpha=.06);ax.legend(loc='upper right');ax.grid(alpha=.2)
fig.suptitle('Causal cloth audit — shaded folds; missing tracks are missing evidence\nSAM/depth/track consistency, not hidden-material ground truth',fontsize=12)
fig.tight_layout();fig.savefig(a.out,dpi=140)
