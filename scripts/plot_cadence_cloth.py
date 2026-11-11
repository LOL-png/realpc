"""Plot fixed-reference cloth alignment; does not run or alter extraction."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for n, color in [(10, '#777777'), (3, '#009E73'), (2, '#D55E00'), (1, '#0072B2')]:
    path = Path(a.root) / f'n{n}/audit/greycloth/episode_0000/camera_audit.json'
    rows = json.loads(path.read_text())['per_frame'][f'n{n}']
    for col, camera in enumerate(('top', 'side')):
        for row, metric in enumerate(('iou', 'depth_med_mm')):
            axes[row, col].plot([r['t'] for r in rows],
                                [r[camera][metric] for r in rows],
                                color=color, lw=1.2, alpha=.85, label=f'Every {n}')
for col, camera in enumerate(('top', 'side')):
    axes[0, col].set_title(camera.capitalize() + ' camera')
    axes[0, col].set_ylim(0, 1)
    axes[1, col].set_xlabel('Frame')
    axes[1, col].set_ylim(bottom=0)
for ax in axes.flat:
    ax.axvspan(160, 250, color='grey', alpha=.08)
    ax.axvspan(335, 452, color='grey', alpha=.08)
    ax.grid(alpha=.2)
axes[0, 0].set_ylabel('Silhouette IoU')
axes[1, 0].set_ylabel('Visible depth error (mm)')
axes[0, 0].legend(ncol=2, loc='lower left')
fig.suptitle('Full cloth episode: identical reference masks/depth at every SAM rate\n'
             'Raw per-frame values; shaded ranges are the fold audit phases. Hidden layers are unmeasured.')
fig.tight_layout()
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(out)
