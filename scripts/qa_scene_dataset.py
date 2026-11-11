#!/usr/bin/env python
"""Sampled per-episode alignment QA, optional full example videos, and tables.

Shared masks measure tracker/segmentation consistency. Independent Q28 masks,
where available, are scored separately and never enter extraction.
"""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(args, log):
    with open(log, 'w') as stream:
        subprocess.run([sys.executable, *args], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def sheet(paths, output, columns=3, size=(480, 360)):
    tiles = [cv2.resize(cv2.imread(str(p)), size) for p in paths if p.exists()]
    if not tiles:
        return
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(output), np.vstack([np.hstack(tiles[i:i + columns])
                                      for i in range(0, len(tiles), columns)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--every', type=int, default=100)
    p.add_argument('--episodes', nargs='*', help='Optional task/episode names')
    p.add_argument('--example-videos', action='store_true')
    p.add_argument('--cloth-reference-inputs', help='Current RGB-D/FK/mask packets, for QA only')
    a = p.parse_args()
    dataset, output = Path(a.dataset), Path(a.out)
    output.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(dataset.glob('*/episode_*.npz'))
             if f.stem.split('_')[-1].isdigit()]
    totals, episodes, task_images, cloth_physics, depth_totals = {}, {}, {}, {}, {}
    pose_coverage, episode_coverage = {}, {}
    for path in files:
        task, episode = path.parent.name, path.stem
        key = task + '/' + episode
        if a.episodes and key not in a.episodes:
            continue
        dest = output / task / episode
        dest.mkdir(parents=True, exist_ok=True)
        data = np.load(path)
        count = len(data['ticks'])
        picks = np.unique(np.linspace(0, count - 1, 5).astype(int)).tolist()
        video = a.example_videos and episode == 'episode_0000'
        if task != 'greycloth':
            episode_coverage[key] = {}
            for index, object_name in enumerate(data['obj_names']):
                name = str(object_name)
                defined = int(data['obj_valid'][index].sum())
                episode_coverage[key][name] = dict(frames=count, pose_defined_frames=defined)
                entry = pose_coverage.setdefault(name, dict(episodes=0, frames=0, pose_defined_frames=0,
                                                           episodes_with_missing_pose=[]))
                entry['episodes'] += 1
                entry['frames'] += count
                entry['pose_defined_frames'] += defined
                if defined < count:
                    entry['episodes_with_missing_pose'].append(key)
            args = ['scripts/audit_v3.py', '--converted', str(path), '--scene-seg-dir', str(dataset),
                    '--reference-name', 'stream', '--assets', str(ROOT / 'assets/objects_v3'),
                    '--depth-check', '--every', str(1 if video else a.every), '--out', str(dest),
                    '--visual', *map(str, picks)]
            if video:
                args += ['--video']
            run(args, dest / 'audit.log')
            result = json.loads((dest / 'metrics.json').read_text())
            episodes[key] = result['summary']
            for name, rows in result.get('depth_per_frame', {}).items():
                entry = depth_totals.setdefault(name, dict(rows=[], episodes=[]))
                entry['rows'].extend(rows)
                entry['episodes'].append(key)
            for mode, objects in result['per_frame'].items():
                for name, rows in objects.items():
                    entry = totals.setdefault(mode, {}).setdefault(name, dict(scores=[], episodes=[], episode_pass=[]))
                    entry['scores'] += [x[1] for x in rows]
                    entry['episodes'].append(key)
                    entry['episode_pass'].append(float(np.mean([x[1] >= .5 for x in rows])))
            images = [dest / f'{task}_{episode}_{t:04d}.jpg' for t in picks]
            sheet(images, dest / 'contact_sheet.jpg')
            task_images.setdefault(task, []).append(images[len(images) // 2])
            if video:
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(dest / 'top_overlay.mp4'),
                    '-i', str(dest / 'side_overlay.mp4'), '-filter_complex', '[0:v][1:v]vstack=inputs=2[v]',
                    '-map', '[v]', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(dest / 'both_cameras.mp4')], check=True)
        else:
            fit = dest / 'fit'
            fit.mkdir(exist_ok=True)
            links = {'mesh.npz':path.with_name(episode + '_cloth_mesh.npz'),
                     'metrics.json':path.with_name(episode + '_cloth_metrics.json'),
                     'template_info.json':path.with_name(episode + '_cloth_template.json')}
            for name, source in links.items():
                link = fit / name
                if link.is_symlink():
                    link.unlink()
                shutil.copy2(source,link)
            run(['scripts/cloth_quality_report.py', str(fit), '--all'], dest / 'physics.log')
            physics = json.loads((fit / 'quality_report.json').read_text())
            cloth_physics[key] = dict(
                endpoint_crossing_frames=sum(r['intersects'] for r in physics['physical_audit']),
                audited_frames=len(physics['physical_audit']),
                max_extension=max(r['max_extension'] for r in physics['physical_audit']),
                max_compression=max(r['max_compression'] for r in physics['physical_audit']),
                valid_frames=int(data['cloth_valid'].sum()),
                phases=physics['phases'])
            if a.cloth_reference_inputs:
                run(['scripts/cloth_multiview_audit.py', '--fit', 'unified=' + str(fit / 'mesh.npz'),
                    '--inputs', a.cloth_reference_inputs, '--out', str(dest / 'camera_audit.json')], dest / 'audit.log')
                camera_audit = json.loads((dest / 'camera_audit.json').read_text())
                episodes[key] = camera_audit['summary']['unified']
                for cam in ('top', 'side'):
                    totals.setdefault('cloth_reference', {})['cloth/' + cam] = dict(
                        scores=[r[cam]['iou'] for r in camera_audit['per_frame']['unified']], episodes=[key],
                        episode_pass=[float(np.mean([r[cam]['iou'] >= .5 for r in camera_audit['per_frame']['unified']]))])
                if video:
                    run(['scripts/cloth_video_report.py', '--fit', str(fit), '--inputs', a.cloth_reference_inputs,
                         '--out', str(dest)], dest / 'video.log')
                    import trimesh
                    mesh = np.load(fit / 'mesh.npz')
                    # flat_rest is enabled: V0 is the fitted clean garment metric.
                    trimesh.Trimesh(mesh['V0'], mesh['F'], process=False).export(fit / 'clean_template.ply')
                    run(['scripts/shirt_template_report.py', '--fit', str(fit), '--out', str(dest)], dest / 'template.log')
                    shutil.copy2(dest / 'frame_0000.jpg', dest / 'first_frame_fit.jpg')
                    task_images.setdefault(task, []).append(dest / 'frame_0350.jpg')
                else:
                    from cloth_video_report import render
                    mesh = np.load(fit / 'mesh.npz')
                    layers = json.loads((fit / 'template_info.json').read_text()).get('layers', 1)
                    images = []
                    for tick in picks:
                        with np.load(Path(a.cloth_reference_inputs) / 'frames' / f'{tick:04d}.npz') as packet:
                            frame = {k: packet[k] for k in packet.files}
                        rows = [np.hstack([frame[c + '_rgb'], render(frame, c, mesh['X'][tick], mesh['F'], layers=layers)])
                                for c in ('top', 'side')]
                        image_path = dest / f'frame_{tick:04d}.jpg'
                        cv2.imwrite(str(image_path), np.vstack(rows))
                        images.append(image_path)
                    sheet(images, dest / 'contact_sheet.jpg')
                    task_images.setdefault(task, []).append(images[len(images) // 2])
        print('QA complete', key, flush=True)
    aggregate = {}
    for mode, objects in totals.items():
        aggregate[mode] = {}
        for name, entry in objects.items():
            x = np.asarray(entry['scores'])
            aggregate[mode][name] = dict(episodes=len(set(entry['episodes'])), sampled_frames=len(x),
                median_iou=float(np.median(x)), pass_iou_0_5=float(np.mean(x >= .5)),
                mean_episode_pass=float(np.mean(entry['episode_pass'])))
    for task, images in task_images.items():
        sheet(images, output / (task + '_contact_sheet.jpg'), columns=4)
    depth_summary = {}
    for name, entry in depth_totals.items():
        good = [r for r in entry['rows'] if r['pass_depth'] is not None]
        depth_summary[name] = dict(episodes=len(set(entry['episodes'])),
            frames_with_valid_depth=len(entry['rows']), comparable_frames=len(good),
            pass_fraction=float(np.mean([r['pass_depth'] for r in good])) if good else None,
            median_mm=float(np.median([r['median_mm'] for r in good])) if good else None)
    report = dict(aggregate=aggregate, episodes=episodes, cloth_physics=cloth_physics, every=a.every,
        depth_agreement=depth_summary,
        pose_availability=pose_coverage, episode_pose_availability=episode_coverage,
        sampling='Rigid: requested stride plus five spread-out frames; example videos audit every frame. Cloth: every frame.',
        note='Stream consistency is not independent accuracy. Independent labels exist only for selected episodes. Cloth camera audit uses every-frame causal SAM reference masks; folds remain approximate.')
    (output / 'alignment_summary.json').write_text(json.dumps(report, indent=2))
    lines = ['# Per-object alignment', '', 'Sampled visible silhouette IoU>=0.5; independent and stream scores are separate.', '']
    lines += ['## Pose availability', '',
        'A defined pose is not an alignment pass. Missing poses are counted here even when no rendered silhouette exists.', '',
        '| Object | Episodes | Defined / total frames | Episodes with missing poses |', '|---|---:|---:|---:|']
    for name, values in sorted(pose_coverage.items()):
        lines.append(f"| {name} | {values['episodes']} | {values['pose_defined_frames']}/{values['frames']} | {len(values['episodes_with_missing_pose'])} |")
    lines.append('')
    for mode, rows in aggregate.items():
        lines += ['## ' + mode, '', '| Object/view | Episodes | Frames | Frame pass | Mean episode pass | Median IoU |', '|---|---:|---:|---:|---:|---:|']
        for name, row in sorted(rows.items()):
            lines.append(f"| {name} | {row['episodes']} | {row['sampled_frames']} | {100*row['pass_iou_0_5']:.1f}% | {100*row['mean_episode_pass']:.1f}% | {row['median_iou']:.3f} |")
        lines.append('')
    if depth_summary:
        lines += ['## Raw-depth agreement', '',
            'Current depth, excluding closer occluders: at least60 comparable pixels; pass median error<=10mm and free-space fraction<=20% (15mm gate). This checks surface placement, not semantic identity or hidden pose.', '',
            '| Object/view | Episodes | Comparable frames | Pass | Median error (mm) |', '|---|---:|---:|---:|---:|']
        for name, values in sorted(depth_summary.items()):
            if values['pass_fraction'] is not None:
                lines.append(f"| {name} | {values['episodes']} | {values['comparable_frames']} | {100*values['pass_fraction']:.1f}% | {values['median_mm']:.2f} |")
        lines.append('')
    for key, physics in cloth_physics.items():
        lines += ['## Cloth limitations: ' + key, '',
            f"Endpoint crossings: {physics['endpoint_crossing_frames']}/{physics['audited_frames']}; worst local extension/compression: {physics['max_extension']*100:.1f}%/{physics['max_compression']*100:.1f}%; valid frames: {physics['valid_frames']}/{physics['audited_frames']}.",
            'No endpoint crossings does not imply correct material correspondence, bounded stretch, or finite-thickness safety.', '',
            '| Phase | Depth median top/side (mm) | Mask IoU top/side | Soft grasp patch max (mm) | Contact centre max (mm) |',
            '|---|---:|---:|---:|---:|']
        for phase, values in episodes.get(key, {}).items():
            residual = physics['phases'][phase]['max_grasp_residual_mm']
            hard = physics['phases'][phase]['max_hard_grasp_residual_mm']
            hard_text = f'{hard:.2f}' if hard is not None else 'unavailable'
            lines.append(f"| {phase} | {values['top']['depth_med_mm']:.2f}/{values['side']['depth_med_mm']:.2f} | "
                         f"{values['top']['iou']:.3f}/{values['side']['iou']:.3f} | {residual:.1f} | {hard_text} |")
        lines += ['', 'All cloth geometry is approximate. Validity is a current-frame support/physical gate; the alignment audit includes invalid frames too.', '']
    (output / 'ALIGNMENT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
