"""Public entry points must preserve preset selection and causal refresh policy."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from realpc.__main__ import main, preset_config, run_live
from realpc.scene_pipeline import load_scene_config
from realpc.scene_masks import SceneMaskStream


def test_presets_share_current_assets_and_only_differ_in_refresh():
    accurate = load_scene_config(preset_config('accurate'))
    fast = load_scene_config(preset_config('fast'))
    assert accurate['segmentation']['interval'] == 1
    assert fast['segmentation']['interval'] == 0
    fast['segmentation']['interval'] = 1
    assert accurate == fast
    from pathlib import Path
    assert Path(accurate['cloth']['shirt_asset']).is_file()


@pytest.mark.parametrize('name', ['accurate', 'fast'])
def test_convert_dispatches_the_selected_config(name, monkeypatch):
    import sys
    import realpc.__main__ as cli
    seen = []
    monkeypatch.setattr(cli.runpy, 'run_path', lambda *a, **kw: seen.append(list(sys.argv)))
    main([name, 'convert', '--episodes', 'data/hang/episode_0000', '--out', 'out'])
    assert seen[0][1:3] == ['--config', str(preset_config(name))]
    assert seen[0][-2:] == ['--out', 'out']
    with pytest.raises(SystemExit):
        main([name, 'convert', '--config=old.json'])


@pytest.mark.parametrize('name,expected', [('accurate', True), ('fast', False)])
def test_refresh_uses_current_failure_or_preset_cadence(name, expected, monkeypatch):
    import realpc.scene_masks as masks
    labels = np.zeros((2, 2), np.uint8)
    state = SimpleNamespace(lost=0, held=None)
    tracker = SimpleNamespace(state={'part': state}, cfg={})
    converter = SimpleNamespace(trk=tracker, objs=[SimpleNamespace(name='part')])
    stream = SceneMaskStream(converter, load_scene_config(preset_config(name))['segmentation'])
    calls = []
    def refresh(t, *args):
        calls.append(t)
        stream.last_sam = t
        return {'top': labels}
    monkeypatch.setattr(stream, '_refresh', refresh)
    # Initialization must call SAM even in fast mode.
    stream.initialize(0, {}, {}, {})
    assert calls == [0]
    converter.trk = tracker  # Converter reinitializes its depth fit after the SAM seed.
    view = SimpleNamespace(name='top', labels=labels, R=SimpleNamespace(W=2, H=2))
    monkeypatch.setattr(masks, 'associate_depth', lambda *args: None)
    packet = {'top_depth': np.ones((2, 2))}
    stream.step(1, {}, {}, packet, [view], {}, [], [])
    assert stream.last_was_sam is expected
    if name == 'fast':
        for t in range(2, 8):
            stream.step(t, {}, {}, packet, [view], {}, [], [])
        assert calls == [0]  # No periodic SAM between events.
        stream.pending = ['part']
        stream.step(8, {}, {}, packet, [view], {}, [], [])
        assert calls == [0, 8]
        assert stream.last_was_sam


def test_environment_checkpoint_override_is_shared(tmp_path, monkeypatch):
    monkeypatch.setenv('REALPC_MODEL_CACHE', str(tmp_path/'models'))
    for name in ('accurate', 'fast'):
        assert load_scene_config(preset_config(name))['segmentation']['checkpoint'] == str(tmp_path/'models/sam2/sam2.1_hiera_small.pt')
    monkeypatch.setenv('REALPC_SAM_CHECKPOINT', str(tmp_path/'custom.pt'))
    assert load_scene_config(preset_config('fast'))['segmentation']['checkpoint'] == str(tmp_path/'custom.pt')


def test_live_passes_only_current_packets_and_exports_quality(tmp_path, monkeypatch):
    import realpc.live as live
    import realpc.__main__ as cli
    setup = tmp_path/'setup';setup.mkdir()
    for name in ('meta', 'calib', 'table_region'):
        (setup/(name+'.json')).write_text('{}')
    calls = []
    class Pipeline:
        def step(self, tick, frames, q):
            calls.append((tick, frames, q))
            return dict(sam_called=tick == 0, max_observed_tick=tick, cloth_valid=False, cloth_approximate=True)
        def points(self, row):return np.zeros((3, 3), np.float32)
    def build(setup, config, **kwargs):
        assert config == preset_config('fast')
        assert kwargs == {'cloth': True}
        return Pipeline()
    monkeypatch.setattr(live, 'build_live_pipeline', build)
    packets = [dict(tick=t, frames={'current':t}, q={'current':t}) for t in range(2)]
    monkeypatch.setattr(cli.importlib, 'import_module', lambda name: SimpleNamespace(packets=lambda: iter(packets)))
    run_live(SimpleNamespace(source='camera:packets', setup=setup, task='greycloth', out=tmp_path/'out'), preset_config('fast'))
    assert calls == [(p['tick'], p['frames'], p['q']) for p in packets]
    with np.load(tmp_path/'out/000001.npz') as saved:
        assert bool(saved['cloth_approximate']) and not bool(saved['cloth_valid'])
        assert saved['max_observed_tick'] == 1
