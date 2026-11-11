"""A live scene requires only reusable setup metadata, never a recording."""
import json
import numpy as np
import pytest
from realpc.live import SceneSetup


def test_setup_without_recorded_data(tmp_path):
    K = [[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]]
    (tmp_path / 'meta.json').write_text(json.dumps({'intrinsics': {'top': K}}))
    (tmp_path / 'calib.json').write_text(json.dumps(
        dict(table_n=[0, 0, 1], table_d=-.01, T_lb_rb=np.eye(4).tolist())))
    (tmp_path / 'table_region.json').write_text('{}')
    setup = SceneSetup(tmp_path, 'sweep')
    assert np.array_equal(setup.intrinsics('top'), K)
    assert setup.table_plane()[1] == -.01
    assert np.array_equal(setup.T_world_base()['right'], np.eye(4))
    assert not any(hasattr(setup, k) for k in ('low', 'video', 'frame', 'cam_pose', 'files'))
    assert len(list(tmp_path.iterdir())) == 3


def test_missing_workspace_setup_is_reported(tmp_path):
    (tmp_path / 'meta.json').write_text('{}')
    (tmp_path / 'calib.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='table_region.json'):
        SceneSetup(tmp_path, 'sweep')
