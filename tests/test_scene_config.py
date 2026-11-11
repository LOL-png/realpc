import json
from realpc.scene_pipeline import load_scene_config


def test_offline_asset_paths_are_relative_to_config_not_cwd(tmp_path, monkeypatch):
    directory = tmp_path / 'config'
    directory.mkdir()
    path = directory / 'scene.json'
    path.write_text(json.dumps(dict(version=1, cloth=dict(shirt_asset='../assets/shirt.npz'),
        segmentation=dict(checkpoint='../models/sam.pt'))))
    monkeypatch.chdir(tmp_path.parent)
    config = load_scene_config(path)
    assert config['cloth']['shirt_asset'] == str(tmp_path / 'assets/shirt.npz')
    assert config['segmentation']['checkpoint'] == str(tmp_path / 'models/sam.pt')


def test_absolute_asset_paths_are_preserved(tmp_path):
    path = tmp_path / 'scene.json'
    asset = str(tmp_path / 'shirt.npz')
    path.write_text(json.dumps(dict(version=1, cloth=dict(shirt_asset=asset))))
    assert load_scene_config(path)['cloth']['shirt_asset'] == asset
