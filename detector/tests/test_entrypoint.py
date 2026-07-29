from aidetector import _configuration_revision


def test_configuration_revision_tracks_atomic_config_updates(tmp_path):
    config_path = tmp_path / "config.json"
    assert _configuration_revision(config_path) is None

    config_path.write_text('{"detectors":[]}\n')
    first = _configuration_revision(config_path)
    assert first is not None

    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"detectors":[{}]}\n')
    replacement.replace(config_path)
    assert _configuration_revision(config_path) != first
