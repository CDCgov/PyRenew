import json
from pathlib import Path
from pipeline.submit_main import main

import pytest


@pytest.mark.skip("Not written yet")
def test_local_e2e():
    pass


def test_fails_on_bad_config_file(tmp_path):
    """
    This test is for the case that a config file is handed in without either a:
    - model
    - post_production
    key at the top level of the file
    """
    # Write an empty dictionary to a json file
    bad_file: Path = tmp_path / "config.json"
    bad_file.write_text(json.dumps(dict()))

    with pytest.raises(KeyError):
        main(bad_file, Path("/made/up/path"))

def test_warns_on_missing_model(tmp_path, caplog):
    # Write an empty dictionary to a json file
    bad_file: Path = tmp_path / "config.json"
    obj = json.dumps(dict(post_production=1))
    bad_file.write_text(obj)

    toml_file: Path = tmp_path / "configuration.toml"
    toml_file.write_text("")

    main(bad_file, toml_file)
    assert "Could not find a 'model'" in caplog.text
