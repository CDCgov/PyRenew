import json
from pathlib import Path

import pytest

from pipeline.submit_main import main


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

@pytest.fixture
def empty_config_toml() -> str:
    return """[Authentication]
subscription_id=""
resource_group=""
user_assigned_identity=""
tenant_id=""
client_id=""
principal_id=""
application_id=""
vault_url=""
vault_sp_secret_id=""
vault_sa_secret_id=""
vault_ab_secret_id=""
subnet_id=""

[Batch]
batch_account_name=""
batch_url=""
batch_service_url=""
pool-node-count=2  # Pool node count
pool_vm_size=""  # VM Type/Size
pool_id=""
task_threads=16
task_timeout_minutes=45

[Storage]
storage_account_name=""
storage_account_url=""

[Container]
container_account_name=""
container_registry_url=""
container_name=""
container_image_name=""
container_registry_username=""
container_registry_password=""
container_registry_server=""

"""

def test_warns_on_missing_model(tmp_path, caplog, empty_config_toml):
    # Write an empty dictionary to a json file
    bad_file: Path = tmp_path / "config.json"
    obj = json.dumps(dict(post_production=1))
    bad_file.write_text(obj)

    toml_file: Path = tmp_path / "configuration.toml"
    toml_file.write_text(empty_config_toml)

    main(bad_file, toml_file)
    assert "Could not find a 'model'" in caplog.text
