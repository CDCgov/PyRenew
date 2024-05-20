import json
from pathlib import Path
from unittest.mock import call, patch

import pytest

from pipeline.submit_main import main


@pytest.fixture
def primary_config_small() -> str:
    return json.dumps(
        dict(
            model=dict(
                model1=dict(state="AL", disease="Covid-19"),
                model2=dict(state="AL", disease="Influenza"),
                model3=dict(state="AK", disease="Covid-19"),
                model4=dict(state="AK", disease="Influenza"),
            ),
            post_production=dict(
                post_prod1=dict(states="*", disease="Covid-19"),
                post_prod2=dict(states="*", disease="Influenza"),
            ),
        )
    )


@patch("cfa_azure.clients.AzureClient")
def test_local_e2e(mocked_azure_client, primary_config_small, tmp_path):
    # Create the input primary config file
    primary_config: Path = tmp_path / "config.json"
    primary_config.write_text(primary_config_small)

    # Have function calls to `client.add_task()` return random strings
    with patch.object(mocked_azure_client, "add_task") as add_task:
        main(primary_config, mocked_azure_client)

        # Assert that for each model, and each post_prod in
        # primary_config_small, there is a call to `client.add_task()`
        # Since we aren't producing actual files yet, the inputs are basic
        expected_calls = [
            call(
                job_id="multisignal-epi-inference-prod",
                docker_cmd=list(),
                input_files=[str(Path("somemodel.json"))],
            ),
            call(
                job_id="multisignal-epi-inference-prod",
                docker_cmd=list(),
                input_files=[str(Path("somemodel.json"))],
            ),
            call(
                job_id="multisignal-epi-inference-prod",
                docker_cmd=list(),
                input_files=[str(Path("somemodel.json"))],
            ),
            call(
                job_id="multisignal-epi-inference-prod",
                docker_cmd=list(),
                input_files=[str(Path("somemodel.json"))],
            ),
            call(
                job_id="multisignal-epi-inference-prod",
                docker_cmd=list(),
                input_files=[str(Path("somepostprod.json"))],
                # This should be a list of task_ids, but haven't yet figured out how to
                # mock that
                depends_on=[],
            ),
            call(
                job_id="multisignal-epi-inference-prod",
                docker_cmd=list(),
                input_files=[str(Path("somepostprod.json"))],
                # This should be a list of task_ids, but haven't yet figured out how to
                # mock that
                depends_on=[],
            ),
        ]
        # Not using `add_task.assert_has_calls()` bc there are a number of calls to
        # `__str__()`, `__len__()`, and `__iter__()` made by the mocking that we don't
        # care about.
        # Instead, test that each expected call is in the list of actual calls made
        for ec in expected_calls:
            assert ec in add_task.mock_calls


@patch("cfa_azure.clients.AzureClient")
def test_fails_on_bad_config_file(mocked_azure_client, tmp_path):
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
        main(bad_file, mocked_azure_client)


@patch("cfa_azure.clients.AzureClient")
def test_warns_on_missing_model(mocked_azure_client, tmp_path, caplog):
    # Write an empty dictionary to a json file
    bad_file: Path = tmp_path / "config.json"
    obj = json.dumps(dict(post_production=1))
    bad_file.write_text(obj)

    # The point of this isn't to check for the TypeError, but rather to find
    # the warning in the `caplog.text`. We have to catch the TypeError because
    # `post_production=1` is not a valid input for
    # `submit_post_production_tasks()`
    with pytest.raises(TypeError):
        main(bad_file, mocked_azure_client)
    assert "Could not find a 'model'" in caplog.text
    assert "Could not find a 'post_production'" not in caplog.text


@patch("cfa_azure.clients.AzureClient")
def test_warns_on_missing_post_prod(mocked_azure_client, tmp_path, caplog):
    # Write an empty dictionary to a json file
    bad_file: Path = tmp_path / "config.json"
    obj = json.dumps(dict(model=1))
    bad_file.write_text(obj)

    # The point of this isn't to check for the TypeError, but rather to find
    # the warning in the `caplog.text`. We have to catch the TypeError because
    # `post_production=1` is not a valid input for
    # `submit_post_production_tasks()`
    with pytest.raises(TypeError):
        main(bad_file, mocked_azure_client)
    assert "Could not find a 'post_production'" in caplog.text
    assert "Could not find a 'model'" not in caplog.text
