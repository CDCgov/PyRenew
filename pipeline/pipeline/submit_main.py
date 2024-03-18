import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from cfa_azure.clients import AzureClient

logger = logging.getLogger(__name__)


def main(input_config_path: Path, client: AzureClient):
    """
    Kick off the submission of tasks based on the contents of the file at
    `input_config_path`.

    Parameters
    ----------
    input_config_path : Path
        The path to the input configuration file.

        This file should have two top-level fields: 'model' and
        'post_production'. Both fields are optional.

        If 'model' is present, it should contain a list of model configurations
        to run.

        If 'post_production' is present, it should contain a list of
        specifications for each post-production run.

    client: AzureClient
        An AzureClient as created from a configuration file.

    Returns
    -------
    None
    """
    # === Read config file ====================================================
    config: dict = json.loads(input_config_path.read_text())
    logger.debug(f"Read primary config from {input_config_path}")

    # Fail early if don't have necessary keys
    necessary_keys = set(["model", "post_production"])
    if necessary_keys.difference(config.keys()) == necessary_keys:
        msg = (
            "Require at least one of 'model' or 'post_production' at top"
            + " level of config file"
        )
        logger.error(msg)
        raise KeyError(msg)

    # Warn if don't find one or another of the top level keys
    if "model" not in config.keys():
        logger.warning(
            f"Could not find a 'model' key at top level of {input_config_path}"
        )

    if "post_production" not in config.keys():
        logger.warning(
            f"Could not find a 'post_production' key at the top level of {input_config_path}"
        )

    # === Prep Azure client ===================================================
    # TODO: enter actual pool name
    pool_name = "POOL_NAME"
    client.use_pool(pool_name=pool_name)
    logger.debug(f"Using pool {pool_name}")
    job_id = "multisignal-epi-inference-prod"
    client.add_job(job_id=job_id)
    logger.debug("Created job")
    logger.info("Azure client configured")

    if "model" in config.keys():
        model_task_ids = submit_model_tasks(client, job_id, config["model"])
    else:
        model_task_ids = None

    if "post_production" in config.keys():
        submit_post_production_tasks(
            client,
            job_id,
            post_prod_config=config["post_production"],
            depends_on=model_task_ids,
        )

    logger.info("All tasks submitted. Waiting for completion")

    # === Make sure all jobs are cleaned up ===================================
    client.monitor_job(job_id)
    client.delete_job(job_id)


def submit_model_tasks(
    client: AzureClient, job_id: str, model_config: list[dict]
) -> list[str]:
    """
    Submit all the model tasks specified in `model_config` to the `job_id`.

    Parameters
    ----------
    client :
        The client object for interacting with the job submission system.
    job_id : str
        The ID of the job to which model tasks will be submitted.
    model_config : list[dict]
        A list of dictionaries representing model configurations to be
        submitted as tasks.

    Returns
    -------
    list[str]
        A list of task IDs corresponding to the submitted model tasks.
    """
    # === Prep individual configs and docker commands =========================
    model_docker_cmds: list[list[str]] = [
        create_docker_cmd(mcfg) for mcfg in model_config
    ]

    model_config_file_names: list[Path] = [
        create_mdl_cfg_filename(mcfg) for mcfg in model_config
    ]

    # === Kick off model tasks ================================================
    logger.info("Submiting Modeling tasks")
    model_task_ids: list[str] | None = []
    for mcfg, dckr_cmd, cfg_fname in zip(
        model_config, model_docker_cmds, model_config_file_names
    ):
        logger.info(json.dumps(mcfg))
        # Create the config file to upload to blob storage
        cfg_fname.write_text(json.dumps(mcfg))
        logger.debug(f"Wrote model config file {cfg_fname}")

        # Submit the task
        tid = client.add_task(
            job_id=job_id,
            docker_cmd=dckr_cmd,
            input_files=[str(cfg_fname)],
        )
        logger.debug(f"Submitted task {tid}")
        model_task_ids.append(tid)

    return model_task_ids


def submit_post_production_tasks(
    client: AzureClient,
    job_id: str,
    post_prod_config: list[dict],
    depends_on: list[str] | None = None,
):
    """
    Submit all the post-production tasks specified in `post_prod_config` to the
    `job_id`.

    Parameters
    ----------
    client :
        The client object for interacting with the job submission system.
    job_id : str
        The ID of the job to which post-production tasks will be submitted.
    post_prod_config : list[dict]
        A list of dictionaries representing specifications for post-production
        tasks.
    depends_on : list[str] or None, optional
        A list of task IDs that these tasks must wait on before execution.
        Default is None.

    Returns
    -------
    None
    """
    # === Prep individual configs and docker commands =========================
    pp_docker_cmds: list[list[str]] = [
        create_docker_cmd(ppcfg) for ppcfg in post_prod_config
    ]

    pp_config_file_names: list[Path] = [
        create_pp_cfg_filename(ppcfg) for ppcfg in post_prod_config
    ]

    # === Kick off post processing ============================================
    logger.info("Submitting Post Processing tasks")
    for ppcfg, dckr_cmd, cfg_fname in zip(
        post_prod_config, pp_docker_cmds, pp_config_file_names
    ):
        logger.info(json.dumps(ppcfg))
        # Create the config file to upload to blog storage
        cfg_fname.write_text(json.dumps(ppcfg))
        logger.debug(f"Wrote post processing config file {cfg_fname}")

        # Submit the task
        tid = client.add_task(
            job_id=job_id,
            docker_cmd=dckr_cmd,
            input_files=[str(cfg_fname)],
            depends_on=depends_on,
        )
        logger.debug(f"Submitted task {tid}")


def create_docker_cmd(config: dict[str, Any]) -> list[str]:
    """
    For a given configuration, build a command to pass to docker
    """
    # TODO: fill this out
    return list()


def create_mdl_cfg_filename(config: dict[str, Any]) -> Path:
    """
    For a given configuration, create the Path to the file name for that model
    run.
    """
    # TODO: fill this out
    return Path("somemodel.json")


def create_pp_cfg_filename(config: dict[str, Any]) -> Path:
    """
    For a given configuration, create the Path to the file name for that post
    production run.
    """
    # TODO: fill this out
    return Path("somepostprod.json")


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timedelta

    start_time = datetime.now()

    # Get log level from environemnt. Set to debug if not found.
    LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
    logging.basicConfig(level=LOGLEVEL)

    parser = argparse.ArgumentParser(
        description="For deploying runs to Azure Batch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "primary_config", help="Path to primary config file", type=Path
    )

    parser.add_argument("azure_config", help="Path to Azure config file")

    parser.add_argument(
        "--log_file",
        help="What file to put the logs in",
        default="submit_main.log",
    )

    args = parser.parse_args()

    # Set up logging options
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.log_file),
        ],
        format="%(levelname)s:%(asctime)s:%(filename)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
    logger.info(f"Log level set to {LOGLEVEL}")
    logger.info(f"Using primary config file {args.primary_config}")
    logger.info(f"Using Azure config file {args.azure_config}")

    # Create the client outside main because it is easier to test main() when
    # it is passed a client object, rather than a file
    client = AzureClient(args.azure_config)
    logger.info("Created AzureClient")

    main(args.primary_config, client)

    # Calculate run time, rounded to nearest second
    run_time = datetime.now() - start_time
    run_time = timedelta(seconds=round(run_time.total_seconds()))
    logger.info(f"Total runtime was {run_time}")
