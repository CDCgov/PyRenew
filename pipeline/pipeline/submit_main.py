import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from cfa_azure.clients import AzureClient

logger = logging.getLogger(__name__)


def main(input_config_path: Path, azure_config_path: Path):
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

    azure_config_path : Path
        The path to the Azure configuration file.

        This file should contain all necessary fields for connecting with Azure
        Batch, creating pools, submitting jobs, and accessing storage.

    Returns
    -------
    None
    """
    # === Read config file ====================================================
    config = json.loads(input_config_path.read_text())
    logger.debug(f"Read primary config from {input_config_path}")

    # === Prep Azure client ===================================================
    client = AzureClient(config_path=str(azure_config_path))
    logger.debug("Azure client config initialized")
    # TODO: enter actual pool name
    pool_name = "POOL_NAME"
    client.use_pool(pool_name=pool_name)
    logger.debug(f"Using pool {pool_name}")
    job_id = "multisignal-epi-inference-prod"
    client.add_job(job_id=job_id)
    logger.debug("Created job")
    logger.info("Azure client configured")

    # === Prep individual configs and docker commands =========================
    model_configs: list[dict[str, Any]] = config["model"]
    model_docker_cmds: list[list[str]] = [
        create_docker_cmd(mcfg) for mcfg in model_configs
    ]

    model_config_file_names: list[Path] = [
        create_mdl_cfg_filename(mcfg) for mcfg in model_configs
    ]

    # === Kick off model tasks ================================================
    logger.info("Submiting Modeling tasks")
    model_task_ids: list = []
    for mcfg, dckr_cmd, cfg_fname in zip(
        model_configs, model_docker_cmds, model_config_file_names
    ):
        logger.info(json.dumps(mcfg))
        # Create the config file to upload to blob storage
        cfg_fname.write_text(json.dumps(mcfg))
        logger.debug(f"Wrote model config file {cfg_fname}")

        # Submit the task
        tid = client.add_task(
            job_id=job_id, docker_cmd=dckr_cmd, input_files=[str(cfg_fname)]
        )
        logger.debug(f"Submitted task {tid}")
        model_task_ids.append(tid)

    # === Prep individual configs and docker commands =========================
    pp_configs: list[dict[str, Any]] = config["post_production"]
    pp_docker_cmds: list[list[str]] = [
        create_docker_cmd(ppcfg) for ppcfg in pp_configs
    ]

    pp_config_file_names: list[Path] = [
        create_pp_cfg_filename(ppcfg) for ppcfg in pp_configs
    ]

    # === Kick off post processing ============================================
    logger.info("Submitting Post Processing tasks")
    for ppcfg, dckr_cmd, cfg_fname in zip(
        pp_configs, pp_docker_cmds, pp_config_file_names
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
            depends_on=model_task_ids,
        )
        logger.debug(f"Submitted task {tid}")

    logger.info("All tasks submitted. Waiting for completion")

    # === Make sure all jobs are cleaned up ===================================
    client.monitor_job(job_id)
    client.delete_job(job_id)


if __name__ == "__main__":
    import argparse

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

    parser.add_argument(
        "azure_config", help="Path to Azure config file", type=Path
    )

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

    main(args.primary_config, args.azure_config)
