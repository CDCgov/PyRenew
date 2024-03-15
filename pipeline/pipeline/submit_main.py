import logging
import os
import sys

logger = logging.getLogger(__name__)


def main(input_config_path: str):
    """
    Kick off the submission of tasks based on the contents of the file at
    `input_config_path`. The file should have two top level fields: `model`,
    and `post_production`. Both do not have to be present if you, for example,
    only wish to submit model runs, or only wish to do post production runs.

    Within `model` should be what evaluates to a list of model configurations
    to run.

    Within `post_production` should be what evaluates to a list of
    specifications for each post production run.
    """
    pass


if __name__ == "__main__":
    import argparse

    # Get log level from environemnt. Set to debug if not found.
    LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG").upper()
    logging.basicConfig(level=LOGLEVEL)

    parser = argparse.ArgumentParser(
        description="For deploying runs to Azure Batch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", "--input", help="Path to primary config file")
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

    main(args.input)
