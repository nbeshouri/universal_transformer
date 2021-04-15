import logging
import os
import sys

PACKAGE_DIR = os.path.dirname(__file__)
CONTAINING_DIR = os.path.split(PACKAGE_DIR)[0]
DATA_DIR_PATH = os.path.join(PACKAGE_DIR, "data")


def _setup_logger():
    formatter = logging.Formatter(
        r"%(asctime)s [%(name)s] [%(levelname)s] %(message)s", r"[%Y-%m-%d %H:%M:%S %z]"
    )
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger = logging.getLogger("universal_transformer")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


logger = _setup_logger()
