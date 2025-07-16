"""
Utils
"""

import os
import logging


def get_console_logger(name: str = "ConsoleLogger", level: str = "INFO"):
    """
    To get a logger to print on console
    """
    logger = logging.getLogger(name)

    # to avoid duplication of logging
    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False

    return logger


# for the loading utility
def remove_path_from_ref(ref_pathname):
    """
    remove the path from source (ref)
    """
    ref = ref_pathname
    # check if / or \ is contained
    if len(ref_pathname.split(os.sep)) > 0:
        ref = ref_pathname.split(os.sep)[-1]

    return ref
