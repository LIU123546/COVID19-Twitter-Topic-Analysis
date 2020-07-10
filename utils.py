import os
import sys
import logging
from logging import Filter

import numpy as np
from allennlp.common.tee import TeeHandler
from gensim.corpora.dictionary import Dictionary


def seconds2clock(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02.0f}:{minutes:02.0f}:{secs:.6f}"


class ErrorFilter(Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """

    def filter(self, record):
        return record.levelno < logging.ERROR


def set_console_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)


def set_tee_logger(save_dir, file_friendly_logging=False):
    # https://github.com/allenai/allennlp/blob/master/allennlp/common/logging.py
    stdout_file = os.path.join(save_dir, "stdout.log")
    stderr_file = os.path.join(save_dir, "stderr.log")

    # Patch stdout/err.
    sys.stdout = TeeHandler(
        stdout_file,
        sys.stdout,
        file_friendly_terminal_output=file_friendly_logging,
    )
    sys.stderr = TeeHandler(
        stderr_file,
        sys.stderr,
        file_friendly_terminal_output=file_friendly_logging,
    )

    # Handlers for stdout/err logging
    output_handler = logging.StreamHandler(sys.stdout)
    error_handler = logging.StreamHandler(sys.stderr)

    root_logger = logging.getLogger()
    if len(root_logger.handlers) > 0:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    output_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    output_handler.setLevel(logging.DEBUG)
    error_handler.setLevel(logging.ERROR)

    # filter out everything at the ERROR or higher level for output stream
    # so that error messages don't appear twice in the logs.
    output_handler.addFilter(ErrorFilter())

    root_logger.addHandler(output_handler)
    root_logger.addHandler(error_handler)

    root_logger.setLevel(logging.DEBUG)
