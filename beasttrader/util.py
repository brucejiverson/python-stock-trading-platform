from __future__ import annotations
import logging
from typing import Optional, Mapping
import string
import os
import json
import datetime
from typing import Dict


# get the path to the installed directory
INSTALL_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_DIRECTORY = os.path.join(INSTALL_PATH, 'models')
DATA_DIRECTORY = os.path.join(INSTALL_PATH, 'data')
LOG_DIRECTORY = os.path.join(INSTALL_PATH, 'logs')
RESULTS_DIRECTORY = os.path.join(INSTALL_PATH, 'results')


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    # if iteration == 0:
        # print()
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def maybe_make_dir(directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Creating a directory {directory}')


def get_simple_logger(name: str, verbosity=logging.INFO, file=None) -> logging.Logger:
    logger = logging.getLogger(name)

    # Create formatters and add it to handlers, add the handles to the logger
    console_handler = logging.StreamHandler() 
    console_handler.setLevel(verbosity)
    # console_format = logging.Formatter('%(asctime)s; %(name)s; %(levelname)s; %(message)s')
    console_format = ColoredConsoleFormatter()
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Create handlers. These say what to do with items as they get added to the logger
    if file is not None:
        # Create formatters and add it to handlers, add the handles to the logger
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(verbosity)
        file_format = logging.Formatter('%(name)s; %(levelname)s; %(message)s') # %(asctime)s; 
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    logger.setLevel(verbosity)
    logger.info(f'Initialized logger {name}')
    return logger


class ColoredConsoleFormatter(logging.Formatter):
    """
    A logging formatter that differentiates log levels using console color
    escapes. This works by injecting custom attributes into the record before
    forwarding it to the parent class.
    This implementation currently requires the ``'{'`` format style.
    :param str fmt: a standard str.format-style format string. Arbitrary color
    :param str datefmt: date format string
    :param colorspec: a mapping of log level to ``{color name: color definition}``.
        Color names are expected to follow the pattern ``nameColor`` and will not be
        properly handled otherwise. The default colorspec defines ``levelColor`` and
        ``messageColor``.
    """

    colors = {
        logging.DEBUG: {"levelColor": "\033[0;37m", "messageColor": "\033[0;37m"},
        logging.INFO: {"levelColor": "\033[1;32m", "messageColor": "\033[0m"},
        logging.WARNING: {"levelColor": "\033[1;33m", "messageColor": "\033[0m"},
        logging.ERROR: {"levelColor": "\033[1;31m", "messageColor": "\033[0;1m"},
        logging.CRITICAL: {"levelColor": "\033[1;37;41m", "messageColor": ""},
    }

    default_format = "{levelColor}{levelname:>8}{messageColor} {name}: {message}"
    # default_format = "{messageColor}{asctime} {levelColor}{levelname:>8}{messageColor} {name}: {message}"

    COLOR_RESET = "\033[0m"

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        colorspec: Mapping[int, Mapping[str, str]] = None,
    ):
        if not colorspec:
            colorspec = self.colors

        # there is, as far as I know, no reason to support a nonexistent format
        # string.
        if fmt is None:
            fmt = self.default_format

        if not fmt.endswith(self.COLOR_RESET):
            fmt += self.COLOR_RESET

        # sanity check that the colorspec is supported
        color_keys = set(
            key
            for _, key, *_ in string.Formatter().parse(fmt)
            if key and key.endswith("Color")
        )

        for level, color_defs in colorspec.items():
            if not color_keys.issubset(color_defs):
                missing_colors = {*color_keys}.difference(color_defs)
                raise ValueError(
                    f"The provided colorspec is missing the color definitions {missing_colors} at level {level}"
                )

        self._defined_levels = sorted(colorspec.keys())
        self._colorspec = colorspec
        super().__init__(fmt, datefmt, style="{")

    def setColorspec(self, spec: Mapping[int, Mapping[str, str]]):
        new_colorspec = {level: colors.copy() for level, colors in spec.items()}
        self._colorspec = new_colorspec

    def _get_level_spec(self, level: int):
        try:
            return self._colorspec[level]
        except KeyError:
            for defined in self._defined_levels:
                if level <= defined:
                    break

            # memoize
            approximate_level = self._colorspec[defined]
            self._colorspec[level] = approximate_level
            return approximate_level

    def format(self, record):
        """Format the specified record as text."""
        for name, color in self._get_level_spec(record.levelno).items():
            setattr(record, name, color)

        return super().format(record)


def read_config() -> Dict:
    """Reads in the config JSON file, processes it into the proper data types, returns the dictionary"""

    path = './config/config.json'
    with open(path, 'r') as f:
        config = json.loads(f.read())
        return config


def write_config(config: Dict):
    with open('config/config.json', 'w') as f:
        json.dump(config, f, indent=4)

