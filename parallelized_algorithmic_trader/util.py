
import logging
from typing import Optional, Mapping, NamedTuple
import string
import os
from typing import Dict
from datetime import datetime

logger = logging.getLogger('pat')


# get the path to the installed directory
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(ROOT_PATH, '..')

MODEL_DIRECTORY = os.path.join(ROOT_PATH, 'models')
DATA_DIRECTORY = os.path.join(ROOT_PATH, 'data')
LOG_DIRECTORY = os.path.join(ROOT_PATH, 'logs')
ACCOUNT_HISTORY_DIRECTORY = os.path.join(ROOT_PATH, 'account_history')


# a named tuple representing a datetime range from start to end.
class DateRange(NamedTuple):
    """A named tuple representing a datetime range from start to end."""
    start: datetime
    end: datetime
    
    def __repr__(self) -> str:
        return f'from {self.start} to {self.end}'


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


def create_formatted_logger(name:str, verbosity:int=logging.INFO, save_to_file:bool=False) -> logging.Logger:
    logger = get_logger(name)

    # Create formatters and add it to handlers, add the handles to the logger
    console_handler = logging.StreamHandler() 
    console_handler.setLevel(verbosity)
    # console_format = logging.Formatter('%(asctime)s; %(name)s; %(levelname)s; %(message)s')
    console_format = ColoredConsoleFormatter()
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Create handlers. These say what to do with items as they get added to the logger
    if save_to_file is not None:
        # get the folder path by eliminating the file name
        # check if the folder exists. If not, create it
        file_path = os.path.join(LOG_DIRECTORY, 'log')
        # Create formatters and add it to handlers, add the handles to the logger
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(verbosity)
        file_format = logging.Formatter('%(name)s; %(levelname)s; %(message)s') # %(asctime)s; 
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    logger.setLevel(verbosity)
    logger.info(f'Initialized logger {name}')
    return logger


def get_logger(file_name:str) -> logging.Logger:
    """Fetches a logger by the file name updating the root name to be abbreviated"""
    logger_name = file_name.replace("parallelized_algorithmic_trader", "pat")
    if not "pat" in logger_name:
        logger_name = "pat." + logger_name 
    return logging.getLogger(logger_name)


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
