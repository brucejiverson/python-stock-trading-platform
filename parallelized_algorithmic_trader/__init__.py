import logging

from parallelized_algorithmic_trader.util import create_formatted_logger


root_logger = create_formatted_logger('pat', logging.DEBUG, False) # you have to initialize with DEBUG to allow the other modules to set their own levels
root_logger.setLevel(logging.WARNING)