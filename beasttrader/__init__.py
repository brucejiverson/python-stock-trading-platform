import logging

from beasttrader.util import get_simple_logger

_ = get_simple_logger('beasttrader', logging.DEBUG) # you have to initialize with DEBUG to allow the other modules to set their own levels
_.setLevel(logging.WARNING)