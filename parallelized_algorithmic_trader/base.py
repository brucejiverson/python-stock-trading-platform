from parallelized_algorithmic_trader.util import get_logger


class Base(object):
    def __init__(self, name):
        self.logger = get_logger(name)
        self.logger.debug(f'Initializing...')