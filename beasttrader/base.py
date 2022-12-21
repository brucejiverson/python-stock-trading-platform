import logging


class Base:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.debug(f'Initializing...')