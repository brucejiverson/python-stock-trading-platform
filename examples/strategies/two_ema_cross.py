import uuid
from typing import List, Dict

import parallelized_algorithmic_trader.orders as orders                                     # trading
from parallelized_algorithmic_trader.strategy import StrategyBase                           # algorithm
from parallelized_algorithmic_trader.broker import SimulatedAccount
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping    # feature construction


"""Warning! Do no expect any of these to make you money! They are just examples to help you get started."""
class TwoEMACross(StrategyBase):
    def __init__(
        self, 
        account_number:uuid.UUID, 
        indicators:IndicatorMapping,
        tickers:List[str],
        log_level:int=None):
        name = self.__class__.__name__
        super().__init__(account_number, name, tickers, indicators)
        if log_level is not None: self.logger.setLevel(log_level)
        
    def get_indicator_mapping_by_name(self, name:str) -> IndicatorConfig:
        for ic in self.indicator_mapping:
            if ic.name == name: return ic
        raise ValueError(f'IndicatorConfig with name {name} not found')

    def act(self, account:SimulatedAccount, state:Dict[str, float]) -> orders.OrderBase | None:
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if account.has_pending_order:
            # self.logger.debug(f'waiting for order to execute...')
            return None

        if state[self.indicator_mapping[0].names[0]] > state[self.indicator_mapping[1].names[0]]:
            # Check if we are in the market
            if not account.has_exposure:   # check to buy
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.logger.debug('BUY CREATE')
                return [orders.MarketOrder(self.account_number, self._tickers[0], orders.OrderSide.BUY)]

        elif account.has_exposure:
            # SELL, SELL, SELL!!! (with all possible default parameters)
            self.logger.debug('SELL CREATE')
            return [orders.MarketOrder(self.account_number, self._tickers[0], orders.OrderSide.SELL)]

        return None
