import uuid

from parallelized_algorithmic_trader.indicators import IndicatorMapping
from parallelized_algorithmic_trader.strategy import StrategyBase
from parallelized_algorithmic_trader.orders import OrderBase, MarketOrder, OrderSide
from parallelized_algorithmic_trader.trading.simulated_broker import Account


class RandomBot(StrategyBase):
    """On init and after each order, this strategy will wait a random number of timesteps 
    (within some bounds) before placing another order."""
    def __init__(
        self, 
        account_number:uuid.UUID, 
        indicator_mapping:IndicatorMapping,
        tickers:list[str],
        log_level:int=None):
        name = __name__ + '.' + self.__class__.__name__ + '.' + str(account_number)[:8]
        super().__init__(account_number, name, tickers, indicator_mapping)

        if log_level is not None: self.logger.setLevel(log_level)

        self.max_time_between_orders:int = 9*60                            # minutes
        self.min_time_between_orders:int = 0.2*60

        self.rng = np.random.default_rng(60)                                # random number generator
        self.time_til_next_order:int = 0
        self.set_time_til_next_order()

    @property
    def time_to_enter_order(self) -> bool:
        return self.time_til_next_order <= 0 

    def set_time_til_next_order(self):
        self.time_til_next_order = self.rng.integers(
            high=self.max_time_between_orders, 
            low=self.min_time_between_orders
        )
        self.logger.debug(f'time til next order is now: {self.time_til_next_order} timesteps.')

    def act(self, _:dict[str, float]=None) -> list[OrderBase] | None:
        # self.log('Close, %.2f' % self.dataclose[0])
        self.time_til_next_order -= 1
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self._pending_order:
            self.logger.debug(f'waiting for order to execute...')
            return None

        if self.time_to_enter_order:
            # Check if we are in the market
            if not self._position:   # check to buy
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.logger.debug('BUY CREATE')

                # Keep track of the created order to avoid a 2nd order
                self._pending_order = True
                self.set_time_til_next_order()
                return MarketOrder(self._tickers[0], OrderSide.BUY, self.account_number)
            elif self._position:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.logger.debug('SELL CREATE')

                # Keep track of the created order to avoid a 2nd order
                self._pending_order = True
                self.set_time_til_next_order()
                return MarketOrder(self._tickers[0], OrderSide.SELL, self.account_number)
        return None


class NeverSell(StrategyBase):
    def __init__(
        self, 
        account_number:uuid.UUID, 
        indicator_mapping:IndicatorMapping,
        tickers:list[str],
        log_level:int=None):
        name = __name__ + '.' + self.__class__.__name__ + '.' + str(account_number)[:8]
        super().__init__(account_number, name, tickers, indicator_mapping)

        if log_level is not None: self.logger.setLevel(log_level)

    def act(self, _:dict[str, float]=None) -> OrderBase | None:
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self._pending_order:
            self.logger.debug(f'waiting for order to execute...')
            return None

        # Check if we are in the market
        if not self._position:   # check to buy
            # BUY, BUY, BUY!!! (with all possible default parameters)
            self.logger.debug('BUY CREATE')

            # Keep track of the created order to avoid a 2nd order
            self._pending_order = True
            return MarketOrder(self._tickers[0], OrderSide.BUY, self.account_number)
        return None


class SimpleSignal(StrategyBase):
    """Give it the kwg(s) for buy and sell for indicators that signal buy > 0.5 and sell < 0.5"""
    def __init__(
        self, 
        account_number:uuid.UUID, 
        indicator_mapping:IndicatorMapping,
        tickers:list[str], 
        log_level:int=None, 
        buy_ind_kwd=None,
        sell_ind_kwd=None,):

        name ='parallelized_algorithmic_trader' + '.' + self.__class__.__name__ + '.' + str(account_number)[:8]
        super().__init__(account_number, name, tickers, indicator_mapping)

        if log_level is not None: 
            self.logger.warning(F'Setting log level to: {log_level}')
            self.logger.setLevel(log_level)

        if buy_ind_kwd is None:
            self._buy_signal_name:str = indicator_mapping[0].names[0]
        else:
            self._buy_signal_name:str = [name for config in self.indicator_mapping for name in config.names if buy_ind_kwd in name][0]
        if sell_ind_kwd is None:
            self._sell_signal_name:str = self._buy_signal_name
        else:
            self._sell_signal_name:str = [name for config in self.indicator_mapping for name in config.names if sell_ind_kwd in name][0]
        self.logger.info(f'Using signal name for buy: {self._buy_signal_name}, Using signal name for buy: {self._sell_signal_name}')
    
    def act(self, account:Account, cur_state:dict[str, float]) -> OrderBase | None:
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if account.has_pending_order: # and type(account.get_pending_order()) != TrailingStopOrder:
            return None
        # Check if we are in the market
        if not account.has_exposure:                            # check to buy
            buy_signal = cur_state[self._buy_signal_name] > 0.5
            if buy_signal:
                self.logger.debug('BUY CREATE')
                return [MarketOrder(self.account_number, self._tickers[0], OrderSide.BUY)]
        else:                                                   # check to sell
            sell_signal = cur_state[self._sell_signal_name] < 0.5
            if sell_signal:
                self.logger.debug('SELL CREATE')
                return [MarketOrder(self.account_number, self._tickers[0], OrderSide.SELL)]
        return None
