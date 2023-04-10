import uuid
from typing import List, Dict

import parallelized_algorithmic_trader.orders as orders                                     # trading
from parallelized_algorithmic_trader.strategy import StrategyBase                           # algorithm
from parallelized_algorithmic_trader.trading.simulated_broker import SimulatedAccount
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping    # feature construction


"""Warning! Do no expect any of these to make you money! They are just examples to help you get started."""
class TwoEMACrossWithStopLimit(StrategyBase):
    """A strategy that buys when the fast EMA crosses above the slow EMA and sells when the fast EMA crosses below the slow EMA. Also includes a stop loss."""
    def __init__(
        self, 
        account_number:uuid.UUID, 
        indicators:IndicatorMapping,
        tickers:List[str],
        log_level:int=None):
        name = self.__class__.__name__
        super().__init__(account_number, name, tickers, indicators)
        if log_level is not None: self.logger.setLevel(log_level)
        self._stop_loss_price = None
        self._take_profit_price = None
                
        # parse the indicator mapping for the EMA with the smallest and largest window
        fast_idx = None
        slow_idx = None
        cur_price_idx = None
        smallest_period_found = None
        
        for idx, ic in enumerate(self.indicator_mapping):
            if 'EMA' in ic.config_name:
                if smallest_period_found is None:
                    smallest_period_found = ic.args[0]
                    fast_idx = idx
                elif ic.args[0] < smallest_period_found:
                    smallest_period_found = ic.args[0]
                    fast_idx = idx
                else:
                    slow_idx = idx
            elif 'close' in ic.config_name:
                cur_price_idx = idx
        
        self._fast_ema_name = self.indicator_mapping[fast_idx].names[0]
        self._slow_ema_name = self.indicator_mapping[slow_idx].names[0]
        self._price_name = self.indicator_mapping[cur_price_idx].target
        self.logger.info(f'Using fast EMA: {self._fast_ema_name}, slow EMA: {self._slow_ema_name}, price: {self._price_name}')
        self._stop_order_placed = False
        self._stopped_out = False
        
    def get_indicator_mapping_by_name(self, name:str) -> IndicatorConfig:
        for ic in self.indicator_mapping:
            if ic.name == name: return ic
        raise ValueError(f'IndicatorConfig with name {name} not found')

    def act(self, account:SimulatedAccount, state:Dict[str, float]) -> List[orders.OrderBase] | None:
        fast_ema_val = state[self._fast_ema_name]
        slow_ema_val = state[self._slow_ema_name]
        
        # check to see if we got stopped out
        has_pending_stop_order = any([isinstance(o, orders.StopOrder) for o in account._pending_orders])
        has_pending_limit_order = any([isinstance(o, orders.LimitOrder) for o in account._pending_orders])
        if self._stop_order_placed:
            if not has_pending_stop_order or not has_pending_limit_order:
                self.logger.debug(f'Detected that stop order or limit order triggered. Canceling any remaining orders.')
                self._stopped_out = True
                self._stop_order_placed = False
                # cancel the limit sell order
                return [orders.CancelAllOrders(self.account_number)]

        # entry condition
        if all([
            fast_ema_val > slow_ema_val, 
            not account.has_exposure, 
            not account.has_pending_order, 
            not self._stopped_out]):
            self.logger.debug('BUY CREATE')
            return [orders.MarketOrder(self.account_number, self._tickers[0], orders.OrderSide.BUY)]

        # set a stop loss
        if account.has_exposure and not account.has_pending_order:
            percent = 0.01
            entry_price = account._order_history[self._tickers[0]][orders.OrderSide.BUY][-1].execution_price
            stop_price = entry_price * (1 - percent)
            stop_price = round(stop_price, 2)
            self.logger.debug(f'Setting a stop loss at {percent} percent, or ${stop_price}')
            self._stop_order_placed = True
            return [
                orders.StopOrder(self.account_number, self._tickers[0], orders.OrderSide.SELL, stop_price),
                orders.LimitOrder(self.account_number, self._tickers[0], orders.OrderSide.SELL, entry_price * (1 + percent*2))
                ]

        # exit condition: cancel stop loss and do a market order
        if fast_ema_val < slow_ema_val:
            if account.has_exposure:
                order_actions = []
                if account.has_pending_order:
                    order_actions.append(orders.CancelAllOrders(self.account_number))
                order_actions.append(orders.MarketOrder(self.account_number, self._tickers[0], orders.OrderSide.SELL))
                return order_actions
            # reset the flag indicating that we have been stopped out
            elif self._stopped_out:
                self._stopped_out = False
                
        return None


class TwoEMACross(StrategyBase):
    """A strategy that buys when the fast EMA crosses above the slow EMA and sells when the fast EMA crosses below the slow EMA. Also includes a stop loss."""
    def __init__(
        self, 
        account_number:uuid.UUID, 
        indicators:IndicatorMapping,
        tickers:List[str],
        log_level:int=None):
        name = self.__class__.__name__
        super().__init__(account_number, name, tickers, indicators)
        if log_level is not None: self.logger.setLevel(log_level)

        # parse the indicator mapping for the EMA with the smallest and largest window
        fast_idx = None
        slow_idx = None
        cur_price_idx = None
        smallest_period_found = None
        
        for idx, ic in enumerate(self.indicator_mapping):
            if 'EMA' in ic.config_name:
                if smallest_period_found is None:
                    smallest_period_found = ic.args[0]
                    fast_idx = idx
                elif ic.args[0] < smallest_period_found:
                    smallest_period_found = ic.args[0]
                    fast_idx = idx
                else:
                    slow_idx = idx
            elif 'close' in ic.config_name:
                cur_price_idx = idx
        
        self._fast_ema_name = self.indicator_mapping[fast_idx].names[0]
        self._slow_ema_name = self.indicator_mapping[slow_idx].names[0]
        self._price_name = self.indicator_mapping[cur_price_idx].target
        self.logger.info(f'Using fast EMA: {self._fast_ema_name}, slow EMA: {self._slow_ema_name}, price: {self._price_name}')
        
    def get_indicator_mapping_by_name(self, name:str) -> IndicatorConfig:
        for ic in self.indicator_mapping:
            if ic.name == name: return ic
        raise ValueError(f'IndicatorConfig with name {name} not found')

    def act(self, account:SimulatedAccount, state:Dict[str, float]) -> List[orders.OrderBase] | None:
        fast_ema_val = state[self._fast_ema_name]
        slow_ema_val = state[self._slow_ema_name]
        
        # entry condition
        if fast_ema_val > slow_ema_val and not account.has_exposure and not account.has_pending_order:
            return [orders.MarketOrder(self.account_number, self._tickers[0], orders.OrderSide.BUY)]

        # exit condition: cancel stop loss and do a market order
        if fast_ema_val < slow_ema_val and account.has_exposure and not account.has_pending_order:
            return [orders.MarketOrder(self.account_number, self._tickers[0], orders.OrderSide.SELL)]

