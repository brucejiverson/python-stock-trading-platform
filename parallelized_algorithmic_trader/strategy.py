
from enum import Enum
from typing import List, Dict, Tuple, Any
import uuid
import numpy as np
from dataclasses import dataclass, field

from parallelized_algorithmic_trader.orders import MarketOrder, OrderBase, OrderSide, LimitOrder, StopOrder
from parallelized_algorithmic_trader.indicators import IndicatorMapping
from parallelized_algorithmic_trader.base import Base
from parallelized_algorithmic_trader.trading.simulated_broker import SimulatedAccount


class PositionSizer(Enum):
    # the size of order is calculated to be equal for each ticker. So if you have 3 tickers and 100 dollars and place an order for ticker1, you will spend $33 on that order.
    FULL_AVAILABLE = "FULL_AVAILABLE"
    
    # order side is calculated as fixed percentage (given as a parameter) of the current account value
    FIXED_PERCENTAGE = "FIXED_PERCENTAGE"
    

class StrategyBase(Base):
    """This is the class from which all other strategies inherit from.
    The purpose is to enforce certain standards between all of the strategies, 
    and to make additional functionality scale. Every subclass should have a method called 'act'
    which takes in the current state of the environment and returns an action to take."""
    
    __slots__ = ('_tickers', 'indicator_mapping', 'account_number', 'ordered_feature_names', '_position_sizer')
    
    def __init__(self, 
        account_number:uuid.UUID|int, 
        name:str,
        tickers:List[str],
        indicator_mapping:IndicatorMapping=[],
        position_sizer:PositionSizer=PositionSizer.FULL_AVAILABLE,):

        super().__init__(name)

        self._tickers = tickers
        self.indicator_mapping = indicator_mapping
        self.logger.debug(f'Indicator mapping: {self.indicator_mapping}')
        self.account_number = account_number
        
        self._position_sizer = position_sizer
        self._position_sizer_args:Tuple[Any] = ()
        
        self.ordered_feature_names = [n for sublist in [ind.names for ind in self.indicator_mapping] for n in sublist]
        self.logger.debug(f'Ordered feature names: {self.ordered_feature_names}')

    def set_position_sizer(self, position_sizer:PositionSizer, position_sizer_args:tuple=()):
        assert isinstance(position_sizer, PositionSizer)
        assert isinstance(position_sizer_args, tuple)
        self.logger.warning(f'Position sizer set to {position_sizer.name} with args {position_sizer_args}')
        
        self._position_sizer = position_sizer
        self._position_sizer_args = position_sizer_args

    def get_sized_market_order(self, ticker:str, account:SimulatedAccount, side:OrderSide=OrderSide.BUY, cur_state:Dict[str, float]=None) -> MarketOrder:
        """Returns a market order for the given ticker and side.
        
        :param ticker: ticker to buy
        :param account: account to buy from
        """
        
        match self._position_sizer:
            case PositionSizer.FULL_AVAILABLE:
                match side:
                    case OrderSide.BUY:
                        n_equities_held = sum([1 for t in self._tickers if account.check_for_exposure(t)])
                        position_size_dollars = account.cash/ ( len(self._tickers) - n_equities_held)
                        position_size_shares = position_size_dollars / self._get_cur_price_from_state(cur_state, ticker)
                        return MarketOrder(self.account_number, ticker, side, shares=position_size_shares)
                    case OrderSide.SELL:
                        return MarketOrder(self.account_number, ticker, side, shares=account.get_exposure(ticker))
                                                                
            case PositionSizer.FIXED_PERCENTAGE:
                percentage = self._position_sizer_args[0]
                assert isinstance(percentage, float)
                assert percentage > 0 and percentage < 1
                position_size_dollars = account.get_most_recent_valuation() * percentage
                
                match side:
                    case OrderSide.BUY:
                        position_size_dollars = min(position_size_dollars, account.cash)
                    case OrderSide.SELL: 
                        self.logger.debug(f'For now, defaulting to sell all shares for {ticker} despite position sizer being set to {self._position_sizer.name}')
                        return MarketOrder(self.account_number, ticker, side, shares=account.get_exposure(ticker))
                        # """Sells are a"""
                        # assert cur_state is not None, "cur_state must be provided if side is SELL for fixed percentage."
                        # available_shares = account.get_exposure(ticker)
                        # position_size_dollars = min(position_size_dollars, available_shares * self._get_cur_price_from_state(cur_state, ticker))
                        raise NotImplementedError("Selling is not yet supported for fixed percentage position sizing.")
            case _:
                raise NotImplementedError("Only equal allocation is supported at this time.")

    def _get_cur_price_from_state(self, cur_state:Dict[str, float], ticker:str=None) -> float:
        """Returns the current price for the given ticker from the state
        
        If ticker is none, returns for the first ticker in the self._tickers attr
        
        :param cur_state: current state of the environment which much include the close price for ticker
        :param ticker: ticker to get the close price for
        """
        if ticker is None:
            return cur_state[self._tickers[0]+'_close']
        else:
            return cur_state[ticker+'_close']

    def _get_sized_limit_order(self, ticker:str, account:SimulatedAccount, side:OrderSide, price:float, cur_state:Dict[str, float]=None):
        """Returns a stop or limit order for the given ticker and side.
        
        :param ticker: ticker to buy
        :param account: account to buy from
        :param side: side of the order
        :param price: price to set the order at
        :param cur_state: current state of the environment
        """
        
        assert side in [OrderSide.BUY, OrderSide.SELL]

        match self._position_sizer:
            case PositionSizer.FULL_AVAILABLE:
                match side:
                    case OrderSide.BUY:
                        n_equities_held = sum([1 for t in self._tickers if account.check_for_exposure(t)])
                        position_size_dollars = account.cash / ( len(self._tickers) - n_equities_held)
                        return LimitOrder(self.account_number, ticker, side, dollars=position_size_dollars, limit_price=price)
                    case OrderSide.SELL:
                        self.logger.debug(f'For now, defaulting to sell all shares for {ticker} despite position sizer being set to {self._position_sizer.name}')
                        return LimitOrder(self.account_number, ticker, side, shares=account.get_exposure(ticker), price=0)
                
            case PositionSizer.FIXED_PERCENTAGE:
                match side:
                    case OrderSide.BUY:
                        percentage = self._position_sizer_args[0]
                        assert isinstance(percentage, float)
                        assert percentage > 0 and percentage < 1
                        position_size_dollars = account.get_most_recent_valuation() * percentage
                        position_size_dollars = min(position_size_dollars, account.cash)
                        return LimitOrder(self.account_number, ticker, side, dollars=position_size_dollars, limit_price=price)
                    case OrderSide.SELL:
                        assert cur_state is not None, "cur_state must be provided if side is SELL for fixed percentage."
                        available_shares = account.get_exposure(ticker)
                        percentage = self._position_sizer_args[0]
                        assert isinstance(percentage, float)
                        assert percentage > 0 and percentage < 1
                        position_size_dollars = account.get_most_recent_valuation() * percentage
                        position_size_dollars = min(position_size_dollars, available_shares * self._get_cur_price_from_state(cur_state, ticker))
                        return LimitOrder(self.account_number, ticker, side, dollars=position_size_dollars, limit_price=price)

    def act(self, state:Dict[str, float], account:SimulatedAccount) -> List[OrderBase] | None:
        """This method should be overridden by all subclasses. It should take in the current state of the environment and return an action to take.
        This method exists here in this form for type hinting purposes only"""
        raise NotImplementedError("This method should be overridden by all subclasses.")

    
@dataclass(slots=True)
class StrategyConfig:
    """This class holds all of the information defining a strategy, and can build an instance of that strategy."""
    indicator_mapping:IndicatorMapping
    strategy:type
    tickers:List[str] = field(default_factory=list)
    args:tuple = field(default_factory=tuple)
    kwargs:dict = field(default_factory=dict)

    def instantiate(self, uuid:uuid.UUID|int) -> StrategyBase:
        return self.strategy(uuid, self.indicator_mapping, self.tickers, *self.args, **self.kwargs)


StrategySet = Dict[uuid.UUID, StrategyBase]
