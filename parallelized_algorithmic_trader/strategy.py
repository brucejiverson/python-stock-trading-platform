
from enum import Enum
from typing import List, Dict
import uuid
import numpy as np
from dataclasses import dataclass, field

from parallelized_algorithmic_trader.orders import MarketOrder, OrderBase, OrderSide
from parallelized_algorithmic_trader.indicators import IndicatorMapping
from parallelized_algorithmic_trader.base import Base
from parallelized_algorithmic_trader.broker import SimulatedAccount


class PositionSizer(Enum):
    EQUAL_ALLOCATION = "EQUAL_ALLOCATION"   # equal allocation to each ticker


class StrategyBase(Base):
    """This is the class from which all other strategies inherit from.
    The purpose is to enforce certain standards between all of the strategies, 
    and to make additional functionality scale. Every subclass should have a method called 'act'
    which takes in the current state of the environment and returns an action to take."""
    
    __slots__ = ('_tickers', 'indicator_mapping', 'account_number', 'ordered_feature_names', '_position_sizer')
    
    def __init__(self, 
        account_number:uuid.UUID, 
        name:str,
        tickers:List[str],
        indicator_mapping:IndicatorMapping=[],
        position_sizer:PositionSizer=PositionSizer.EQUAL_ALLOCATION
        ):

        super().__init__(name)

        self._tickers = tickers
        self.indicator_mapping = indicator_mapping
        self.account_number = account_number
        self._position_sizer = position_sizer
        self.ordered_feature_names = [n for sublist in [ind.names for ind in self.indicator_mapping] for n in sublist]

    def _get_sized_market_buy_order(self, ticker:str, account:SimulatedAccount) -> MarketOrder:
        """Returns a market order for the given ticker and side.
        
        :param ticker: ticker to buy
        :param account: account to buy from
        """
        if self._position_sizer == PositionSizer.EQUAL_ALLOCATION:
            n_equities_held = sum([1 for t in self._tickers if account.check_for_exposure(t)])
            position_size_dollars = account.cash/ ( len(self._tickers) - n_equities_held)
            return MarketOrder(self.account_number, ticker, OrderSide.BUY, position_size_dollars)
        else:
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
    quantity:int = 1

    def instantiate(self, uuid:uuid.UUID) -> StrategyBase:
        return self.strategy(uuid, self.indicator_mapping, self.tickers, *self.args, **self.kwargs)


StrategySet = Dict[uuid.UUID, StrategyBase]
