from typing import Set, Optional, Any, Collection
import os
import pickle
import datetime
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from numba import jit, types
from numba.typed import Dict

from dataclasses import dataclass

from parallelized_algorithmic_trader.base import Base
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.orders import *
from parallelized_algorithmic_trader.util import ACCOUNT_HISTORY_DIRECTORY
from parallelized_algorithmic_trader.util import printProgressBar, get_logger
from parallelized_algorithmic_trader.data_management.data import CandleData


logger = get_logger(__name__ )
import logging
root_logger = logging.getLogger('pat')


class TradingHours(Enum):
    REGULAR = "REGULAR"
    EXTENDED = "EXTENDED"
    ALL_HOURS = "ALL_HOURS"


class Brokerage(Base):
    """The main type of environment. Subclasses might be real interfaces with different exchanges, or may be simulated exchanges for backtesting."""
    def __init__(self, name):
        super().__init__(name)

        self._tickers:Set[str] = set()
        self._current_data:pd.Series|None = None
        self._current_timestamp:pd.Timestamp|None = None
        # these two below params get set in the set_trading_hours method
        self._market_open_time:datetime.time = datetime.time(9, 30)
        self._market_close_time:datetime.time = datetime.time(16, 0)
        self._trading_hours = self.set_trading_hours(TradingHours.EXTENDED)

    def add_ticker(self, t:str|list[str]):
        """Informs the exchange to expect data related to this ticker to come in over time."""
        if isinstance(t, str):
            # add the ticker to the set of tickers
            self._tickers.add(t)
        elif isinstance(t, list):
            for ticker in t:
                self._tickers.add(ticker)
        self.logger.info(f'Added {t} to instruments list')

    def _set_expected_resolution(self, resolution:TemporalResolution):
        """Informs the exchange of the expected temporal resolution of the incoming data."""
        self._expected_resolution = resolution

    def set_trading_hours(self, trading_hours:TradingHours) -> TradingHours:
        """Limits the trading hours to the specified trading hours defined as follows: REGULAR: 9:30am - 4pm EST, EXTENDED: 9:30am - 8pm EST, ALL_HOURS: 24/7"""
        self._trading_hours = trading_hours
        if self._trading_hours == TradingHours.REGULAR:
            self._market_open_time = datetime.time(9, 30)
            self._market_close_time = datetime.time(16, 0)
        elif self._trading_hours == TradingHours.EXTENDED:
            self._market_open_time = datetime.time(8, 0)
            self._market_close_time = datetime.time(20, 0)
        elif self._trading_hours == TradingHours.ALL_HOURS:
            self._market_open_time = datetime.time(0, 0)
            self._market_close_time = datetime.time(23, 59)
        return self._trading_hours
    
    def _check_if_market_is_open(self, cur_time:pd.Timestamp=None) -> bool:
        """Returns True if the market is open for trading based on the set trading hours. Always is True if resolution is DAY or higher.
        
        Parameters:
            cur_time: The current time to check. If None, uses the current time in the self._current_timestamp
        """
        
        if any([self._expected_resolution in (TemporalResolution.DAY, TemporalResolution.WEEK), self._trading_hours in (TradingHours.ALL_HOURS, TradingHours.EXTENDED)]):
            return True
        
        if cur_time is None: cur_time = self._current_timestamp
        if not isinstance(cur_time, pd.Timestamp): raise NotImplementedError(f'Expected cur_time to be a pd.Timestamp, but got {type(cur_time)}')
        if self._market_open_time <= cur_time.time() <= self._market_close_time:
            return True
        else: return False


class SlippageModelsMarketOrders(Enum):
    """Slippage models"""
    NEXT_OPEN = "Orders executed at the open of the subsequent candle price"
    NEXT_CLOSE = "Orders executed at the CLOSE of the next candle"
    RANDOM_IN_NEXT_CANDLE = "Orders executed at a random price (uniform distribution) bounded between the current price and the next minute's CLOSE"
    RANDOM_AROUND_NEXT_CANDLE_OPEN = "Orders executed at a random price centered on the open of the subsequent candle with a standard deviation of 1/4 of the candle's range"


@dataclass
class Trade:
    """Holds all the information defining a trade, and provides some utility functions for analyzing performance."""
    buy:OrderBase
    sell:OrderBase
    max_draw_down:float|None = None
    max_percent_up:float|None = None

    @property
    def ticker(self) -> str:
        return self.buy.ticker

    def get_net_profit(self) -> float:
        """Returns the net profit of the trade, calculated as the difference between the buy and sell value.
        
        :return: float representing the net profit of the trade.
        """
        return self.sell.get_transaction_value() - self.buy.get_transaction_value()
    
    def get_profit_percent(self) -> float:
        """Returns the profit of the trade as a percentage gain or loss.
        
        :return: float representing the percentage gain or loss of the trade.
        """
        return (self.sell.execution_price - self.buy.execution_price) / self.buy.execution_price

    def get_duration(self) -> datetime.timedelta:
        """Returns the duration of the trade.
        
        :return: datetime.timedelta object representing the duration of the trade.
        """
        return self.sell.execution_timestamp - self.buy.execution_timestamp


# dict[ticker, dict[OrderSide, list[OrderBase]]]
OrderData = dict[str, dict[OrderSide, list[OrderBase]]] 
TradeSet = list[Trade]


class SimulatedAccount(Base):
    def __init__(self, 
                 account_number, 
                 starting_cash:float=10000, 
                 compile_trades_at_run_time:bool=False):
        super().__init__(__name__ + '.' + self.__class__.__name__ + '.' + str(account_number)[:4])
        self.account_number = account_number
        self.starting_cash = starting_cash
        self.cash:float = starting_cash             # this is the cash that the account has available to trade with
        self._order_history:OrderData = {}
        self._trade_history:TradeSet = []
        self.value_history:dict[pd.Timestamp, float] = {}
        self._equities:dict[str, float] = {}             # this dict uses the symbols as keys and the value is the quantity of shares that this account owns
        self._pending_orders:list[OrderBase] = []
        self._compile_trades_at_run_time:bool = compile_trades_at_run_time
      
    ########################################################
    ### Methods for submitting and managing orders
    ########################################################   
    def append_order_to_account_history(self, order:OrderBase):
        """Marks the current pending order as executed
        
        :param order: The order to mark as executed
        """
        order.open = False
        if order.ticker not in self._order_history:
            self._order_history[order.ticker] = {OrderSide.BUY:[], OrderSide.SELL:[]}
        self._order_history[order.ticker][order.side].append(order)
        
        if self._compile_trades_at_run_time:
            if order.side == OrderSide.SELL:
                trade = Trade(self._order_history[order.ticker][OrderSide.BUY][-1], order)
                self._trade_history.append(trade)

    #########################################################
    ### Methods for analysis of the account's performance ###
    #########################################################
    def set_time_index_for_val_history(self, time_index:pd.DatetimeIndex):
        """Sets the time index for the value history of the account"""
        self.value_history = {ts:val for ts, val in zip(time_index, self._tmp_val_hist)}
    
    def get_trades(self) -> TradeSet:
        return self._trade_history
    
    def parse_order_history_into_trades(self, price_history:pd.DataFrame|None=None) -> TradeSet:
        """Compiles the order history into a list of trades.
        
        :return: A list of trades.
        """
        self.logger.debug(f'Compiling order history into trades')
        trades:TradeSet = []
        self.logger.debug(f'There are {len(self._order_history)} tickers in the order history')

        try:
            for ticker in self._order_history.keys():
                # log n trades
                n_buys = len(self._order_history[ticker][OrderSide.BUY])
                n_sells = len(self._order_history[ticker][OrderSide.SELL])
                self.logger.debug(f'Compiling trades for {ticker}. There are {n_buys} buys and {n_sells} sells')
                self._order_history[ticker][OrderSide.BUY].sort(key=lambda o: o.execution_timestamp)
                self._order_history[ticker][OrderSide.SELL].sort(key=lambda o: o.execution_timestamp)

                # this implicitly assumes that the buys and sells alternate.
                for i in range(len(self._order_history[ticker][OrderSide.SELL])):
                    trades.append(Trade(self._order_history[ticker][OrderSide.BUY][i], self._order_history[ticker][OrderSide.SELL][i]))

            self._trade_history = trades
            
            if price_history is not None:
                self.calculate_meta_data_for_trades(price_history)
            return trades
        except Exception as e:
            self.logger.warning(f'Could not parse order history into trades. Error: {e}')
    
    def calculate_meta_data_for_trades(self, price_history:pd.DataFrame):
        """Compiles the order history of an account into a list of trades.
        
        :param account: The account to parse the order history of.
        :param price_history: The price history of the account. If this is not None, then the max drawdown and max percent up
        :return: A list of trades.
        """    
        # parse out the max drawdown and max percent up for each trade
        trade_index = 0
        flag = False
        max_draw_down = 0
        max_percent_up = 0

        for ts, row in price_history.iterrows():
            if trade_index >= len(self._trade_history):
                break
            trade = self._trade_history[trade_index]
            # monitor if we are in a trade
            if not flag and ts > trade.buy.execution_timestamp:
                flag = True
                
            # trade has ended
            if flag and ts > trade.sell.execution_timestamp:
                flag = False
                trade.max_percent_up = max_percent_up
                trade.max_draw_down = max_draw_down
                trade_index += 1

            # update the max drawdown and max percent up
            if flag:
                cur_percent_trade_value = (trade.buy.execution_price - row[trade.ticker+'_close']) / trade.buy.execution_price
                if cur_percent_trade_value > max_percent_up:
                    max_percent_up = cur_percent_trade_value
                if cur_percent_trade_value < max_draw_down:
                    max_draw_down = cur_percent_trade_value

    def get_all_buys(self) -> list[OrderBase]:
        """Returns a list of all buy orders that have been executed"""
        buys = []
        for ticker in self._order_history:
            buys.extend(self._order_history[ticker][OrderSide.BUY])
        return buys

    def get_all_sells(self) -> list[OrderBase]:
        """Returns a list of all sell orders that have been executed"""
        sells = []
        for ticker in self._order_history:
            sells.extend(self._order_history[ticker][OrderSide.SELL])
        return sells

    def get_history_as_list(self) -> list[float]:
        """Returns the value of the account as a list of floats"""
        return list(self.value_history.values())

    @classmethod
    def initialize_from_lite(
        cls, 
        tickers, 
        account_lite:list[float], 
        val_history:dict[pd.Timestamp, float], 
        order_submissions:list[list]) -> 'SimulatedAccount':
        acc = cls(None, val_history[next(iter(val_history))])
        logger.debug(f'Creating account from lite info. Tickers: {tickers}, account: {account_lite}')
        acc._equities = {ticker:shares for ticker, shares in zip(tickers, account_lite[1:])}
        
        # parse the orderlites into order objects for history
        new_order_hist = {}
        # type, side, ticker, shares, execution price, execution np.datetime64
        for orderlite in order_submissions:
            logger.debug(f'Parsing orderlite: {orderlite}')
            execution_price = orderlite[3] if len(orderlite) == 5 else orderlite[4]
            execution_ts = orderlite[4] if len(orderlite) == 5 else orderlite[5]
            shares = None if len(orderlite) == 5 else orderlite[3]        
            if orderlite[0] == 'MarketOrder':
                order = MarketOrder(None, orderlite[2], OrderSide[orderlite[1]], shares=shares)
                order.execution_timestamp = execution_ts    # pd.Timestamp(orderlite[5])
                order.execution_price = execution_price
            else: raise NotImplementedError(f'Unknown order type: {orderlite[0]}')
            if not order.ticker in new_order_hist:
                new_order_hist[order.ticker] = {OrderSide.BUY:[], OrderSide.SELL:[]}
            new_order_hist[order.ticker][order.side].append(order)
        
        # set the order history and value history
        acc._order_history = new_order_hist
        acc.value_history = val_history
        return acc
    
AccountSet = dict[uuid.UUID, SimulatedAccount]


@dataclass
class TradingHistory:
    """A class to hold the history of an account."""
    strategy:Base       # This should be the class type that the strategy is derived from
    account:SimulatedAccount
    tickers:list[str]
    resolution:TemporalResolution

    @property
    def final_value(self) -> float:
        v = list(self.account.value_history.values())[-1]
        return v
    
    def save_to_file(self, result:'TradingHistory', file_name:str='latest.pkl'):
        """Save the account history to file in binary format using the pickle module to the ./account_history directory.
        
        Parameters:
        -----------
        result: TradingHistory object to save
        file_path: The path to save the file to
        
        """
        
        file_path = os.path.join(ACCOUNT_HISTORY_DIRECTORY, file_name)
        print(f'Saving account history to {file_path}')
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

    @staticmethod
    def load_from_file(file_name:str='latest.pkl') -> 'TradingHistory':
        """Load the specified account history from file in the ./account_history directory.
        
        Parameters:
        file_name: The name of the file to load
        """
        file_path = os.path.join(ACCOUNT_HISTORY_DIRECTORY, file_name)
        with open(file_path, 'rb') as f:
            return pickle.load(f)


TradingHistorySet = list[TradingHistory]


################################################
############ Functions for pricing  ############
################################################
@jit(nopython=True, cache=True)
def get_cur_open(ticker:types.unicode_type, cur_data:np.ndarray, ticker_price_idxs) -> types.float64:
    return cur_data[ticker_price_idxs[ticker+'_open']]
    
@jit(nopython=True, cache=True)
def get_cur_high(ticker:types.unicode_type, cur_data:np.ndarray, ticker_price_idxs) -> types.float64:
    return cur_data[ticker_price_idxs[ticker+'_high']]

@jit(nopython=True, cache=True)
def get_cur_low(ticker:types.unicode_type, cur_data:np.ndarray, ticker_price_idxs) -> types.float64:
    return cur_data[ticker_price_idxs[ticker+'_low']]

@jit(nopython=True, cache=True)
def get_cur_close(ticker:types.unicode_type, cur_data:np.ndarray, ticker_price_idxs) -> types.float64:
    return cur_data[ticker_price_idxs[ticker+'_close']]

# @jit('types.float64(types.float64[:], types.float64[:], types.uint16[:])', nopython=True, cache=True)
@jit(nopython=True, cache=True)
def get_execution_price_for_market_order(
    order:types.float64[:],
    cur_data:np.ndarray,
    ticker_price_idxs,
    spread_percent:types.float64,
    slippage_model:types.unicode_type
    ) -> float:
    """Returns the execution price for an order based on the current candle, the slippage model, and the spread. 
    Does not update the variables of the order or the account. 
    
    Parameters:
        spread_percent: The spread of the exchange as a percentage
        slippage_model: The slippage model to use
        data_res: The resolution of the data in time (ie hourly, minute, etc)
        order: The order to get the execution price for
    """

    # logger.debug(f'Getting execution price for order: {order}')
    # switch on the order type
    # assert order[0] == 'MarketOrder', f'Unknown order type: {type(order)}'
    if order[0] != 'MarketOrder': raise ValueError(f'Unknown order type: {order[0]}')
    # get the current price data for the relevent instrument
    if slippage_model == 'NEXT_OPEN':
        # logger.debug(f'Getting execution price for order {order} using slippage model {slippage_model}, \ndata: {cur_data}, idxs {ticker_price_idxs}')
        price = get_cur_open(order[2], cur_data, ticker_price_idxs)
    elif slippage_model == 'NEXT_CLOSE':
        price = get_cur_close(order[2], cur_data, ticker_price_idxs)
    elif slippage_model == 'RANDOM_IN_NEXT_CANDLE':
        high = get_cur_high(order[2], cur_data, ticker_price_idxs)
        low = get_cur_low(order[2], cur_data, ticker_price_idxs)
        price = np.random.uniform(high, low)
    elif slippage_model == 'RANDOM_AROUND_NEXT_CANDLE_OPEN':
        open = get_cur_open(order[2], cur_data, ticker_price_idxs)
        high = get_cur_high(order[2], cur_data, ticker_price_idxs)
        low = get_cur_low(order[2], cur_data, ticker_price_idxs)
        
        range = abs(high - low)
        price = np.random.normal(open, range)
    else: raise ValueError(f'Unknown slippage model: {slippage_model}')
    
    # account for the spread
    if order[1] == 'BUY':
        # logger.debug(f'Incorporating spread in price {price} for order {order}, spread percent: {spread_percent}')
        price = price * (1 + spread_percent/(100))
    elif order[1] == 'SELL':
        price = price * (1 - spread_percent/(100))
    return round(price, 2)

################################################
############ Functions for order flow ##########
################################################
@jit(nopython=True, cache=True)    
def execute_buy_order(
    account:dict[types.unicode_type, types.float64], 
    execution_price:types.float64, 
    order:list, 
    shares_idx:types.int8,
    commission_percent:types.float64=0.0):
    
    commission = execution_price * commission_percent/100
    account[0] -= commission
    
    shares_quantity = int(account[0] / execution_price)
    total_trade_value = shares_quantity * execution_price
        
    # Do the trade
    account[0] -= total_trade_value
    account[shares_idx] += shares_quantity
    # logger.info(f"Executed {order[0]} to BUY {shares_quantity:.2f} shares of {order[2]} at ${execution_price:.2f} totaling ${total_trade_value:.2f} acc: {account}")  #  Open: ${:.2f}, Close: ${:.2f} at {
    # logger.debug(f'Updated holdings: {account}')
    
    
@jit(nopython=True, cache=True)    
def execute_sell_order(
    account:types.float64[:], 
    execution_price:types.float64, 
    order:list,
    shares_idx:types.int8,):
    
    quantity = account[shares_idx]
    total_trade_value = execution_price * account[shares_idx]

    account[0] += total_trade_value
    account[shares_idx] -= account[shares_idx]
    # logger.info("  Executed {} to SELL {:.2f} shares of {} at ${:.2f} totaling ${:.2f} account {}.".format(   #  Open: ${:.2f}, Close: ${:.2f} at {
    #     order[0],
    #     quantity,
    #     order[2],
    #     execution_price,
    #     total_trade_value,
    #     account
    #     # get_cur_open(order[2]),
    #     # get_cur_close(order[2]),
    #     # _current_timestamp
    # ))
    
   