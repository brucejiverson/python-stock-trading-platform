from typing import Dict, Set, List
import numpy as np
import datetime
from enum import Enum
import uuid
import pandas as pd

from dataclasses import dataclass
import os
import pickle

from parallelized_algorithmic_trader.base import Base
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.orders import *
from parallelized_algorithmic_trader.util import ACCOUNT_HISTORY_DIRECTORY


class TradingHours(Enum):
    REGULAR = "REGULAR"
    EXTENDED = "EXTENDED"
    ALL_HOURS = "ALL_HOURS"


class Brokerage(Base):
    """The main type of environment. Subclasses might be real interfaces with different exchanges, or may be simulated exchanges for backtesting."""
    def __init__(self, name):
        super().__init__(name)

        self._tickers:Set[str] = set()
        self._current_candles:pd.Series|None = None

        # these two below params get set in the set_trading_hours method
        self._market_open_time:datetime.time = datetime.time(9, 30)
        self._market_close_time:datetime.time = datetime.time(16, 0)
        self._trading_hours = self.set_trading_hours(TradingHours.EXTENDED)

    def add_ticker(self, t:str|List[str]):
        """Informs the exchange to expect data related to this ticker to come in over time."""
        if isinstance(t, str):
            # add the ticker to the set of tickers
            self._tickers.add(t)
        elif isinstance(t, list):
            for ticker in t:
                self._tickers.add(ticker)
        self.logger.info(f'Added {t} to instruments list')

    def set_expected_resolution(self, resolution:TemporalResolution):
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
    
    def _check_if_market_is_open(self, ts:pd.Timestamp) -> bool:
        """Returns True if the market is open for trading based on the set trading hours."""
        if self._trading_hours == TradingHours.ALL_HOURS:
            return True
        if ts.time() >= self._market_open_time and ts.time() <= self._market_close_time:
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


# Dict[ticker, Dict[OrderSide, List[OrderBase]]]
OrderData = Dict[str, Dict[OrderSide, List[OrderBase]]] 
TradeSet = List[Trade]


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
        self.value_history:Dict[pd.Timestamp, float] = {}
        self._equities:Dict[str, float] = {}             # this dict uses the symbols as keys and the value is the quantity of shares that this account owns
        self._pending_orders:List[OrderBase] = []
        self._compile_trades_at_run_time:bool = compile_trades_at_run_time
        
    ########################################################
    ### Set up and clean up ######
    ########################################################
    def set_cash(self, cash:float):
        """Sets the cash balance of the account, and tracks this as the starting cash in USD.
        
        :param cash: The cash balance of the account in USD
        """
        self.cash = cash
        self.starting_cash = cash

    def reset(self):
        """Resets the account to its starting cash balance and wipes all information about the account's history"""
        self.cash = self.starting_cash
        self._order_history = []
        self.value_history = {}
        self._equities = {}
        self._pending_orders = []
    
    ########################################################
    ### Methods for providing telementry
    ########################################################
    @property
    def has_pending_order(self) -> bool:
        """Returns True if the account has a pending order"""
        return self._pending_orders is not None and len(self._pending_orders) > 0

    @property 
    def has_exposure(self) -> bool:
        """Returns True if the account has any exposure to any equities"""
        return sum(self._equities.values()) > 0

    def check_for_exposure(self, ticker:str) -> bool:
        """Returns True if the account has any exposure to the specified ticker
        
        :param ticker: The ticker to check for exposure to
        """
        return ticker in self._equities and self._equities[ticker] > 0
      
    def get_most_recent_valuation(self) -> float:
        """Returns the most recent valuation of the account"""
        return self.value_history[max(self.value_history.keys())]
     
    def get_exposure(self, ticker:str) -> float:
        """Returns the number of shares of the specified ticker that the account owns
        
        :param ticker: The ticker to check for exposure to
        """
        return self._equities[ticker]
    
    ########################################################
    ### Methods for submitting and managing orders
    ########################################################   
    def submit_new_orders(self, orders:List[OrderBase]):
        """Sets the pending order to the specified order
        
        :param orders: The order to set as the pending order
        """
        # if self.has_pending_order:
        #     self.logger.warning(f'Already has pending order(s). Cannot accept another order until the pending order is filled or cancelled. New order: {orders}')
        #     return
        for o in orders:
            if o.dollar_amount_to_use == 0 and o.shares == 0:
                self.logger.debug(f'Order {o} has no dollar amount or share amount specified. Cannot submit order.')
                continue
            
            self.logger.debug(f'New pending {o}')
            self._pending_orders.append(o)
    
    def get_pending_orders(self) -> List[OrderBase]:
        """Returns the pending order"""
        return self._pending_orders
    
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

    def cancel_all_pending_orders(self):
        """Cancels all pending orders"""
        self.logger.debug('Cancelling all pending orders')
        # remove all the order before the cancel order
        while type(self._pending_orders[0]) != CancelAllOrders:
            self.logger.debug(f'Removing {self._pending_orders[0]}')
            self._pending_orders.pop(0)
        # remove the cancel order
        self._pending_orders.pop(0)

    def cancel_order(self, order_to_cancel:OrderBase):
        """Searches through pending orders for the order with matching id(object) and cancels it. If no id is matching, uses ticker side and order type."""
        for i, order in enumerate(self._pending_orders):
            if id(order) == id(order_to_cancel):
                self.logger.debug(f'Canceling order: {order}')
                self._pending_orders.pop(i)
                return
        self.logger.warning(f'Could not find order to cancel: {order}')
        
    #########################################################
    ### Methods for analysis of the account's performance ###
    #########################################################
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

    def get_all_buys(self) -> List[OrderBase]:
        """Returns a list of all buy orders that have been executed"""
        buys = []
        for ticker in self._order_history:
            buys.extend(self._order_history[ticker][OrderSide.BUY])
        return buys

    def get_all_sells(self) -> List[OrderBase]:
        """Returns a list of all sell orders that have been executed"""
        sells = []
        for ticker in self._order_history:
            sells.extend(self._order_history[ticker][OrderSide.SELL])
        return sells

    def get_history_as_list(self) -> List[float]:
        """Returns the value of the account as a list of floats"""
        return list(self.value_history.values())


AccountSet = Dict[uuid.UUID, SimulatedAccount]


@dataclass
class AccountHistoryFileHandler:
    """A class to hold the history of an account."""
    strategy:Base       # This should be the class type that the strategy is derived from
    account:SimulatedAccount
    tickers:List[str]
    resolution:TemporalResolution

    @property
    def final_value(self):
        return list(self.account.value_history.values())[-1]

    def save_to_file(self, result:'AccountHistoryFileHandler', file_name:str='latest.pkl'):
        """Save the account history to file in binary format using the pickle module to the ./account_history directory.
        
        Parameters:
        -----------
        result: AccountHistoryFileHandler object to save
        file_path: The path to save the file to
        
        """
        
        file_path = os.path.join(ACCOUNT_HISTORY_DIRECTORY, file_name)
        print(f'Saving account history to {file_path}')
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

    @staticmethod
    def load_from_file(file_name:str='latest.pkl') -> 'AccountHistoryFileHandler':
        """Load the specified account history from file in the ./account_history directory.
        
        Parameters:
        file_name: The name of the file to load
        """
        file_path = os.path.join(ACCOUNT_HISTORY_DIRECTORY, file_name)
        with open(file_path, 'rb') as f:
            return pickle.load(f)


AccountHistoryFileHandlerSet = List[AccountHistoryFileHandler]


class SimulatedStockBrokerCandleData(Brokerage):
    """This class is used to simulate an exchange. It is used for backtesting and for training Strategys.
    
    
    """
    
    def __init__(
        self,
        spread_percent:float=0.1,                                   # the spread of the exchange as a percentage
        limit_order_slippage:float=0.02,                            # the slippage of limit and stop orders in dollars, default to two ticks. 
        market_order_slippage_model:SlippageModelsMarketOrders=SlippageModelsMarketOrders.NEXT_OPEN,     # the commission of the exchange as a percentage
        commision_percent:float=0.0,                                # the slippage of the exchange as a percentage
        log_level=None):
        
        super().__init__(__name__ + '.' + self.__class__.__name__)
        if log_level is not None: self.logger.setLevel(log_level)

        self.spread_percent:float = spread_percent
        self.commission_percent:float = commision_percent
        self.market_order_slippage_model:SlippageModelsMarketOrders = market_order_slippage_model
        self.limit_order_slippage:float = limit_order_slippage
        # self._accounts:Dict[uuid.UUID, SimulatedAccount]= {}
        self.logger.info(f'Initialized with spread: {self.spread_percent}%, commission: {self.commission_percent}, market order slippage: {self.market_order_slippage_model.name}, limit order slippage: ${self.limit_order_slippage}')

    def set_spread(self, spread:float):
        """Sets the spread (difference between bid and ask) of the exchange. Value should be a percentage
        
        :param spread: The spread of the exchange as a percentage
        """
        self.spread_percent = spread
        self.logger.debug(f'Spread set to {spread}')
    
    def set_commission(self, commission:float):
        """Sets the commission of the exchange. Value should be a percentage.
        
        :param commission: The commission of the exchange as a percentage
        """
        self.commission_percent = commission
        self.logger.debug(f'Commission set to {commission}')
    
    def set_slippage_model(self, model:SlippageModelsMarketOrders):
        """Sets the slippage of the exchange. Value should be a percentage
        
        :param model: The slippage model to use
        """
        self.slippage = model
        self.logger.debug(f'Slippage set to {model.name}, which means {model.value}')

    def get_account_value(self, account:SimulatedAccount) -> float:
        """Returns the current value of the account in USD
        
        :param account: The account to get the value of
        """
        value = account.cash
        for ticker, quantity in account._equities.items():
            value += quantity * self._current_candles[ticker+'_close']
        return value
        
    def reset(self):
        """Resets the market to its initial state, clearing all orders and price data"""

        self._tickers = set()

    def clean_up(self):
        """Cleans up the market, removing any references to the market. 
        
        (delete information that was only good for this moment in time to ensure that the next time we check, we have a clean slate)"""
        
        self._current_candles = None

    ###########################################################
    ######## Methods for order flow  ##########################
    ###########################################################
    def set_prices(self, current_price_info:pd.Series):
        """Sets the current price information for the simulation. This should be called once per tick.
        
        :param current_price_info: A pandas series containing the current price information for the simulation
        """
        self._current_candles = current_price_info
        
    def _get_execution_price_for_market_order(self, order:OrderBase) -> float:
        """Returns the execution price for an order based on the current candle, the slippage model, and the spread. 
        Does not update the variables of the order or the account. 
        
        :param order: The order to get the execution price for
        """

        # this storage attr for the execution prices gets wiped every time the exchange receives new data, so it should be none
        ticker = order.ticker
        
        # switch on the order type
        if isinstance(order, MarketOrder):
            candle_open = self._current_candles[ticker+'_open']
            candle_close = self._current_candles[ticker+'_close']
            candle_high = self._current_candles[ticker+'_high']
            candle_low = self._current_candles[ticker+'_low']

            # get the current price data for the relevent instrument
            if self.market_order_slippage_model == SlippageModelsMarketOrders.NEXT_OPEN:
                price = candle_open
            elif self.market_order_slippage_model == SlippageModelsMarketOrders.NEXT_CLOSE:
                price = candle_close
            elif self.market_order_slippage_model == SlippageModelsMarketOrders.RANDOM_IN_NEXT_CANDLE:
                price = np.random.uniform(candle_high, candle_low)
            elif self.market_order_slippage_model == SlippageModelsMarketOrders.RANDOM_AROUND_NEXT_CANDLE_OPEN:
                delta_per_min = abs(candle_high - candle_low)/(self._expected_resolution.get_as_minutes())
                price = np.random.normal(candle_open, delta_per_min/4)
                    
            # account for the spread
            if order.side == OrderSide.BUY:
                price = price * (1 + self.spread_percent/(100))
            elif order.side == OrderSide.SELL:
                price = price * (1 - self.spread_percent/(100))
            return round(price, 2)
        else:
            raise Exception(f'Unknown order type: {type(order)}')

    def _execute_buy_order(self, account:SimulatedAccount, execution_price:float, order:OrderBase):
        commission = execution_price * self.commission_percent/100
        account.cash -= commission
        if order.dollar_amount_to_use is not None:
            order.dollar_amount_to_use -= commission
            quantity = order.dollar_amount_to_use / execution_price
            total_trade_value = order.dollar_amount_to_use
        else:
            quantity = account.cash / execution_price
            total_trade_value = account.cash
            
        # validate the account has enough cash to execute the order
        if account.cash < total_trade_value:
            account.logger.warning('Not enough cash to execute the pending {} order. Order will be cancelled. Current cash: ${}, total trade val: ${}'.format(
                order.side.name,
                account.cash,
                total_trade_value
            ))
            account.logger.warning('Order: {}'.format(order))
            account.logger.warning(f'Pending orders: {account._pending_orders}')
            account.logger.warning(f'Holdings: {account._equities}')
            return
        
        # Do the trade
        account.cash -= total_trade_value
        if order.ticker in account._equities:
            account._equities[order.ticker] += quantity
        else:
            account._equities[order.ticker] = quantity
        account.logger.info("Bought {:.2f} shares of {} at ${:.2f} totaling ${:.2f} with {}. Open: ${:.2f}, Close: ${:.2f} at {}".format(
            quantity,
            order.ticker,
            execution_price,
            total_trade_value,
            type(order).__name__,
            self._current_candles[order.ticker+'_open'],
            self._current_candles[order.ticker+'_close'],
            self._current_candles.name
        ))
        account.logger.debug(f'Updated holdings: {account._equities}')
        
        order.shares = quantity
        order.commision = commission
        
    def _execute_sell_order(self, account:SimulatedAccount, execution_price:float, order:OrderBase):
    
        quantity = account._equities[order.ticker]
        total_trade_value = execution_price * quantity

        account.cash += total_trade_value
        account._equities[order.ticker] -= quantity
        account.logger.info("  Sold {:.2f} shares of {} at ${:.2f} totaling ${:.2f} with {}. Open: ${:.2f}, Close: ${:.2f} at {}".format(
            quantity,
            order.ticker,
            execution_price,
            total_trade_value,
            type(order).__name__,
            self._current_candles[order.ticker+'_open'],
            self._current_candles[order.ticker+'_close'],
            self._current_candles.name
        ))
        
        order.shares = quantity
        
    def _execute_pending_order(self, account:SimulatedAccount, execution_price:float, order_index:int):
        """Executes the pending trade at the current price depending on the slippage model. 
        The order of operations should be structured such that this gets called the tick after the order is placed.
        
        Returns the executed order, or None if the order was not executed.
        
        :param account: The account to execute the order for
        :param execution_price: The execution price for the order
        :param order_index: The index of the order to execute
        """

        self.logger.debug(f'Executing pending {type(account._pending_orders[order_index]).__name__} idx {order_index} at ${execution_price}')
        order = account._pending_orders.pop(order_index)
        # figure out how much the account is able to buy/sell
        if order.side == OrderSide.BUY:
            self._execute_buy_order(account, execution_price, order)
        elif order.side == OrderSide.SELL:
            self._execute_sell_order(account, execution_price, order)
        else: raise ValueError('Invalid order side: {}'.format(order.side))

        # update the account history, fill in meta data
        order.execution_price = execution_price

        # check the slippage mode to set the timestamp --> below broke some plotting
        # if self.market_order_slippage_model == SlippageModelsMarketOrders.NEXT_OPEN:
        #     order.execution_timestamp = self._current_candles.name - datetime.timedelta(minutes=self._expected_resolution.get_as_minutes())
        # elif self.market_order_slippage_model == SlippageModelsMarketOrders.NEXT_CLOSE:
        #     order.execution_timestamp = self._current_candles.name

        order.execution_timestamp = self._current_candles.name
        account.append_order_to_account_history(order)

    def process_orders_for_account(self, account:SimulatedAccount):
        """Checks all pending orders for execution and executes them if some conditions are met.
        Uses the candle high and low values to check if the order should be executed on limit, stop orders etc.
        
        :param account: The account to process the orders for
        """

        # check if this is a valid time to trade and if there is an order
        ts:pd.Timestamp = self._current_candles.name
        if not self._check_if_market_is_open(ts): return
        while len(account._pending_orders) > 0:
            # loop over each order and check if it should be executed. If so, break as execution will change the list
            for idx, order in enumerate(account.get_pending_orders()):
                # handle cases for order types that may not have all ticker/price data properties first
                if isinstance(order, CancelAllOrders):
                    account.cancel_all_pending_orders()
                    break
                elif isinstance(order, CancelOrder):
                    account.cancel_order(order.order_to_cancel)
                    self.logger.debug('Cancelling pending order...')
                    break
                
                high = self._current_candles[order.ticker+'_high']
                low = self._current_candles[order.ticker+'_low']
                
                # now depending on the order type check to execute order, and/or update with current price info
                if isinstance(order, MarketOrder):
                    ex_price = self._get_execution_price_for_market_order(order)
                    self._execute_pending_order(account, ex_price, idx)
                    break
                
                elif isinstance(order, TrailingStopOrder):
                    if order.side == OrderSide.BUY:
                        order.update(high)
                        if order.stop_triggered:
                            price = order.stop_price + self.limit_order_slippage
                            self._execute_pending_order(account, price, idx)
                            break
                    elif order.side == OrderSide.SELL:
                            order.update(low)
                            if order.stop_triggered:
                                price = order.stop_price - self.limit_order_slippage
                                self._execute_pending_order(account, price, idx)
                                break
                
                elif isinstance(order, LimitOrder):
                    # check the price and see if this order should get filled
                    if order.side == OrderSide.BUY and low <= order.limit_price:
                        price = order.limit_price + self.limit_order_slippage
                        self._execute_pending_order(account, price, idx)
                        break
                    elif order.side == OrderSide.SELL and high >= order.limit_price:
                        price = order.limit_price - self.limit_order_slippage
                        self._execute_pending_order(account, price, idx)
                        break
                
                elif isinstance(order, StopOrder):
                    # check the price and see if this order should get filled
                    if order.side == OrderSide.BUY and high >= order.stop_price:
                        price = order.stop_price + self.limit_order_slippage
                        self._execute_pending_order(account, price, idx)
                        break
                    elif order.side == OrderSide.SELL and low <= order.stop_price:
                        price = order.stop_price - self.limit_order_slippage
                        self._execute_pending_order(account, price, idx)
                        break
                
                elif isinstance(order, StopLimitOrder):
                    # check the price and see if this order should get filled
                    if order.side == OrderSide.BUY:
                        if not order.stop_triggered:
                            order.update(high)
                        else:
                            order.update(low)
                        if order.limit_triggered:
                            price = order.limit_price + self.limit_order_slippage
                            self._execute_pending_order(account, price, idx)
                            break
                    elif order.side == OrderSide.SELL:
                        if not order.stop_triggered:
                            order.update(low)
                        else:
                            order.update(high)
                        if order.limit_triggered:
                            price = order.limit_price - self.limit_order_slippage
                            self._execute_pending_order(account, price, idx)
                            break
                
                else:
                    raise ValueError('Invalid order type: {}'.format(type(order)))
                
                # if we've looped over the entire list and not executed anything, we've processed all orders for this tick
                if idx == len(account.get_pending_orders()) - 1: return   
                