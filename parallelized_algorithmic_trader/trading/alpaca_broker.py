from typing import List
import datetime
import pandas as pd
import os

from simulated_broker import Brokerage
from parallelized_algorithmic_trader.orders import *

# trading
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
from alpaca.trading.models import Position
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data import StockDataStream


class RealBrokerage(Brokerage):
    def __init__(self, class_name:str, log_level=None):
        
        name = __name__ + '.' + class_name
        if log_level is None: super().__init__(name)
        else: super().__init__(log_level=log_level)

        self._max_lookback_period = datetime.timedelta(days=14)

    def set_max_lookback_period(self, max_lookback_period_days:int):
        """Sets the maximum number of days to look back when getting historical data.
        
        :param max_lookback_period_days: The maximum number of days to look back when getting historical data."""
        self._max_lookback_period = datetime.timedelta(days=max_lookback_period_days)


class AlpacaBroker(RealBrokerage):

    def __init__(self, 
                 API_KEY:str|None=None,
                 SECRET_KEY:str|None=None,
                 ):
        
        super().__init__(class_name='AlpacaBroker', log_level=10)
        
        if not API_KEY:
            self.logger.warning(f'API_KEY not provided. Using environment variable.')
            API_KEY = os.environ.get('ALPACA')
            SECRET_KEY = os.environ.get('ALPACA_SECRET')
        
        self._trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self._account_info = self._trading_client.get_account()
        self._data_stream = StockDataStream(API_KEY, SECRET_KEY)
        self._current_positions:List[Position]
        
    def update_current_positions(self):
        # Get all open positions and print each of them
        positions = self._trading_client.get_all_positions()
        for position in positions:
            for property_name, value in position:
                self.logger.debug(f"\"{property_name}\": {value}")
                
        """
        "asset_id": 64bbff51-59d6-4b3c-9351-13ad85e3c752
        "symbol": BTCUSD
        "exchange": FTXU
        "asset_class": crypto
        "avg_entry_price": 20983
        "qty": 0.9975
        "side": long
        "market_value": 20928.5475
        "cost_basis": 20930.5425
        "unrealized_pl": -1.995
        "unrealized_plpc": -0.0000953152552066
        "unrealized_intraday_pl": -1.995
        "unrealized_intraday_plpc": -0.0000953152552066
        "current_price": 20981
        "lastday_price": 19344
        "change_today": 0.084625723738627
        """
    
    def get_account_number(self) -> str:
        return self._account_info['account_number']
    
    def print_account_information(self):
        """Pulls the latest info from the account and prints it to the console."""
        self._account_info = self._trading_client.get_account()
        # Getting account information and printing it
        for property_name, value in self._account_info:
            self.logger.info(f"\"{property_name}\": {value}")

    def place_order(self, OrderBase):
        
        if isinstance(OrderBase, MarketOrder):
            # Setting parameters for our buy order
            market_order_data = MarketOrderRequest(
                        symbol="BTC/USD",
                        qty=1,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC
                    )
            
            # Submitting the order and then printing the returned object
            market_order = self._trading_client.submit_order(market_order_data)
            for property_name, value in market_order:
                print(f"\"{property_name}\": {value}")
        else:
            raise NotImplementedError(f'Order type {type(OrderBase)} not implemented.')


    def get_current_candle_data(self) -> pd.DataFrame:

        # keys are required for live data
        crypto_stream = CryptoDataStream("api-key", "secret-key")

        # Subscribing to Real-Time Quote Data
        from alpaca.data import CryptoDataStream, StockDataStream

        # keys are required for live data
        crypto_stream = CryptoDataStream("api-key", "secret-key")

        # keys required
        stock_stream = StockDataStream("api-key", "secret-key")

        # https://alpaca.markets/docs/python-sdk/trading.html