
from typing import List
from dataclasses import dataclass, field, asdict
from td.client import TDClient, VALID_CHART_VALUES
import pandas as pd
import os

from parallelized_algorithmic_trader.data_management.market_data import TemporalResolution, CandleData
from parallelized_algorithmic_trader.broker import RealBrokerage
from parallelized_algorithmic_trader.orders import *


CONSUMER_KEY = os.environ['TDAMERITRADE_CONSUMER_KEY']
REDIRECT_URI = os.environ['TDAMERITRADE_REDIRECT_URI']


CREDENTIALS_PATH = os.path.join('./','parallelized_algorithmic_trader/td_ameritrade_wrapper/config/credentials.json')
ACCOUNT_NUMBER_PATH = os.path.join('./','parallelized_algorithmic_trader/td_ameritrade_wrapper/config/account_number.json')


@dataclass
class PriceHistoryConfig:
    """Config for the price history data from TDAmeritrade API
    Example: For a 2 day / 1 min chart, the values would be:
        period: 2
        periodType: day
        frequency: 1
        frequencyType: minute
    """
    symbol:str
    period_type:str = field(default='day')          # valid are day, month, year, or ytd.

    period:int = field(default=10)                   
    """Valid periods by periodType (defaults marked with an asterisk): 
        day: 1, 2, 3, 4, 5, 10*;
        month: 1*, 2, 3, 6
        year: 1*, 2, 3, 5, 10, 15, 20
        ytd: 1*"""

    frequency_type:str = field(default='minute')
    """
    The type of frequency with which a new candle is formed.
    Valid frequencyTypes by periodType (defaults marked with an asterisk):
        day: minute*
        month: daily, weekly*
        year: daily, weekly, monthly*
        ytd: daily, weekly*"""

    frequency:int= field(default=1)
    """The number of the frequencyType to be included in each candle.
    Valid frequencies by frequencyType (defaults marked with an asterisk):
        minute: 1*, 5, 10, 15, 30
        daily: 1*
        weekly: 1*
        monthly: 1*"""

    def get_resolution_from_period(self) -> TemporalResolution:
        """Convert the period and frequency to a backtrader TemporalResolution"""
        return TemporalResolution[self.frequency_type.upper()]
    
    @staticmethod
    def convert_resolution_to_period(resolution:TemporalResolution):
        """Convert the backtrader TemporalResolution to a period and frequency"""
        unsupported_periods = (TemporalResolution.HOUR, TemporalResolution.MINUTE, TemporalResolution.WEEK)
        supported_periods = (TemporalResolution.DAY, TemporalResolution.MONTH, TemporalResolution.YEAR)
        supported_strs = [p.value.lower() for p in supported_periods]
        if resolution in unsupported_periods:
            raise ValueError(f"TDAmeritrade doesn't support {resolution.value} data for period. Try {supported_strs}.")
        return resolution.value.lower()
        
    @staticmethod
    def convert_resolution_to_frequency(resolution:TemporalResolution):
        """Convert the backtrader TemporalResolution to a period and frequency"""
        unsupported_frequencies = (TemporalResolution.HOUR,)
        if resolution in unsupported_frequencies:
            raise ValueError(f"TDAmeritrade doesn't support {resolution.value} data for frequency. Try day, week, or month.")
        elif resolution == TemporalResolution.MINUTE:
            return 'minute'
        elif resolution == TemporalResolution.DAY:
            return 'daily'
        elif resolution == TemporalResolution.WEEK:
            return 'weekly'
        elif resolution == TemporalResolution.MONTH:
            return 'monthly'
        else:
            raise ValueError(f'Unknown: {resolution}')


class TDAmeritradeBroker(RealBrokerage):
    """A wrapper for the TDAmeritrade API matching the module standards for representing exchanges."""
    
    def __init__(self, resolution:TemporalResolution, tickers:List[str]=[]):

        super().__init__('TDAmeritrade')

        self.add_ticker(tickers)
        # connect to the TDAmeritrade API
        self.set_expected_resolution(resolution)
        self.connect()
        self.logger.info('TDAmeritrade broker initialized')
        self._account_number:str = os.environ["TDAMERITRADE_ACCOUNTNUMBER"]

    def connect(self):
        """Connect to the TDAmeritrade API"""

        # Get a TDClient object that can be used to make requests to the TD Ameritrade API
        
        # Create a new session, credentials path is required.
        self.logger.debug(f'Connecting to TDAmeritrade API. ')
        TDSession = TDClient(
            # client_id='brucejiverson',
            client_id=CONSUMER_KEY,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH,
        )

        # Login to the session
        TDSession.login()
        self._session:TDClient = TDSession

    def get_candle_data(self, config:PriceHistoryConfig) -> CandleData:
        """CandleList:
        {
        "candles": [
            {
            "close": 0,
            "datetime": 0,
            "high": 0,
            "low": 0,
            "open": 0,
            "volume": 0
            }
        ],
        "empty": false,
        "symbol": "string"
        }
        Example:
        {'candles': [
            {'open': 414.97, 'high': 414.97, 'low': 414.7, 'close': 414.7, 'volume': 20993, 'datetime': 1654513200000}, 
            {'open': 414.73, 'high': 414.82, 'low': 414.73, 'close': 414.8, 'volume': 2104, 'datetime': 1654513260000}, 
            {'open': 414.75, 'high': 414.77, 'low': 414.75, 'close': 414.77, 'volume': 700, 'datetime': 1654513320000}...
        """
        self.logger.info(f'Getting candle data for {config.symbol}')
        self.logger.debug(f'Config: {asdict(config)}')
        
        try:
            raw_candles = self._session.get_price_history(**asdict(config))
        except Exception as e:
            self.logger.error(f'Failed to get candle data for {config.symbol}. Likely issue with period and frequency. Valid: {VALID_CHART_VALUES}')
            raise e
        
        if raw_candles is None or len(raw_candles) == 0:
            self.logger.debug(raw_candles)
            raise ValueError('DataFrame is empty or None. Some invalid parameter must have been given or issue wit TD account configuration.')
        candles = pd.DataFrame(raw_candles['candles'])
        
        # do some formatting 
        candles['datetime'] = pd.to_datetime(candles['datetime'], unit='ms')
        # rename the datetime column to timestamp
        new_col_names = {
            'datetime':'timestamp', 
            'open': config.symbol + '_open', 
            'close': config.symbol + '_close', 
            'high': config.symbol + '_high', 
            'low': config.symbol + '_low', 
            'volume': config.symbol + '_volume'}
        
        candles.rename(columns=new_col_names, inplace=True)
        candles.set_index(['timestamp'], drop=True, inplace=True)
        candles['Source'] = ['TDAmeritrade' for _ in range(len(candles))]
        
        self.logger.debug(f'Successfully retrieved candle data for {config.symbol}')
        return CandleData(
            candles, 
            [config.symbol], 
            config.get_resolution_from_period()
            )

    def get_recent_price_history(self, n_periods=10) -> CandleData:
        """Gets candle data for the recent past.
        
        :param n_periods: The number of periods to get data for.
        """
        
        price_history_config = PriceHistoryConfig(
            symbol=list(self._tickers)[0],
            frequency_type=PriceHistoryConfig.convert_resolution_to_frequency(self._expected_resolution),
            period=n_periods
            )
        return self.get_candle_data(price_history_config)

    def submit_order(self, order:OrderBase):
        """Submits an order to the TDAmeritrade API
        
        :param order: The order to submit.
        """
        
        otype = type(order).__name__.replace('Order', '').upper()

        """
        {
            "session": "'NORMAL' or 'AM' or 'PM' or 'SEAMLESS'",
            "duration": "'DAY' or 'GOOD_TILL_CANCEL' or 'FILL_OR_KILL'",
            "orderType": "'MARKET' or 'LIMIT' or 'STOP' or 'STOP_LIMIT' or 'TRAILING_STOP' or 'MARKET_ON_CLOSE' or 'EXERCISE' or 'TRAILING_STOP_LIMIT' or 'NET_DEBIT' or 'NET_CREDIT' or 'NET_ZERO'",
            "cancelTime": {
                "date": "string",
                "shortFormat": false
            },
            "complexOrderStrategyType": "'NONE' or 'COVERED' or 'VERTICAL' or 'BACK_RATIO' or 'CALENDAR' or 'DIAGONAL' or 'STRADDLE' or 'STRANGLE' or 'COLLAR_SYNTHETIC' or 'BUTTERFLY' or 'CONDOR' or 'IRON_CONDOR' or 'VERTICAL_ROLL' or 'COLLAR_WITH_STOCK' or 'DOUBLE_DIAGONAL' or 'UNBALANCED_BUTTERFLY' or 'UNBALANCED_CONDOR' or 'UNBALANCED_IRON_CONDOR' or 'UNBALANCED_VERTICAL_ROLL' or 'CUSTOM'",
            "quantity": 0,
            "filledQuantity": 0,
            "remainingQuantity": 0,
            "requestedDestination": "'INET' or 'ECN_ARCA' or 'CBOE' or 'AMEX' or 'PHLX' or 'ISE' or 'BOX' or 'NYSE' or 'NASDAQ' or 'BATS' or 'C2' or 'AUTO'",
            "destinationLinkName": "string",
            "releaseTime": "string",
            "stopPrice": 0,
            "stopPriceLinkBasis": "'MANUAL' or 'BASE' or 'TRIGGER' or 'LAST' or 'BID' or 'ASK' or 'ASK_BID' or 'MARK' or 'AVERAGE'",
            "stopPriceLinkType": "'VALUE' or 'PERCENT' or 'TICK'",
            "stopPriceOffset": 0,
            "stopType": "'STANDARD' or 'BID' or 'ASK' or 'LAST' or 'MARK'",
            "priceLinkBasis": "'MANUAL' or 'BASE' or 'TRIGGER' or 'LAST' or 'BID' or 'ASK' or 'ASK_BID' or 'MARK' or 'AVERAGE'",
            "priceLinkType": "'VALUE' or 'PERCENT' or 'TICK'",
            "price": 0,
            "taxLotMethod": "'FIFO' or 'LIFO' or 'HIGH_COST' or 'LOW_COST' or 'AVERAGE_COST' or 'SPECIFIC_LOT'",
            "orderLegCollection": [
                {
                    "orderLegType": "'EQUITY' or 'OPTION' or 'INDEX' or 'MUTUAL_FUND' or 'CASH_EQUIVALENT' or 'FIXED_INCOME' or 'CURRENCY'",
                    "legId": 0,
                    "instrument": "The type <Instrument> has the following subclasses [Equity, FixedIncome, MutualFund, CashEquivalent, Option] descriptions are listed below\"",
                    "instruction": "'BUY' or 'SELL' or 'BUY_TO_COVER' or 'SELL_SHORT' or 'BUY_TO_OPEN' or 'BUY_TO_CLOSE' or 'SELL_TO_OPEN' or 'SELL_TO_CLOSE' or 'EXCHANGE'",
                    "positionEffect": "'OPENING' or 'CLOSING' or 'AUTOMATIC'",
                    "quantity": 0,
                    "quantityType": "'ALL_SHARES' or 'DOLLARS' or 'SHARES'"
                }
            ],
            "activationPrice": 0,
            "specialInstruction": "'ALL_OR_NONE' or 'DO_NOT_REDUCE' or 'ALL_OR_NONE_DO_NOT_REDUCE'",
            "orderStrategyType": "'SINGLE' or 'OCO' or 'TRIGGER'",
            "orderId": 0,
            "cancelable": false,
            "editable": false,
            "status": "'AWAITING_PARENT_ORDER' or 'AWAITING_CONDITION' or 'AWAITING_MANUAL_REVIEW' or 'ACCEPTED' or 'AWAITING_UR_OUT' or 'PENDING_ACTIVATION' or 'QUEUED' or 'WORKING' or 'REJECTED' or 'PENDING_CANCEL' or 'CANCELED' or 'PENDING_REPLACE' or 'REPLACED' or 'FILLED' or 'EXPIRED'",
            "enteredTime": "string",
            "closeTime": "string",
            "accountId": 0,
            "orderActivityCollection": [
                "\"The type <OrderActivity> has the following subclasses [Execution] descriptions are listed below\""
            ],
            "replacingOrderCollection": [
                {}
            ],
            "childOrderStrategies": [
                {}
            ],
            "statusDescription": "string"
        }
        """
        order_data = {
            "session": 'NORMAL',
            "duration": 'DAY',
            "orderType": otype,
            "orderStrategyType": 'SINGLE',
            "orderLegCollection": []
        }
        self._session.place_order(
            self._account_number,
            order_data,
        )

    def get_order_history(self) -> List[OrderBase]:
        pass

    def update_current_position(self):
        """Updates the current position of the account"""
        # self._current_position = self._session.get_accounts(
        #     self._account_number,
        #     fields=Account.Fields.POSITIONS,
        # ).json()
        pass



    def cancel_any_pending_orders(self):
        pass

    def get_last_price(self, ticker:str) -> float:
        """Returns the last price for a ticker"""
        pass

    def get_account_value(self) -> float:
        """Returns the current value of the account in USD"""
        pass

    def get_available_cash(self) -> float:
        """Returns the amount of cash available in the account"""
        pass

