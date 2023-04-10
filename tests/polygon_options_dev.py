import polygon
import datetime
import logging
from enum import Enum

from parallelized_algorithmic_trader.data_management.polygon_io import get_polygon_key
from parallelized_algorithmic_trader.data_management.data import CandleData, TemporalResolution, sanitize_dataframe


logger = get_logger(__name__)


class OptionsSide(Enum):
    CALL = 'call'
    PUT = 'put'


def get_options_data(
    ticker:str, 
    strike_price:float, 
    strike_date:datetime.date, 
    start_date:datetime.date, 
    end_date:datetime.date, 
    temporal_resolution:TemporalResolution):
    """_summary_

    Args:
        ticker (str): _description_
        strike_price (float): _description_
        strike_date (datetime.date): _description_
        start_date (datetime.date): _description_
        end_date (datetime.date): _description_
        temporal_resolution (TemporalResolution): _description_
    """
    
    options_client = polygon.OptionsClient(get_polygon_key())  # for usual sync client

    # options symbols: OSI is a 21 character string that defines the option contract.
    # https://www.optionstaxguy.com/option-symbols-osi
    # symbol last 2 digits of year, month, day, C/P, strike price, strike price decimal (3 digits)
    
    # example of how to make a symbol
    # symbol1 = polygon.build_option_symbol('AMD', datetime.date(2022, 6, 28), 'call', 546.56)
    
    #  For example, if timespan = ‘minute’ and multiplier = ‘5’ then 5-minute bars will be returned
    symbol = polygon.build_option_symbol(ticker, strike_date, 'call', strike_price)
    logger.debug(f'Built options symbol: {symbol}.')
    agg_bars = options_client.get_aggregate_bars(
        symbol,
        datetime.date(2022, 6, 1),
        datetime.date(2022, 7, 1),
        limit=5000,
        timespan='day',
        multiplier=1
        )
    logger.debug(f'got data.')
    print(agg_bars)
    

get_options_data(
    'AMD', 
    datetime.date(2022, 7, 1), 
    datetime.date(2022, 6, 1), 
    datetime.date(2022, 7, 1), 
    TemporalResolution.MINUTE)
    
    