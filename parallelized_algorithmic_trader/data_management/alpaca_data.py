import pandas as pd

from parallelized_algorithmic_trader.util import get_logger, DateRange
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution

# market data
# StockHistoricalDataClient, CryptoHistoricalDataClient, CryptoDataStream, StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


logger = get_logger(__name__)
import logging
logger.setLevel(logging.INFO)


def temporal_resolution_to_alpaca_timeframe(resolution: TemporalResolution) -> TimeFrame:
    if resolution == TemporalResolution.DAY:
        return TimeFrame.Day
    elif resolution == TemporalResolution.HOUR:
        return TimeFrame.Hour
    elif resolution == TemporalResolution.MINUTE:
        return TimeFrame.Minute
    else:
        raise ValueError(f"Unsupported temporal resolution: {resolution}")

    
def _get_single_candle_data_batch(
    API_KEY:str, 
    SECRET_KEY:str,
    ticker:str, 
    time_range:DateRange,
    resolution:TemporalResolution) -> pd.DataFrame:
    
    """Get candle data from Alpaca API with a single API call
    
    Get aggregate bars for a ticker over a given date range in custom time window sizes.
    Returns a dict with the ticker as the keys and the dataframes as the values.

    Parameters:
        API_KEY: The API key to use for the request.
        SECRET_KEY: The secret key to use for the request.
        ticker: The ticker symbol.
        time_range: The time range to get data for.
        resolution: The resolution of the data (time that one candle represents)
    Returns:
        pandas dataframe with the following columns: open, high, low, close, volume and datetime index
    """
    
    logger.debug(f'Making API call to Alpaca for {ticker}')
    
    time_frame = temporal_resolution_to_alpaca_timeframe(resolution)
    
    logger.info(f'Fetching {ticker} data from Alpaca from {time_range} for resolution {resolution.name}')
    
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    request_params = StockBarsRequest(
                            symbol_or_symbols=[ticker],
                            timeframe=time_frame,
                            start=time_range.start,
                            end=time_range.end,
                            )

    candles = client.get_stock_bars(request_params).df
    candles.drop(columns=['trade_count', 'vwap'], inplace=True)
    
    column_names = {
        'open': ticker + '_open',
        'close': ticker + '_close',
        'high': ticker + '_high',
        'low': ticker + '_low',
        'volume': ticker + '_volume'
    }
    
    # rename the columns
    candles.rename(columns=column_names, inplace=True)
    # get just the timestamps from the index
    timestamps = candles.index.get_level_values('timestamp')
    # loop over the DatetimeIndex and alter the pd.TimeStamp to be able to be compared to offset-naive datetimes
    naive_timestamps = []
    for i, aware_ts in enumerate(timestamps):
        aware_ts:pd.Timestamp
        naive_ts = aware_ts.tz_localize(None)
        naive_timestamps.append(naive_ts)

    # convert the column (it's a string) to datetime type
    # datetime_series = pd.to_datetime(df['date_of_birth'])

    datetime_index = pd.DatetimeIndex(naive_timestamps)
    datetime_index.name = 'timestamp'
    
    # now remove the current index and insert the timestamp column. Indexes resist mutation which is why this is a little roundabout
    candles.reset_index(drop=True, inplace=True)
    candles.set_index(datetime_index, inplace=True)
    
    logger.debug(f'Got {candles.shape[0]} candles for {ticker}')
    logger.debug(f'First candles: \n{candles.head()}')
    
    if candles.empty:
        logger.warning(f'No candles returned for {ticker}')
        return None
    return candles

