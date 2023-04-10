import os
from datetime import datetime, timedelta
import time
import pandas as pd
from typing import List, Dict
from polygon import RESTClient

from parallelized_algorithmic_trader.data_management.data import *
from parallelized_algorithmic_trader.util import DATA_DIRECTORY, get_logger, DateRange
import parallelized_algorithmic_trader.data_management.data_utils as data_utils


logger = get_logger(__name__)
import logging
logger.setLevel(logging.INFO)
    
    
def get_single_candle_data_batch_from_polygon(
    API_KEY:str, 
    ticker:str, 
    time_range:DateRange,
    resolution:data_utils.TemporalResolution, 
    adjusted=True) -> pd.DataFrame:
    """Get candle data from Polygon API with a single API call
    
    Get aggregate bars for a ticker over a given date range in custom time window sizes.
    Returns a dict with the tickers as the keys and the dataframes as the values.

    Parameters:
        API_KEY: The API key to use for the request.
        ticker: The ticker symbol.
        time_range: The time range to get data for.
        resolution: The resolution of the data (time that one candle represents)
        adjusted: Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
    Returns:
        pandas dataframe with the following columns: open, high, low, close, volume and datetime index
    """
    logger.debug(f'Making API call to polygon.io for {ticker}')

    # convert the resolution to a string
    candle_size = resolution.name.lower()
    if not candle_size in ('minute', 'hour', 'day'): 
        raise ValueError(f'Invalid resolution: {resolution}. Must be minute, hour, or day with polygon.io')

    client = RESTClient(api_key=API_KEY)
    
    start, end = [str(dt.date()) for dt in (time_range.start, time_range.end)]
    logger.info(f'Fetching {ticker} data from polygon.io from {start} to {end} for resolution {candle_size.upper()}')
    aggs = client.get_aggs(ticker, 1, candle_size, start, end, adjusted=adjusted, sort='asc', limit=50000)
    logger.debug(f'Got {len(aggs)} candles for {ticker}')
    logger.debug(f'First agg: {aggs[0]}')
    
    # each agg has the following information: 
    """
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    vwap: Optional[float] = None
    timestamp: Optional[int] = None
    transactions: Optional[int] = None
    otc: Optional[bool] = None
    """
    
    # build the dataframe for this tickertime_range.start.
    ticker_df = pd.DataFrame(columns=[ticker + '_open', ticker + '_high', ticker + '_low', ticker + '_close', ticker + '_volume', 'timestamp'])
    for agg in aggs:
        candle = pd.DataFrame({
            ticker + '_open': [agg.open], 
            ticker + '_high': [agg.high], 
            ticker + '_low': [agg.low], 
            ticker + '_close': [agg.close], 
            ticker + '_volume': [agg.volume], 
            'timestamp': [agg.timestamp]})
        ticker_df = pd.concat([ticker_df, candle], ignore_index=True)
    # now convert the timestamp to a datetime
    ticker_df['timestamp'] = pd.to_datetime(ticker_df['timestamp'], unit='ms')
    return ticker_df


# # global variables used to track API calls and make sure we don't exceed the rate limit.
# N_API_CALLS_SINCE_START = 0
# API_CALLS_START_TIME:float = None


# def get_aggregated_candle_data_for_ticker_from_polygon(
#     API_KEY:str,
#     ticker:str, 
#     time_range:DateRange,
#     resolution:data_utils.TemporalResolution, 
#     adjusted=True, 
#     max_api_calls_per_min:int=5) -> pd.DataFrame:
#     """Polygon limits on the amount of data returned per call. This function repeatedly calls the API looping over the full time range
#     in order to get all of the data, which is then sorted, serialized, and stored in a file.

#     Parameters:
#         ticker: The ticker symbols.
#         time_range: The time range to get data for.
#         resolution: The resolution of the data.
#         adjusted: Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
#         max_api_calls_per_min: The maximum number of API calls per minute. Polygon limits this to 5 for the basic package. All other pricing plans are unlimited
        
#     Returns:
#         pd.DataFrame object
#     """

#     global N_API_CALLS_SINCE_START
#     global API_CALLS_START_TIME

#     logger.debug(f'Aggregating candle data from {time_range.start.date()} to {time_range.end.date()} for ticker {ticker} with resolution {resolution.name}')
    
#     s = time_range.start    # variable representing the start date for fetching data. Each loop this gets moved to the end of the retrieved data
#     df = pd.DataFrame()

#     if API_CALLS_START_TIME is None:
#         API_CALLS_START_TIME = time.monotonic()

#     while s < time_range.end - timedelta(days=2):
#         # fetch the data from polygon
#         tm_rnge = DateRange(s, time_range.end)
#         small_df = get_single_candle_data_batch_from_polygon(API_KEY, ticker, tm_rnge, resolution, adjusted)
#         N_API_CALLS_SINCE_START += 1

#         # append it to the dataframe
#         df = pd.concat([df, small_df], ignore_index=True)

#         # update the start dates
#         s = small_df['timestamp'].iloc[-2]

#         # sleep if we've hit the max number of API calls to avoid API call limit
#         if not max_api_calls_per_min is None and N_API_CALLS_SINCE_START >= max_api_calls_per_min:
#             elapsed = time.monotonic() - API_CALLS_START_TIME
#             if 1 or elapsed < 60:
#                 sleep_time = 61
#                 logger.info('Waiting {:.1f} seconds to make the next API call...'.format(sleep_time))
#                 time.sleep(sleep_time)
            
#             N_API_CALLS_SINCE_START = 0
#             API_CALLS_START_TIME = time.monotonic()

#     # remove duplicates in the timestamp column
#     return data_utils.sanitize_dataframe(df)


# def get_candle_data_for_ticker(
#     API_KEY:str,
#     ticker:str, 
#     time_range:DateRange, 
#     resolution:data_utils.TemporalResolution,
#     adjusted=True) -> pd.DataFrame:
#     """Checks the disk to see if the data for ticker is already on the harddrive. If
#     not, it will fetch the data from polygon.io, store it locally and return it."""

#     logger.info(f'Checking for local data for ticker {ticker} from {time_range.start.date()} to {time_range.end.date()} with resolution {resolution.name}')

#     # check to see if a file already exists for this ticker and resolution
#     file_name:str = build_data_file_name(ticker, resolution)
#     file_path:str = os.path.join(DATA_DIRECTORY, file_name)
#     df:pd.DataFrame = pd.DataFrame()

#     # check if the file exists and get the data if it does
#     if os.path.exists(file_path):
#         logger.debug(f'Found a file for {ticker} at {file_path}')
#         df:pd.DataFrame = pd.read_pickle(file_path)
#         file_data_time_range = DateRange(df.index[0], df.index[-1])
#         logger.debug(f'File data start: {file_data_time_range.start}, end: {file_data_time_range.end}')
        
#         logger.debug(f'Checking to see if the data covers the desired date range')
#         file_covers_range = data_utils.check_if_data_covers_timerange(df, time_range)
#         if file_covers_range:
#             logger.info(f'File data {file_data_time_range} covers the desired date range {time_range}')
#             return df
#         else:
#             logger.info(f'File data {file_data_time_range} does not cover the desired date range {time_range}')
            
#         # ensure that there are no gaps in the data but also only pull the range that we need to 
#         if time_range.end > file_data_time_range.end:
#             needed_time_range = DateRange(
#                 file_data_time_range.end - timedelta(days=1),   
#                 time_range.end
#             )
#         else: 
#             needed_time_range = DateRange(
#                 time_range.start,
#                 file_data_time_range.start + timedelta(days=1)
#             )
#     else:
#         logger.debug(f"No local files found for ticker {ticker}")
#         needed_time_range = time_range

#     # get the data from polygon.io
#     new_data = get_aggregated_candle_data_for_ticker_from_polygon(
#         API_KEY, 
#         ticker, 
#         needed_time_range,
#         resolution, 
#         adjusted=adjusted)
    
#     # append the new data to the existing data for this ticker that may have been loaded from file
#     # if new_data is not empty
#     if not new_data.empty:
#         logger.debug(f"New data dates: {new_data.index[0]} to {new_data.index[-1]}")
#         df = pd.concat([df, new_data])
#         # sort by index
#         df.sort_index(inplace=True)
#         # logger.debug(f'New dataframe dates: {df["timestamp"].iloc[0]} to {df["timestamp"].iloc[-1]}')
#         df = data_utils.sanitize_dataframe(df)
#         logger.debug(f'Dates after cleaning: {df.index[0]} to {df.index[-1]}')
#     else:
#         logger.warning(f'Warning: retrived no data from polygon')
#         if df.empty:
#             logger.error(f'Error: no data found for ticker {ticker}')
#             raise Exception(f'No data found locally or through polygon for ticker {ticker}')
        
#     df.to_pickle(file_path)
#     # logger.warning(f'Skipping save..')
#     logger.debug(f'saved to file: {file_path}')
#     logger.info(f'Successfully loaded data for ticker {ticker}.')
    
#     # filter for the desired date range
#     df = df[df.index >= time_range.start]
#     df = df[df.index <= time_range.end]
#     return df


# def get_candle_data(
#     API_KEY:str, 
#     tickers:List[str], 
#     start_date:datetime, 
#     end_date:datetime, 
#     resolution:data_utils.TemporalResolution, 
#     adjusted=True) -> CandleData:
#     """Fetches data from either disk or polygon API, formatted in dataframe for each ticker. Returns formal CandleData object.
    
#     :param API_KEY: API key for polygon.io
#     :param tickers: list of tickers to fetch data for
#     :param start_date: start date for data
#     :param end_date: end date for data as datetime
#     :param resolution: time resolution of data 
#     :param adjusted: whether to have polygon.io adjust the data for splits and dividends
#     """

#     # strip out the time information
#     logger.info(f'Fetching historical data for tickers {tickers} from {start_date.date()} to {end_date.date()} with resolution {resolution.name}')
    
#     time_range = DateRange(
#         data_utils.ceiling_to_subsequent_business_date(start_date), 
#         data_utils.floor_to_preceding_business_day(end_date)
#         )
    
#     df = pd.DataFrame()
#     for t in tickers:
#         ticker_df = get_candle_data_for_ticker(API_KEY, t, time_range, resolution, adjusted)
#         # append the new data to the existing data for this ticker that may have been loaded from file
#         df = pd.concat([df, ticker_df], axis=1)
    
#     df['Source'] = ['polygonio' for i in range(len(df))]
#     return CandleData(df, tickers, resolution)
    

