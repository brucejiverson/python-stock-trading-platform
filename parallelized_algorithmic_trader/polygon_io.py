import os
from datetime import datetime, timedelta
import time
import pandas as pd
from typing import List, Dict
from polygon import RESTClient

from parallelized_algorithmic_trader.market_data import *
from parallelized_algorithmic_trader.util import DATA_DIRECTORY, get_logger


logger = get_logger(__name__)


def get_single_candle_data_batch_from_polygon(
    API_KEY:str, 
    ticker:str, 
    start_date:datetime, 
    end_date:datetime, 
    resolution:TemporalResolution, 
    adjusted=True) -> pd.DataFrame:
    """Get candle data from Polygon API with a single API call
    
    Get aggregate bars for a ticker over a given date range in custom time window sizes.
    Returns a dict with the tickers as the keys and the dataframes as the values.

    :param API_KEY: The API key to use for the request.
    :param ticker: The ticker symbol.
    :param start_date: The start of the aggregate time window as a datetime.
    :param end_date: The end of the aggregate time window as a datetime.
    :param resolution: The resolution of the data (time that one candle represents)
    :param adjusted: Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
    :return: pandas dataframe with the following columns: open, high, low, close, volume and datetime index
    """

    # convert the resolution to a string
    timespan = resolution.name.lower()
    if not timespan in ('minute', 'hour', 'day'): 
        raise ValueError(f'Invalid resolution: {resolution}. Must be minute hour or day with polygon.io')

    client = RESTClient(api_key=API_KEY)
    
    start, end = str(start_date.date()), str(end_date.date())
    logger.info(f'Fetching {ticker} data from polygon.io from {start} to {end} for resolution {timespan.upper()}')
    aggs = client.get_aggs(ticker, 1, timespan, start, end, adjusted=adjusted, sort='asc', limit=50000)

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
    
    # build the dataframe for this ticker
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


# global variables used to track API calls and make sure we don't exceed the rate limit.
N_API_CALLS_SINCE_START = 0
API_CALLS_START:float = time.monotonic()


def get_candle_data_for_ticker_from_polygon(
    API_KEY:str,
    ticker:str, 
    start_date:datetime, 
    end_date:datetime, 
    resolution:TemporalResolution, 
    adjusted=True, 
    max_api_calls_per_min:int=5) -> pd.DataFrame:
    """Polygon limits on the amount of data returned per call. This function repeatedly calls the API looping over the full time range
    in order to get all of the data, which is then sorted, serialized, and stored in a file.

    :param ticker: The ticker symbols.
    :param start_date: The start of the time window as a datetime.
    :param end_date: The end of the time window as a datetime.
    :param resolution: The resolution of the data.
    :param adjusted: Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
    :param max_api_calls_per_min: The maximum number of API calls per minute. Polygon limits this to 5 for the basic package. All other pricing plans are unlimited
    :return: pd.DataFrame object
    """

    logger.debug(f'Aggregating candle data from {start_date.date()} to {end_date.date()} for ticker {ticker} with resolution {resolution.name}')
    
    s = start_date
    df = pd.DataFrame()

    N_API_CALLS_SINCE_START = 0
    API_CALLS_START = time.monotonic()

    while s < end_date - timedelta(days=2):
        # fetch the data from polygon
        small_df = get_single_candle_data_batch_from_polygon(API_KEY, ticker, s, end_date, resolution, adjusted)
        N_API_CALLS_SINCE_START += 1

        # append it to the dataframe
        df = pd.concat([df, small_df], ignore_index=True)

        # update the start dates
        s = small_df['timestamp'].iloc[-1]

        # sleep if we've hit the max number of API calls
        if not max_api_calls_per_min is None and N_API_CALLS_SINCE_START >= max_api_calls_per_min:
            elapsed = time.monotonic() - API_CALLS_START
            if 1 or elapsed < 60:
                sleep_time = 61
                logger.info('Waiting {:.1f} seconds to make the next API call...'.format(sleep_time))
                time.sleep(sleep_time)
            
            N_API_CALLS_SINCE_START = 0
            API_CALLS_START = time.monotonic()

    return df


def get_candle_data_for_ticker(
    API_KEY:str,
    ticker:str, 
    start_date:datetime, 
    end_date:datetime, 
    resolution:TemporalResolution,
    adjusted=True) -> pd.DataFrame:
    """Checks the disk to see if the data for ticker is already on the harddrive. If
    not, it will fetch the data from polygon.io, store it locally and return it."""

    # strip out the time information
    input_start_date, input_end_date = datetime(start_date.year, start_date.month, start_date.day), datetime(end_date.year, end_date.month, end_date.day)    
    logger.info(f'Checking for local historical data for ticker {ticker} from {input_start_date.date()} to {input_end_date.date()} with resolution {resolution.name}')

    # check to see if a file already exists for this ticker and resolution
    file_name:str = get_data_file_name(ticker, resolution)
    file_path:str = os.path.join(DATA_DIRECTORY, file_name)
    ticker_df:pd.DataFrame = pd.DataFrame()

    # check if the file exists and get the data if it does
    if os.path.exists(file_path):
        logger.debug(F'Found a file for {ticker} at {file_path}')
        # load the existing data
        ticker_df = pd.read_pickle(file_path)
        # check to see if the data covers the desired date range.
        if type(ticker_df.index) == pd.DatetimeIndex:
            file_data_start, file_data_end = ticker_df.index[0], ticker_df.index[-1]
        else:
            file_data_start, file_data_end = ticker_df['timestamp'].iloc[0], ticker_df['timestamp'].iloc[-1]
        file_data_end += timedelta(hours = 2)
        # if the data covers the desired date range, then we're done
        logger.debug(F'Checking to see if the data covers the desired date range')
        logger.debug(F'File data start: {file_data_start}, end: {file_data_end}')
        if file_data_start <= input_start_date + timedelta(days=2) and file_data_end >= input_end_date - timedelta(days=2):   # this is fudged by 2 days on either end to account for weekends. holidays could mess this up
            logger.info(f'File data for ticker {ticker} from {file_data_start} to {file_data_end} covers the desired date range from {input_start_date} to {input_end_date}')
            return ticker_df
        else:
            # if the data does not cover the desired date range, then we need to get the data from polygon.io
            logger.info(f'File data for ticker {ticker} from {file_data_start} to {file_data_end} does not cover the desired date range from {input_start_date} to {input_end_date}')
    else:
        logger.debug(f"No local files found for ticker {ticker}")
        file_data_start, file_data_end = input_start_date, input_end_date

    # get the data from polygon.io
    # ensure that there are no gaps in the data but also only pull the range that we need to 
    s = min(input_start_date, file_data_end) - timedelta(days=1)
    e = max(input_end_date, file_data_start)# + timedelta(days=1) 

    new_data = get_candle_data_for_ticker_from_polygon(API_KEY, ticker, s, e, resolution, adjusted=adjusted)
    # append the new data to the existing data for this ticker that may have been loaded from file
    ticker_df = pd.concat([ticker_df, new_data], ignore_index=True)
    ticker_df = clean_dataframe(ticker_df)
    # save the data
    ticker_df.to_pickle(file_path)
    logger.debug(f'saved to file: {file_path}')
    logger.info(f'Successfully loaded data for ticker {ticker}, date range now from {ticker_df.index[0]} to {ticker_df.index[-1]}')
    return ticker_df


def get_candle_data(
    API_KEY:str, 
    tickers:List[str], 
    start_date:datetime, 
    end_date:datetime, 
    resolution:TemporalResolution, 
    adjusted=True) -> CandleData:
    """Fetches data from either disk or polygon API, formatted in dataframe for each ticker. Returns formal CandleData object.
    
    :param API_KEY: API key for polygon.io
    :param tickers: list of tickers to fetch data for
    :param start_date: start date for data
    :param end_date: end date for data as datetime
    :param resolution: time resolution of data 
    :param adjusted: whether to have polygon.io adjust the data for splits and dividends
    """

    # strip out the time information
    input_start_date, input_end_date = datetime(start_date.year, start_date.month, start_date.day), datetime(end_date.year, end_date.month, end_date.day)    
    logger.info(f'Fetching historical data for tickers {tickers} from {input_start_date.date()} to {input_end_date.date()} with resolution {resolution.name}')
    logger.debug(f'Using API key: {API_KEY}')
    df = pd.DataFrame()
    for t in tickers:
        ticker_df = get_candle_data_for_ticker(API_KEY, t, start_date, end_date, resolution, adjusted)
        # append the new data to the existing data for this ticker that may have been loaded from file
        df = pd.concat([df, ticker_df], axis=1)

    # now we have a dataframe with all the data we need
    # filter for the desired date range
    df = df[df.index >= input_start_date]
    df = df[df.index <= input_end_date]
    
    cd = CandleData(resolution, 'polygonio')
    cd.add_data(df, tickers)
    return cd

