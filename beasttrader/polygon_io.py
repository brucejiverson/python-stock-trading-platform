from time import monotonic
import logging
from polygon import RESTClient
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import List
import time


from beasttrader.market_data import CandleData, TemporalResolution, clean_dataframe
from beasttrader.util import DATA_DIRECTORY

logger = logging.getLogger(__name__)


def get_polygon_key():
    """Get Polygon API key from file."""
    f_path = os.path.join(os.path.dirname(__file__), "polygon_key.txt")
    # logger.info(F'install path: {f_path}')
    with open(f_path, 'r') as f:
        key = f.read()
    return key


def get_candle_data_as_df(tickers:List[str], start_date:datetime, end_date:datetime, resolution:TemporalResolution, adjusted=True) -> pd.DataFrame:
    """Get candle data from Polygon API. With a single API call.
    
    Get aggregate bars for a ticker over a given date range in custom time window sizes.

    :param ticker: The ticker symbol.
    :param multiplier: The size of the timespan multiplier.
    :param timespan: The size of the time window.
    :param _from: The start of the aggregate time window as YYYY-MM-DD, a date, Unix MS Timestamp, or a datetime.
    :param to: The end of the aggregate time window as YYYY-MM-DD, a date, Unix MS Timestamp, or a datetime.
    :param adjusted: Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
    :param sort: Sort the results by timestamp. asc will return results in ascending order (oldest at the top), desc will return results in descending order (newest at the top).The end of the aggregate time window.
    :param limit: Limits the number of base aggregates queried to create the aggregate results. Max 50000 and Default 5000. Read more about how limit is used to calculate aggregate results in our article on Aggregate Data API Improvements.
    :param params: Any additional query params
    :param raw: Return raw object instead of results object
    :return: List of aggregates
    """

    # convert the resolution to a string
    if resolution == TemporalResolution.MINUTE:
        timespan = 'minute'
    elif resolution == TemporalResolution.HOUR:
        timespan = 'hour'
    elif resolution == TemporalResolution.DAY:
        timespan = 'day'
    else: raise ValueError(f'Invalid resolution: {resolution}')

    client = RESTClient(get_polygon_key())

    start, end = str(start_date.date()), str(end_date.date())
    for ticker in tickers:
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
    

def get_candles(tickers:List[str], start_date:datetime, end_date:datetime, resolution:TemporalResolution, adjusted=True) -> CandleData:
    """Fetches data from the polygon API, constructs a dataframe and then makes a formalized CandleData object.
    
    Get aggregate bars for a ticker over a given date range in custom time window sizes.
    This function pulls data from the exchange, the cumulative repository, and the data download in that order
        depending on the input date range. Note that the datetimes are converted to dates here. Interday data operations are not supported.

    :param ticker: The ticker symbol.
    :param multiplier: The size of the timespan multiplier.
    :param timespan: The size of the time window.
    :param _from: The start of the aggregate time window as YYYY-MM-DD, a date, Unix MS Timestamp, or a datetime.
    :param to: The end of the aggregate time window as YYYY-MM-DD, a date, Unix MS Timestamp, or a datetime.
    :param adjusted: Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
    :param sort: Sort the results by timestamp. asc will return results in ascending order (oldest at the top), desc will return results in descending order (newest at the top).The end of the aggregate time window.
    :param limit: Limits the number of base aggregates queried to create the aggregate results. Max 50000 and Default 5000. Read more about how limit is used to calculate aggregate results in our article on Aggregate Data API Improvements.
    :param params: Any additional query params
    :param raw: Return raw object instead of results object
    :return: List of aggregates
    """

    """algorithm pseudocode:
    
    look for a file with the ticker name and desired resolution
    if it exists, load it, and check the start and end dates

    if needed, get data from polygon.io
    if file already existed, append the new data
    save data to file

    note that this is significatly simpler/faster because the bittrex exchange was only offering the most recent 2 weeks of data.
    """

    def get_file_name(ticker:str, resolution:TemporalResolution) -> str:
        return ticker+'_'+resolution.name.lower()+'.pkl'

    # strip out the time information
    input_start_date, input_end_date = datetime(start_date.year, start_date.month, start_date.day), datetime(end_date.year, end_date.month, end_date.day)    
    logger.info(f'Fetching historical data for tickers {tickers} from {input_start_date.date()} to {input_end_date.date()} with resolution {resolution.name}')

    df = pd.DataFrame()
    for t in tickers:
        # check to see if a file already exists for this ticker and resolution
        file_name = get_file_name(t, resolution)
        file_path = os.path.join(DATA_DIRECTORY, file_name)
        ticker_df = pd.DataFrame()

        # check if the file exists and get the data if it does
        if os.path.exists(file_path):
            logger.info(F'Found a file for {t} at {file_path}')
            # load the existing data
            ticker_df:pd.DataFrame = pd.read_pickle(file_path)
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
                logger.info(f'File data for ticker {t} from {file_data_start} to {file_data_end} covers the desired date range from {input_start_date} to {input_end_date}')
                df = pd.concat([df, ticker_df], axis=1)
                continue
        else:
            file_data_start, file_data_end = input_start_date, input_end_date

        # get the data from polygon.io
        # ensure that there are no gaps in the data but also only pull the range that we need to 
        s = min(input_start_date, file_data_end) - timedelta(days=1)
        e = max(input_end_date, file_data_start)# + timedelta(days=1) 

        new_data = aggregate_many_candles([t], s, e, resolution, adjusted=adjusted)
        # append the new data to the existing data for this ticker that may have been loaded from file
        ticker_df = pd.concat([ticker_df, new_data], ignore_index=True)
        ticker_df = clean_dataframe(ticker_df)
        # save the data
        ticker_df.to_pickle(file_path)
        logger.info(f'saved to file: {file_path}')
        logger.info(f'Fetched data for ticker {t}, date range now from {ticker_df.index[0]} to {ticker_df.index[-1]}')
        df = pd.concat([df, ticker_df], axis=1)

    # now we have a dataframe with all the data we need
    cd = CandleData(resolution)
    cd.add_data(df, tickers)
    return cd


# global variables used to track API calls and make sure we don't exceed the rate limit. Not exactly elegant but it works. 
N_API_CALLS_SINCE_START = 0
API_CALLS_START:float = monotonic()


def aggregate_many_candles(
    tickers:List[str], 
    start_date:datetime, 
    end_date:datetime, 
    resolution:TemporalResolution, 
    adjusted=True, 
    max_api_calls_per_min:int=5) -> CandleData:
    """Polygon seems to set limits on the amount of data they return per call. This function will repeatedly call the polygon API looping over the full time range
    in order to get all of the data. Subsequently, the data will be serialized and stored in a file.

    :param tickers: The ticker symbols.
    :param start_date: The start of the aggregate time window as YYYY-MM-DD, a date, Unix MS Timestamp, or a datetime.
    :param end_date: The end of the aggregate time window as YYYY-MM-DD, a date, Unix MS Timestamp, or a datetime.
    :param resolution: The resolution of the data.
    :param adjusted: Whether or not the results are adjusted for splits. By default, results are adjusted. Set this to false to get results that are NOT adjusted for splits.
    :param max_api_calls_per_min: The maximum number of API calls per minute. Polygon limits this to 5 for the basic package. All other pricing plans are unlimited
    :return: CandleData object
    """

    logger.debug(f'Aggregating candle data from {start_date.date()} to {end_date.date()} for tickers {tickers} with resolution {resolution.name}')
    
    s = start_date
    df = pd.DataFrame()

    N_API_CALLS_SINCE_START = 0
    API_CALLS_START = monotonic()

    while s < end_date - timedelta(days=2):
        # fetch the data from polygon
        small_df = get_candle_data_as_df(tickers, s, end_date, resolution, adjusted)
        N_API_CALLS_SINCE_START += 1

        # append it to the dataframe
        df = pd.concat([df, small_df], ignore_index=True)

        # update the start dates
        s = small_df['timestamp'].iloc[-1]

        # sleep if we've hit the max number of API calls
        if not max_api_calls_per_min is None and N_API_CALLS_SINCE_START >= max_api_calls_per_min:
            elapsed = monotonic() - API_CALLS_START
            if 1 or elapsed < 60:
                sleep_time = 61
                logger.info('Waiting {:.1f} seconds to make the next API call...'.format(sleep_time))
                time.sleep(sleep_time)
            
            N_API_CALLS_SINCE_START = 0
            API_CALLS_START = monotonic()

        # logger.debug(f'new s: {s}. end date: {end_date}')
    return df
