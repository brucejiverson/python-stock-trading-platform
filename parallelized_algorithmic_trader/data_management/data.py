from __future__ import annotations
from typing import Tuple, List
import os
import time
import datetime
import pandas as pd

from parallelized_algorithmic_trader.base import Base
import parallelized_algorithmic_trader.data_management.data_utils as data_utils
from parallelized_algorithmic_trader.util import DATA_DIRECTORY, get_logger, DateRange

# specific data handlers that handle the interface with the actual API and standardize the data format for the rest of the system
import parallelized_algorithmic_trader.data_management.alpaca_data as alpaca_data
import parallelized_algorithmic_trader.data_management.polygon_io as po


logger = get_logger(__name__)


class EquityData(Base):
    """This class represents the data for a set of Equities."""
    def __init__(self, resolution: data_utils.TemporalResolution):
        """Initialize the data object."""
        self.resolution:data_utils.TemporalResolution = resolution
        self.tickers:List[str] = []
        super().__init__('candle_data')


class CandleData(EquityData):
    def __init__(self, df:pd.DataFrame, tickers:List[str], resolution: data_utils.TemporalResolution):

        super().__init__(resolution)
        self.df:pd.DataFrame = CandleData.vaidate_and_clean_data(df, tickers)
        self.tickers:List[str] = tickers
        self.logger.debug(f'Initializing with dataframe head: \n{self.df.head()}')
        assert isinstance(df.index, pd.DatetimeIndex), 'Dataframe index must be a DatetimeIndex'
        
        # update the start and end
        self.start_date:pd.Timestamp = self.df.index[0]
        self.end_date:pd.Timestamp = self.df.index[-1]
        
    @staticmethod
    def vaidate_and_clean_data(df:pd.DataFrame, tickers:List[str]):
        """This function validates and cleans the data for the given dataframe.
        
        """
        
        # validate that the column names are named correctly
        kwards = ['open', 'high', 'low', 'close', 'volume']
        for tckr in tickers:
            for kw in kwards:
                assert tckr + '_' + kw in df.columns, f'Column {tckr + "_" + kw} not found in df'

        if 0:
            cleaned_df = df.copy()
        else:
            cleaned_df = data_utils.sanitize_dataframe(df)
        return cleaned_df

        # do some basic clean up and validation of the dataframe format
        # data_resolution = self._infer_data_resolution(cleaned_df)
        # print(f'df after infer')
        # print(cleaned_df.index[0:6])
        # print(f'Detected data resolution: {data_resolution.name}')

        # if self.resolution != data_resolution:
        #     raise ValueError (f'Given resolution {self.resolution.name} does not match the data resolution {data_resolution.name}')
        
        # old logic for mutating the attr of this class to add more data
        # if self.df is None:
        #     self.df = cleaned_df
        # else:
        #     # df = pd.concat([df, ticker_df], axis=1)        
        #     self.df = pd.merge(self.df, cleaned_df, left_index=True, right_index=True, how='inner')
        # self.tickers.append(ticker)

    def _check_to_convert_to_resolution(spm:float, MARGIN_PERCENT:float = 10) -> data_utils.TemporalResolution:
        """This function checks if the given number of samples per minute is close enough to the given resolution to be considered that resolution.
        
        :param spm: The number of samples per minute
        :param MARGIN_PERCENT: The margin of error in the percentage of samples per minute
        """
        # this accounts for missing data samples and gives some flexibility in the calculations.
        # The assumption here is that even if data is missing it will only be individual samples and not big chunks of data.
        
        if 1*(1 - MARGIN_PERCENT/100) < spm < 1*(1 + MARGIN_PERCENT/100):
            return data_utils.TemporalResolution.MINUTE
        elif 60*(1 - MARGIN_PERCENT/100) < spm < 60*(1 + MARGIN_PERCENT/100):
            return data_utils.TemporalResolution.HOUR
        elif 60*24*(1 - MARGIN_PERCENT/100) < spm < 60*24*(1 + MARGIN_PERCENT/100):
            return data_utils.TemporalResolution.DAY
        elif 60*24*7*(1 - MARGIN_PERCENT/100) < spm < 60*24*7*(1 + MARGIN_PERCENT/100):
            return data_utils.TemporalResolution.WEEK
        else:
            raise ValueError('Data resolution could not be determined. Please check the data and try again.')

    def _infer_data_resolution(self, df:pd.DataFrame=None) -> data_utils.TemporalResolution:
        """This function parses the timestamp column of the dataframe and determines the resolution of the data.
        
        :param df: The dataframe to parse
        """
        # get the difference between the first and last timestamp
        mean_samples_per_min:pd.Timedelta = df.index[1] - df.index[0]
        first_timestamp:pd.Timestamp = df.index[0]
        k = 0.1
        i = -1

        for i, timestamp in enumerate(df.index):
            if timestamp == first_timestamp:
                last_timestamp = timestamp
                continue

            # use exponential moving average to smooth out the data
            mean_samples_per_min = mean_samples_per_min * k + (timestamp - last_timestamp) * (1- k)
            last_timestamp = timestamp

            samples_per_min = mean_samples_per_min.total_seconds() / 60
            if i == 2: 
                res = self._check_to_convert_to_resolution(samples_per_min, 1) 
            elif i > 20:
                res = self._check_to_convert_to_resolution(samples_per_min)
            else: continue
            if res is not None: return res
        return self._check_to_convert_to_resolution(samples_per_min) # case where we have fewer candles than the limit in the for loop

    def split(self, fraction:float=0.2) -> Tuple[CandleData, CandleData]:
        """Returns two new objects split into two fractions. Returns a tuple of two FinancialData objects, (train, test)"""
        train_df, test_df = split_data_frame_by_fraction(self.df, fraction)

        train_data = CandleData(train_df, self.tickers, self.resolution)
        test_data = CandleData(test_df, self.tickers, self.resolution)
        
        return train_data, test_data
    
    # def k_fold_split(self, k:int=5) -> List[CandleData]:
    #     """Returns a list of k CandleData objects split into k folds."""
    #     print('\nSplitting data...')
    #     data = self.df

    #     folds = []
    #     last_split_idx = 0
    #     for i in range(k):
    #         fold = CandleData(self.resolution)
    #         split_idx = round( ((i+1)/k) * data.shape[0])
    #         fold.add_data(data.iloc[last_split_idx::i*split_idx], self.tickers)
    #         folds.append(fold)
    #     return folds


def split_data_frame_by_fraction(df:pd.DataFrame, fraction:float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns two dataframes split into two fractions as a tuple.
    
    :param df: The dataframe to split
    :param fraction: The fraction of the dataframe to split off
    """
    
    assert fraction < 1, 'Fraction must be less than 1'
    
    split_idx = round(fraction * df.shape[0])
    df1, df2 = df.iloc[0:split_idx], df.iloc[split_idx::]
    return df1, df2


# global variables used to track API calls and make sure we don't exceed the rate limit.
N_API_CALLS_SINCE_START = 0
API_CALLS_START_TIME:float = None


def _aggregate_candle_data_for_ticker(
    API_KEY:str,
    SECRET_KEY:str,
    ticker:str, 
    time_range:DateRange,
    resolution:data_utils.TemporalResolution, 
    data_source:str,
    max_api_calls_per_min:int=100) -> pd.DataFrame|None:
    """Alpaca limits on the amount of data returned per call. This function repeatedly calls the API looping over the full time range
    in order to get all of the data, which is then sorted, serialized, and stored in a file.

    Parameters:
        ticker: The ticker symbols.
        time_range: The time range to get data for.
        resolution: The resolution of the data.
        max_api_calls_per_min: The maximum number of API calls per minute. Polygon limits this to 5 for the basic package. All other pricing plans are unlimited
        
    Returns:
        pd.DataFrame object
    """

    global N_API_CALLS_SINCE_START
    global API_CALLS_START_TIME

    logger.debug(f'Aggregating candle data from {time_range.start.date()} to {time_range.end.date()} for ticker {ticker} with resolution {resolution.name}')
    
    s = time_range.start    # variable representing the start date for fetching data. Each loop this gets moved to the end of the retrieved data
    next_request_start = time_range.start
    df = None

    if API_CALLS_START_TIME is None:
        API_CALLS_START_TIME = time.monotonic()

    while s < time_range.end - datetime.timedelta(days=3):
        logger.debug(f's: {s} | time_range.end: {time_range.end}')
        # fetch the data from polygon
        tm_rnge = DateRange(next_request_start, time_range.end)
        
        match data_source:
            case 'alpaca':
                partial_history = alpaca_data._get_single_candle_data_batch(API_KEY, SECRET_KEY, ticker, tm_rnge, resolution)
            case 'polygon':
                raise NotImplementedError
            # default
            case _:
                raise ValueError(f'Unsupported data source: {data_source}')
        
        N_API_CALLS_SINCE_START += 1
        
        if partial_history is None:
            logger.warning('No data received from data source for ticker {} in time range {}. Skipping any further pulling and returning what has been found.'.format(ticker, tm_rnge))
            break

        # append it to the dataframe
        if df is None:
            df = partial_history
        else:
            df = pd.concat([df, partial_history])

        # update the start dates
        next_request_start = partial_history.index[-2]
        s = partial_history.index[-1]
        # sleep if we've hit the max number of API calls to avoid API call limit
        if not max_api_calls_per_min is None and N_API_CALLS_SINCE_START >= max_api_calls_per_min:
            elapsed = time.monotonic() - API_CALLS_START_TIME
            if 1 or elapsed < 60:
                sleep_time = 60 + 1
                logger.info('Waiting {:.1f} seconds to make the next API call...'.format(sleep_time))
                time.sleep(sleep_time)
            
            N_API_CALLS_SINCE_START = 0
            API_CALLS_START_TIME = time.monotonic()

    # remove duplicates in the timestamp column
    if df is None:
        logger.warning(f'No data found for ticker {ticker} in time range {time_range}')
        return None
    return data_utils.sanitize_dataframe(df)


def _get_candle_data_for_ticker(
    API_KEY:str,
    SECRET_KEY:str,
    ticker:str, 
    requested_time_range:DateRange, 
    resolution:data_utils.TemporalResolution,
    data_source:str) -> pd.DataFrame:
    """Checks the disk to see if the data for ticker is already on the harddrive. If
    not, it will fetch the data from alpaca, store it locally and return it."""

    logger.info(f'Checking for local data for ticker {ticker} from {requested_time_range.start.date()} to {requested_time_range.end.date()} with resolution {resolution.name}')

    # check to see if a file already exists for this ticker and resolution
    file_name:str = data_utils.build_data_file_name(ticker, resolution)
    file_path:str = os.path.join(DATA_DIRECTORY, file_name)
    df:pd.DataFrame = pd.DataFrame()

    # check if the file exists and get the data if it does
    if os.path.exists(file_path):
        logger.debug(f'Found a file for {ticker} at {file_path}')
        df:pd.DataFrame = pd.read_pickle(file_path)
        file_data_time_range = DateRange(df.index[0], df.index[-1])
        logger.debug(f'File data start: {file_data_time_range.start}, end: {file_data_time_range.end}')
        
        logger.debug(f'Checking to see if the data covers the desired date range')
        file_covers_range = data_utils.check_if_data_covers_timerange(df, requested_time_range)
        if file_covers_range:
            logger.info(f'File data with date range {file_data_time_range} covers desired date range {requested_time_range}')
            return data_utils.filter_df_for_daterange(df, requested_time_range)
        else:
            logger.info(f'Found file data, data {file_data_time_range} does not cover desired range {requested_time_range}')
            
        # ensure that there are no gaps in the data but also only pull the range that we need to 
        if requested_time_range.end > file_data_time_range.end:
            needed_time_range = DateRange(
                file_data_time_range.end - datetime.timedelta(days=1),   
                requested_time_range.end
            )
        else:
            needed_time_range = DateRange(
                requested_time_range.start,
                file_data_time_range.start + datetime.timedelta(days=1)
            )
    else:
        logger.info(f"No local files found for ticker {ticker}")
        needed_time_range = requested_time_range

    # get the data from
    new_data = _aggregate_candle_data_for_ticker(
        API_KEY, 
        SECRET_KEY,
        ticker, 
        needed_time_range,
        resolution,
        data_source)
    
    # append the new data to the existing data for this ticker that may have been loaded from file
    # if new_data is not empty
    if new_data is not None:
        logger.debug(f"New data dates: {new_data.index[0]} to {new_data.index[-1]}")
        df = pd.concat([df, new_data])
        df.sort_index(inplace=True)
        df = data_utils.sanitize_dataframe(df)
        logger.debug(f'Dates after cleaning: {df.index[0]} to {df.index[-1]}')
    else:
        logger.warning(f'Warning: retrived no data from {data_source} for ticker {ticker}!')
        if df.empty:
            logger.error(f'Error: no data found for ticker {ticker}')
            raise Exception(f'No data found locally or through {data_source} for ticker {ticker}')
        
    df.to_pickle(file_path)
    logger.debug(f'saved to file: {file_path}')
    logger.info(f'Successfully loaded data for ticker {ticker}.')
    
    return data_utils.filter_df_for_daterange(df, requested_time_range)


def get_candle_data(
    tickers:List[str], 
    resolution:data_utils.TemporalResolution,
    start_date:datetime.datetime, 
    end_date:datetime.datetime|None=None,                    # if None, will default to today
    data_source:str='alpaca',
    API_KEY:str=None,
    SECRET_KEY:str=None) -> CandleData:
    """Fetches data from either disk or polygon API, formatted in dataframe for each ticker. Returns formal CandleData object.
    
    :param tickers: list of tickers to fetch data for
    :param resolution: time resolution of data 
    :param start_date: start date for data
    :param end_date: end date for data as datetime. If None, defaults to two days ago.
    :param API_KEY: API key. If not provided, will default to looking for environment variable named ALPACA
    :param SECRET_KEY: secret key, optional. If not provided, will default to looking for environment variable named ALPACA_SECRET
    """

    if API_KEY is None: 
        logger.warning(f'API_KEY not provided, defaulting to environment variable ALPACA.')
        API_KEY = os.environ.get('ALPACA')
    if SECRET_KEY is None:
        SECRET_KEY = os.environ.get('ALPACA_SECRET')
        
    if end_date is None:
        logger.debug('No end date provided, defaulting to today.')
        end_date = datetime.datetime.now() - datetime.timedelta(days=2)
        
    # strip out the time information
    logger.info(f'Fetching historical data for tickers {tickers} from {start_date.date()} to {end_date.date()} with resolution {resolution.name}')
    
    time_range = DateRange(
        data_utils.ceiling_to_subsequent_business_date(start_date), 
        data_utils.floor_to_preceding_business_day(end_date)
        )
    logger.debug(f'Time range after business day adjustment: {time_range.start} to {time_range.end}')
    
    df = pd.DataFrame()
    for t in tickers:
        ticker_df = _get_candle_data_for_ticker(API_KEY, SECRET_KEY, t, time_range, resolution, data_source)
        # append the new data to the existing data for this ticker that may have been loaded from file
        df = pd.concat([df, ticker_df], axis=1)
    
    df['Source'] = ['alpaca' for i in range(len(df))]
    logger.debug(f'Total real time range retrieved: {df.index[0]} to {df.index[-1]}')
    return CandleData(df, tickers, resolution)


if __name__ == "__main__":
    tickers = ['SPY', 'AAPL']
    res = data_utils.TemporalResolution.DAY

    end = datetime.datetime.now()
    start = end - datetime.timedelta(weeks=5*52)
    
    candles = get_candle_data(tickers, start, end, res)
    
    print(candles.df.head())
    print(candles.df.tail())
    