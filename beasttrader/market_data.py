from __future__ import annotations
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime, timedelta
import os
import pickle
from typing import Tuple, Optional, List

from beasttrader.util import maybe_make_dir, printProgressBar, DATA_DIRECTORY


class TemporalResolution(Enum):
    """This class is used to define the resolution of the data"""
    MINUTE = "minute"
    FIVE_MINUTE = "5minute"
    FIFTEEN_MINUTE = "15minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

    def get_as_minutes(self):
        if self == TemporalResolution.MINUTE:
            return 1
        elif self == TemporalResolution.HOUR:
            return 60
        elif self == TemporalResolution.DAY:
            return 1440
        elif self == TemporalResolution.WEEK:
            return 10080
        elif self == TemporalResolution.MONTH:
            return 43200
        else:
            raise ValueError("Unknown resolution")
 

class EquityData:
    def __init__(self, resolution: TemporalResolution):
        """Initialize the data object."""
        self.resolution:TemporalResolution = resolution
        self.tickers:List[str] = []


    def _get_file_path(self):
        """Get the file name for the data."""
        return os.path.join(DATA_DIRECTORY, f'{self.ticker}/{self.data_type.name}_{self.resolution.name}.pkl')


class OrderBookData(EquityData): pass


class CandleData(EquityData):
    def __init__(
        self, 
        resolution: TemporalResolution):

        super().__init__(resolution)
        self.df:pd.DataFrame|None = None
        self.start:pd.Timestamp|None = None
        self.end:pd.Timestamp|None = None
        
    def add_data(self, new_df:pd.DataFrame, tickers:List[str], suppress_cleaning_data:bool=False):
        # validate that the column names are named correctly
        kwards = ['open', 'high', 'low', 'close', 'volume']
        for t in tickers:
            for kw in kwards:
                assert t + '_' + kw in new_df.columns, f'Column {t + "_" + kw} not found in new_df'

        if suppress_cleaning_data:
            cleaned_df = new_df.copy()
        else:
            cleaned_df = clean_dataframe(new_df)
        # print(f'cleaned')
        # print(cleaned_df.head())

        # do some basic clean up and validation of the dataframe format
        # data_resolution = self._infer_data_resolution(cleaned_df)
        # print(f'df after infer')
        # print(cleaned_df.index[0:6])
        # print(f'Detected data resolution: {data_resolution.name}')

        # if self.resolution != data_resolution:
        #     raise ValueError (f'Given resolution {self.resolution.name} does not match the data resolution {data_resolution.name}')
        
        if self.df is None:
            self.df = cleaned_df
        else:
            self.df = pd.merge(self.df, cleaned_df, left_index=True, right_index=True, how='inner')
        [self.tickers.append(t) for t in tickers if t not in self.tickers]

        # set the start and end
        self.start = self.df.index[0]
        self.end = self.df.index[-1]

    def filter_data_by_date(self, start:datetime=None, end:datetime=None):
        """This function filters the data by the given start and end dates."""
        if start is not None:
            self.df = self.df[self.df.index >= start]
        if end is not None:
            self.df = self.df[self.df.index <= end]

    def _infer_data_resolution(self, df:pd.DataFrame=None) -> TemporalResolution:
        """This function parses the timestamp column of the dataframe and determines the resolution of the data"""
        # get the first and last timestamp
        first:pd.Timestamp = df.index[0]
        
        # get the difference between the first and last timestamp
        mean_samples_per_min:pd.Timedelta = df.index[1] - df.index[0]
        first_timestamp:pd.Timestamp = df.index[0]
        k = 0.1
        i = -1

        def _check_to_convert_to_resolution(spm:float, MARGIN_PERCENT:float = 10) -> TemporalResolution:
            """This function checks if the given number of samples per minute is close enough to the given resolution to be considered that resolution"""
            # this accounts for missing data samples and gives some flexibility in the calculations.
            # The assumption here is that even if data is missing it will only be individual samples and not big chunks of data.
            
            if 1*(1 - MARGIN_PERCENT/100) < samples_per_min < 1*(1 + MARGIN_PERCENT/100):
                return TemporalResolution.MINUTE
            elif 60*(1 - MARGIN_PERCENT/100) < samples_per_min < 60*(1 + MARGIN_PERCENT/100):
                return TemporalResolution.HOUR
            elif 60*24*(1 - MARGIN_PERCENT/100) < samples_per_min < 60*24*(1 + MARGIN_PERCENT/100):
                return TemporalResolution.DAY
            elif 60*24*7*(1 - MARGIN_PERCENT/100) < samples_per_min < 60*24*7*(1 + MARGIN_PERCENT/100):
                return TemporalResolution.WEEK
            else:
                print(f'First timestamp: {first}, samples per minute: {samples_per_min}, last timestamp: {last_timestamp}')
                print(df.head())
                raise ValueError('Data resolution could not be determined. Please check the data and try again.')

        for i, timestamp in enumerate(df.index):
            if timestamp == first_timestamp:
                last_timestamp = timestamp
                continue

            # use exponential moving average to smooth out the data
            mean_samples_per_min = mean_samples_per_min * k + (timestamp - last_timestamp) * (1- k)
            last_timestamp = timestamp

            samples_per_min = mean_samples_per_min.total_seconds() / 60
            if i == 2: 
                res = _check_to_convert_to_resolution(samples_per_min, 1) 
            elif i > 20:
                res = _check_to_convert_to_resolution(samples_per_min)
            else: continue
            if res is not None: return res
        return _check_to_convert_to_resolution(samples_per_min) # case where we have fewer candles than the limit in the for loop

    def resample(self, new_resolution: TemporalResolution) -> pd.DataFrame:
        """This function looks at the Date columns of the df and modifies the df according to the given resolution. 
        The dataframe held by the class is modified in place."""
        df = self.df.copy()
        gran = self.granularity
        if gran == 1:
            print("Granularity is set to 1 minute.")
            return df

        new_df = pd.DataFrame(columns = df.columns)
        start = df.index[0]
        # Get the starting minute as a multiple
        m = start.minute
        start += timedelta(minutes=(gran - m + 1))
        
        oldest = max(df.index)

        #Loop over the entire dataframe. assumption is that candle df is in 1 min intervals
        length = df.shape[0]
        i = 0
        while True:
            if i > 100 and i%20 ==0:
                printProgressBar(i, length, prefix='Progress:', suffix='Complete')

            end = start + timedelta(minutes=gran-1)
            data = df.loc[(df.index >= start) & (df.index <= end)]
            
            try:
                # Note that timestamps are the close time
                candle = pd.DataFrame({'BTCOpen': data.iloc[0]['BTCOpen'],
                                        'BTCHigh': max(data['BTCHigh']),
                                        'BTCLow': min(data['BTCLow']),
                                        'BTCClose': data.iloc[-1]['BTCClose'],
                                        'BTCVolume': sum(data['BTCVolume'])},
                                        index=[end])
                new_df = new_df.append(candle)
            # Handle empty slices (ignore)
            except IndexError:
                pass
            if end >= oldest: break
            start += timedelta(minutes=gran)
            # This is for printing the progress bar
            try:
                i = df.index.get_loc(start)
            except KeyError:
                pass
        
        self.logger.info(f'Successfully resampled data from {self.resolution.name} to {new_resolution.name} resolution.')
        self.resolution = new_resolution
        return new_df
    
    def save_to_file(self):
        """Save the data to a file."""
        maybe_make_dir(DATA_DIRECTORY)
        # pickle the data
        PATH = self._get_file_path()
        self.logger.info(f'Writing data to file: {PATH}')
        with open(PATH, 'wb') as f:
            pickle.dump(self.df, f)

    def split(self, fraction:float=0.2) -> Tuple[CandleData, CandleData]:
        """Returns two new objects split into two fractions. Returns a tuple of two FinancialData objects, (train, test)"""
        train_df, test_df = split_data_frame(self.df, fraction)

        train_data, test_data = CandleData(self.resolution), CandleData(self.resolution)
        train_data.add_data(train_df, self.tickers)
        test_data.add_data(test_df, self.tickers)
        return train_data, test_data
    
    def k_fold_split(self, k:int=5) -> List[CandleData]:
        """Returns a list of k CandleData objects split into k folds."""
        print('\nSplitting data...')
        data = self.df

        folds = []
        last_split_idx = 0
        for i in range(k):
            fold = CandleData(self.resolution)
            split_idx = round( ((i+1)/k) * data.shape[0])
            fold.add_data(data.iloc[last_split_idx::i*split_idx], self.tickers)
            folds.append(fold)
        return folds


def split_data_frame(df, fraction=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns two dataframes split into two fractions. Returns a tuple of two dataframes, (train, test)"""
    split_idx = round(fraction * df.shape[0])
    df1, df2 = df.iloc[0:split_idx], df.iloc[split_idx::]
    return df1, df2


def clean_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """This function formats the dataframe according to the assets that are in it.
    Needs to be updated to handle multiple assets. Note that this should only be used before high low open are stripped from the data."""
    formatted_df = df.copy()
    
    # check to see if the dataframe has a timestamp column and if so make the index the timestamp
    if 'timestamp' in formatted_df.columns and type(formatted_df.index) != pd.DatetimeIndex:
        # drop any duplicate timestamps
        formatted_df.drop_duplicates(subset='timestamp', inplace=True)
        formatted_df.set_index('timestamp', inplace=True, verify_integrity=True, drop=True)
    elif type(formatted_df.index) != pd.DatetimeIndex:
        raise ValueError('Dataframe must have a column named timestamp or the index must be a DatetimeIndex.')

    # sort by index
    formatted_df.sort_index(inplace=True)

    # drop the last line. polygon.io does this sometimes, not sure why. ignoring for now
    if pd.isnull(formatted_df.index[-1]): 
        formatted_df = formatted_df[:-1]

    # if len(formatted_df.columns) > 6:
    #     print('Dataframe has more than 5 columns. Only open, high, low, close, volume will be used, the rest will be dropped.')
    #     formatted_df = formatted_df[['open', 'high', 'low', 'close', 'volume', 'timestamp']]

    # get the number of duplicates in the dataframe
    # num_duplicates = formatted_df.index.duplicated().sum()
    # print(f'Number of duplicates: {num_duplicates}')
    # formatted_df = formatted_df.loc[~formatted_df.index.duplicated(keep = 'first')]     # this is intended to remove duplicates. ~ flips bits in the mask
    # Replacing infinite with nan 
    formatted_df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    formatted_df.dropna(inplace=True)

    # # search for inconsistent jumps in the timestamps of the data frame
    # for i in range(1, len(formatted_df)):
    #     if formatted_df.index[i] - formatted_df.index[i-1] != timedelta(minutes=1):
    #         print('Inconsistent time stamps found at index: {}'.format(i))
    #         print('Timestamps: {} and {}'.format(formatted_df.index[i-1], formatted_df.index[i]))
    #         print('Difference: {}'.format(formatted_df.index[i] - formatted_df.index[i-1]))
    #         print('Dropping the inconsistent timestamps')
    #         formatted_df.drop(formatted_df.index[i], inplace = True)

    return formatted_df


def compare_resolution_size(first_res: TemporalResolution, second_res: TemporalResolution) -> int:
    """This function compares the size of two resolutions and returns 1 if the first resolution is smaller, -1 if the second resolution is smaller, and 0 if they are equal."""
    if first_res == second_res: return 0
    if first_res == TemporalResolution.MINUTE: return 1
    if first_res == TemporalResolution.HOUR and second_res in (TemporalResolution.DAY, TemporalResolution.WEEK): return 1
    if first_res == TemporalResolution.DAY and second_res == TemporalResolution.WEEK: return 1
    return -1


def get_candle_data_from_file(
    ticker: str, 
    data_type: type=CandleData, 
    resolution: TemporalResolution=TemporalResolution.HOUR, 
    start: datetime=None, 
    end: datetime=None) -> pd.DataFrame:
    """This function returns a dataframe of the equity data from the given file. The data is resampled to the given resolution.
    The start and end dates are inclusive. The granularity is the number of minutes per candle."""
    
    # intuit the file name from the ticker. 
    PATH = os.path.join(DATA_DIRECTORY, f'{ticker}/{data_type.__name__}_{resolution.name}.pkl')

    if not os.path.exists(PATH):
        # search for a file matching the resolution. It's useful to have multiple files for different resolutions as it makes it faster to load the data.
        for res in TemporalResolution:
            # ensure that res is smaller than the given resolution (required for resampling)
            if not compare_resolution_size(res, resolution) == 1: 
                print(f'Found a file for this ticker but the resolution is not smaller than the given resolution. Skipping.')
                continue

            PATH = os.path.join(DATA_DIRECTORY, f'{ticker}/OHLC_{res.name}.pkl')
            if os.path.exists(PATH):
                print(F'Found a file with resolution {res.name} for {ticker}. Loading that file instead. It should get resampled later.')
                break
    
    # get the data from the file
    df = pd.read_csv(f'./data/{ticker}.csv', index_col = 'timestamp', parse_dates = True)
    # create the EquityData object
    start_end = [item for item in [start, end] if item is not None]
    data = CandleData(df, resolution, ticker, *start_end)
    return data

