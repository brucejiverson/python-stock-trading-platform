from __future__ import annotations
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime
from typing import Tuple, List

from parallelized_algorithmic_trader.base import Base


class TemporalResolution(Enum):
    """This class represents units of time."""
    MINUTE = "minute"
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
 

class EquityData(Base):
    """This class represents the data for a set of Equities."""
    def __init__(self, resolution: TemporalResolution):
        """Initialize the data object."""
        self.resolution:TemporalResolution = resolution
        self.tickers:List[str] = []
        super().__init__('candle_data')


def get_data_file_name(ticker:str, resolution:TemporalResolution) -> str:
    """A helper function for getting the name of the file that the data for a ticker
    
    :param ticker: The ticker of the equity
    :resolution: The time per candle
    """
    return ticker+'_'+resolution.name.lower()+'.pkl'


class CandleData(EquityData):
    def __init__(
        self, 
        resolution: TemporalResolution, 
        source:str
        ):

        super().__init__(resolution)
        self.df:pd.DataFrame|None = None
        self.start_date:pd.Timestamp|None = None
        self.end_date:pd.Timestamp|None = None
        self.source = source
        
    def add_data(self, new_df:pd.DataFrame, ticker:str, suppress_cleaning_data:bool=False):
        """Appends the given dataframe to the existing dataframe. Cleans the data.
        
        :param new_df: The dataframe to append to the existing dataframe
        :param ticker: The ticker of the equity
        :param suppress_cleaning_data: If True, the data will not be cleaned. This is useful for testing.
        """
        
        self.logger.debug(f'Adding data for {ticker} to the existing dataframe')
        self.logger.debug(new_df.head())
        
        # validate that the column names are named correctly
        kwards = ['open', 'high', 'low', 'close', 'volume']
        for kw in kwards:
            assert ticker + '_' + kw in new_df.columns, f'Column {ticker + "_" + kw} not found in new_df'

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
        self.tickers.append(ticker)

        # update the start and end
        self.start_date = self.df.index[0]
        self.end_date = self.df.index[-1]

    def filter_data_by_date(self, start:datetime=None, end:datetime=None):
        """This function filters the data by the given start and end dates.
        
        :param start: The start date
        :param end: The end date
        """
        if start is not None:
            self.df = self.df[self.df.index >= start]
        if end is not None:
            self.df = self.df[self.df.index <= end]
        self.start_date = self.df.index[0]
        self.end_date = self.df.index[-1]        
    
    def _check_to_convert_to_resolution(spm:float, MARGIN_PERCENT:float = 10) -> TemporalResolution:
        """This function checks if the given number of samples per minute is close enough to the given resolution to be considered that resolution.
        
        :param spm: The number of samples per minute
        :param MARGIN_PERCENT: The margin of error in the percentage of samples per minute
        """
        # this accounts for missing data samples and gives some flexibility in the calculations.
        # The assumption here is that even if data is missing it will only be individual samples and not big chunks of data.
        
        if 1*(1 - MARGIN_PERCENT/100) < spm < 1*(1 + MARGIN_PERCENT/100):
            return TemporalResolution.MINUTE
        elif 60*(1 - MARGIN_PERCENT/100) < spm < 60*(1 + MARGIN_PERCENT/100):
            return TemporalResolution.HOUR
        elif 60*24*(1 - MARGIN_PERCENT/100) < spm < 60*24*(1 + MARGIN_PERCENT/100):
            return TemporalResolution.DAY
        elif 60*24*7*(1 - MARGIN_PERCENT/100) < spm < 60*24*7*(1 + MARGIN_PERCENT/100):
            return TemporalResolution.WEEK
        else:
            raise ValueError('Data resolution could not be determined. Please check the data and try again.')

    def _infer_data_resolution(self, df:pd.DataFrame=None) -> TemporalResolution:
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
        train_df, test_df = split_data_frame(self.df, fraction)

        train_data = CandleData(self.resolution, self.source)
        test_data = CandleData(self.resolution, self.source)
        
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
            fold = CandleData(self.resolution, self.source)
            split_idx = round( ((i+1)/k) * data.shape[0])
            fold.add_data(data.iloc[last_split_idx::i*split_idx], self.tickers)
            folds.append(fold)
        return folds


def split_data_frame(df:pd.DataFrame, fraction:float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns two dataframes split into two fractions as a tuple.
    
    :param df: The dataframe to split
    :param fraction: The fraction of the dataframe to split off
    """
    
    assert fraction < 1, 'Fraction must be less than 1'
    
    split_idx = round(fraction * df.shape[0])
    df1, df2 = df.iloc[0:split_idx], df.iloc[split_idx::]
    return df1, df2


def clean_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """This function formats the dataframe according to the assets that are in it.
    
    Needs to be updated to handle multiple assets. Note that this should only be used before high low open are stripped from the data.
    
    :param df: The dataframe to format
    """
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
