import datetime
import pandas as pd
import numpy as np
from enum import Enum

from parallelized_algorithmic_trader.util import get_logger, DateRange


logger = get_logger(__name__)


def check_if_date_in_holidays(dt:datetime.datetime) -> bool:
    """Checks to see if the month and day of the Datetime object matches any in the HOLIDAYS list"""
        
    # A list with holidays in this format: {'month': 1, 'day': 1}, use this format for holidays
    HOLIDAYS = [
        {'month': 1, 'day': 1},
        {'month': 1, 'day': 18},
        {'month': 2, 'day': 15},
        {'month': 5, 'day': 31},
        {'month': 7, 'day': 5},
        {'month': 9, 'day': 6},
        {'month': 10, 'day': 11},
        {'month': 11, 'day': 11},
        {'month': 11, 'day': 25},
        {'month': 12, 'day': 25},
        {'month': 12, 'day': 31},
    ]

    for holiday in HOLIDAYS:
        if dt.month == holiday['month'] and dt.day == holiday['day']:
            return True
    return False


class TemporalResolution(Enum):
    """This class represents units of time."""
    MINUTE = "minute"
    FIVE_MINUTE = "5_minute"
    FIFTEEN_MINUTE = "15_minute"
    THIRTY_MINUTE = "30_minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

    def get_as_minutes(self):
        if self == TemporalResolution.MINUTE:
            return 1
        elif self == TemporalResolution.FIVE_MINUTE:
            return 5
        elif self == TemporalResolution.FIFTEEN_MINUTE:
            return 15
        elif self == TemporalResolution.THIRTY_MINUTE:
            return 30
        elif self == TemporalResolution.HOUR:
            return 60
        elif self == TemporalResolution.DAY:
            return 1440
        elif self == TemporalResolution.WEEK:
            return 10080
        elif self == TemporalResolution.MONTH:
            return 43200
        elif self == TemporalResolution.YEAR:
            return 525600
        else:
            raise ValueError("Unknown resolution")


def build_data_file_name(ticker:str, resolution:TemporalResolution) -> str:
    """A helper function for getting the name of the file that the data for a ticker
    
    :param ticker: The ticker of the equity
    :resolution: The time per candle
    """
    # f'stocks/{resolution.name.lower()}/{ticker}.pkl'
    # return formatted string
    return "stocks/{}/{}.pkl".format(resolution.name.lower(), ticker)

 
def floor_to_preceding_business_day(dt:datetime.datetime) -> datetime.datetime:
    """Calculate the preceding business day to the given date. If the date is a weekend, return the preceding Friday.
    If the date is a holiday, return the preceding business day.
    """
    
    # truncate to the minute
    check_date = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    
    # if given datetime.datetime is greater than now, floor to now
    if check_date > datetime.datetime.now():
        check_date = datetime.datetime.now()
        
    # if the date matches today subtract one day
    if check_date.date() == datetime.datetime.now().date():
        check_date = check_date - datetime.timedelta(days=2)
    
    # if the time is after 4pm, floor to 4pm
    if check_date.time() > datetime.datetime(2020, 1, 1, 16, 0, 0).time():
        check_date = datetime.datetime(check_date.year, check_date.month, check_date.day, 16, 0, 0)
          
    while check_if_date_in_holidays(check_date):
        check_date += datetime.timedelta(days=1)
    
    # if the date is a weekend, return the preceding Friday with the same given time
    if check_date.weekday() == 5: # Saturday
        return check_date - datetime.timedelta(days=1)
    elif check_date.weekday() == 6: # Sunday
        return check_date - datetime.timedelta(days=2)
    
    return check_date


def ceiling_to_subsequent_business_date(dt:datetime.datetime) -> datetime.datetime:
    """Calculate the preceding business day to the given date. If the date is a weekend, return the preceding Friday.
    If the date is a holiday, return the preceding business day.
    """
    
    # truncate to the minute
    check_date = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    
    # if after 4, add a day and set time to 9am
    if check_date.time() > datetime.datetime(2020, 1, 1, 16, 0, 0).time():
        check_date = datetime.datetime(check_date.year, check_date.month, check_date.day, 9, 0, 0) + datetime.timedelta(days=1)
    
    # if the time is before 9am, floor to 9am
    if check_date.time() < datetime.datetime(2020, 1, 1, 9, 0, 0).time():
        check_date = datetime.datetime(check_date.year, check_date.month, check_date.day, 9, 0, 0)
          
    while check_if_date_in_holidays(check_date):
        check_date += datetime.timedelta(days=1)
        
    # if the date is a weekend, return the preceding Friday with the same given time
    if check_date.weekday() == 5:       # Saturday
        return check_date + datetime.timedelta(days=2)
    elif check_date.weekday() == 6:     # Sunday
        return check_date + datetime.timedelta(days=1)
    
    return check_date


def check_if_data_covers_timerange(df:pd.DataFrame, time_range:DateRange) -> bool:
    """Checks to see if the data is already in a file. If it is, then it checks to see if the data covers the desired date range. If it does, then it returns the data.
    
    Notes that this does truncate the data to the desired date range.
    
    Parameters:
        df: pd.DataFrame
        time_range: DateRange describing the span of time anchored in absolute time.
        
    Returns:
        bool: Indicates if the data covers the desired date range.
    """
    
    # check to see if the data covers the desired date range.
    if isinstance(df.index, pd.DatetimeIndex):
        data_time_range = DateRange(df.index[0], df.index[-1])
    else:
        data_time_range = DateRange(df['timestamp'].iloc[0], df['timestamp'].iloc[-1])
        
    buffer = datetime.timedelta(days=0, hours=2)
    start_sufficiently_early = data_time_range.start <= time_range.start + buffer
    end_sufficiently_recent = data_time_range.end >= time_range.end - buffer
    
    return start_sufficiently_early and end_sufficiently_recent


def sanitize_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """This function formats the dataframe according to the assets that are in it.
    
    Needs to be updated to handle multiple assets. Note that this should only be used before high low open are stripped from the data.
    
    :param df: The dataframe to format
    """
    formatted_df = df.copy()
    
    # check to see if the dataframe has a timestamp column and if so make the index the timestamp
    if not isinstance(formatted_df.index, pd.DatetimeIndex):
        assert 'timestamp' in formatted_df.columns, f'Dataframe must have a column named timestamp or the index must be a DatetimeIndex. Head: \n {formatted_df.head()}'
        # drop any duplicate timestamps
        formatted_df.drop_duplicates(subset='timestamp', inplace=True)
        formatted_df.set_index('timestamp', inplace=True, verify_integrity=True, drop=True)
    # elif 'timestamp' in formatted_df.columns:
    #     formatted_df.drop('timestamp', axis=1, inplace=True)
    
    # sort by index
    formatted_df.sort_index(inplace=True)
    if formatted_df.index.name != 'timestamp':
        formatted_df.index.name = 'timestamp'

    # drop the last line. polygon.io does this sometimes, not sure why. ignoring for now
    if pd.isnull(formatted_df.index[-1]):
        formatted_df = formatted_df[:-1]

    # get the number of duplicates in the dataframe
    logger.debug(f'Length and start date before drop duplicates: {len(formatted_df)}, {formatted_df.index[0]}')
    formatted_df = formatted_df.loc[~formatted_df.index.duplicated(keep = 'first')]     # this is intended to remove duplicates. ~ flips bits in the mask
    
    # Replacing infinite with nan
    logger.debug(f'Length and start date before replace inf, nan: {len(formatted_df)}, {formatted_df.index[0]}')
    formatted_df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    logger.debug(f'Length and start date before dropna: {len(formatted_df)}, {formatted_df.index[0]}')
    formatted_df.dropna(inplace=True)
    logger.debug(f'Length and start date after dropna: {len(formatted_df)}, {formatted_df.index[0]}')

    # # search for inconsistent jumps in the timestamps of the data frame
    # for i in range(1, len(formatted_df)):
    #     if formatted_df.index[i] - formatted_df.index[i-1] != datetime.timedelta(minutes=1):
    #         print('Inconsistent time stamps found at index: {}'.format(i))
    #         print('Timestamps: {} and {}'.format(formatted_df.index[i-1], formatted_df.index[i]))
    #         print('Difference: {}'.format(formatted_df.index[i] - formatted_df.index[i-1]))
    #         print('Dropping the inconsistent timestamps')
    #         formatted_df.drop(formatted_df.index[i], inplace = True)

    return formatted_df


def filter_df_for_daterange(df:pd.DataFrame, time_range:DateRange) -> pd.DataFrame:
    """Filters the dataframe to the desired date range.
    
    Parameters:
        df: pd.DataFrame
        time_range: DateRange describing the span of time anchored in absolute time.
        
    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    
    assert isinstance(df.index, pd.DatetimeIndex) or 'timestamp' in df.columns, f'Dataframe must have a column named timestamp or the index must be a DatetimeIndex. Head: \n {df.head()}'
    
    logger.debug(f'Data dates before time filter: {df.index[0]} to {df.index[-1]}')
    df = df[df.index >= time_range.start]
    df = df[df.index <= time_range.end]
    logger.debug(f'Data dates after time filter: {df.index[0]} to {df.index[-1]}')
    return df


def compress_candle_data(candle_data: pd.DataFrame, resolution: TemporalResolution) -> pd.DataFrame:
    """
    Compress financial candle data from one temporal resolution to a higher one.

    Args:
        candle_data (pd.DataFrame): A pandas DataFrame containing financial candle data with a datetimeindex.
        resolution (TemporalResolution): A TemporalResolution object representing the desired resolution.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the compressed data.
    """
    logger.debug(f"Compressing candle data with resolution {resolution.name}")
    logger.debug(f'Original candle data size: {candle_data.shape[0]}')
    
    data_as_minutes = candle_data.resample(f'{resolution.get_as_minutes()}T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    })

    compressed_data_size = data_as_minutes.shape[0]
    logger.debug(f"Compressed candle data size: {compressed_data_size}")

    return data_as_minutes.dropna()
