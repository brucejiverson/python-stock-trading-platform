from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from parallelized_algorithmic_trader.util import DateRange


HOLIDAYS = [
    datetime(2020, 1, 1).date(),
    datetime(2020, 1, 20).date(),
    datetime(2020, 2, 17).date(),
    datetime(2020, 5, 25).date(),
    datetime(2020, 7, 3).date(),
    datetime(2020, 9, 7).date(),
    datetime(2020, 10, 12).date(),
    datetime(2020, 11, 11).date(),
    datetime(2020, 11, 26).date(),
    datetime(2020, 12, 25).date(),
]


def floor_to_preceding_business_day(dt:datetime) -> datetime:
    """Calculate the preceding business day to the given date. If the date is a weekend, return the preceding Friday.
    If the date is a holiday, return the preceding business day.
    """
    
    # truncate to the minute
    check_date = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    
    # if given datetime is greater than now, floor to now
    if check_date > datetime.now():
        check_date = datetime.now()
        
    # if the date matches today subtract one day
    if check_date.date() == datetime.now().date():
        check_date = check_date - timedelta(days=2)
    
    # if the time is after 4pm, floor to 4pm
    if check_date.time() > datetime(2020, 1, 1, 16, 0, 0).time():
        check_date = datetime(check_date.year, check_date.month, check_date.day, 16, 0, 0)
          
    global HOLIDAYS
    if dt.date() in HOLIDAYS:
        check_date = dt - timedelta(days=1)
    
    # if the date is a weekend, return the preceding Friday with the same given time
    if check_date.weekday() == 5: # Saturday
        return check_date - timedelta(days=1)
    elif check_date.weekday() == 6: # Sunday
        return check_date - timedelta(days=2)
    
    return check_date


def ceiling_to_subsequent_business_date(dt:datetime) -> datetime:
    """Calculate the preceding business day to the given date. If the date is a weekend, return the preceding Friday.
    If the date is a holiday, return the preceding business day.
    """
    
    # truncate to the minute
    check_date = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    
    # if after 4, add a day and set time to 9am
    if check_date.time() > datetime(2020, 1, 1, 16, 0, 0).time():
        check_date = datetime(check_date.year, check_date.month, check_date.day, 9, 0, 0) + timedelta(days=1)
    
    # if the time is before 9am, floor to 9am
    if check_date.time() < datetime(2020, 1, 1, 9, 0, 0).time():
        check_date = datetime(check_date.year, check_date.month, check_date.day, 9, 0, 0)
          
    global HOLIDAYS
    if dt.date() in HOLIDAYS:
        check_date = dt + timedelta(days=1)
    
    # if the date is a weekend, return the preceding Friday with the same given time
    if check_date.weekday() == 5: # Saturday
        return check_date + timedelta(days=2)
    elif check_date.weekday() == 6: # Sunday
        return check_date + timedelta(days=1)
    
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
    if type(df.index) == pd.DatetimeIndex:
        data_time_range = DateRange(df.index[0], df.index[-1])
    else:
        data_time_range = DateRange(df['timestamp'].iloc[0], df['timestamp'].iloc[-1])
        
    buffer = timedelta(days=0, hours=2)
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
    if type(formatted_df.index) != pd.DatetimeIndex:
        assert 'timestamp' in formatted_df.columns, 'Dataframe must have a column named timestamp or the index must be a DatetimeIndex.'
        # drop any duplicate timestamps
        formatted_df.drop_duplicates(subset='timestamp', inplace=True)
        formatted_df.set_index('timestamp', inplace=True, verify_integrity=True, drop=True)
    elif 'timestamp' in formatted_df.columns:
        formatted_df.drop('timestamp', axis=1, inplace=True)
    
    # sort by index
    formatted_df.sort_index(inplace=True)

    # drop the last line. polygon.io does this sometimes, not sure why. ignoring for now
    if pd.isnull(formatted_df.index[-1]):
        formatted_df = formatted_df[:-1]

    # get the number of duplicates in the dataframe
    formatted_df = formatted_df.loc[~formatted_df.index.duplicated(keep = 'first')]     # this is intended to remove duplicates. ~ flips bits in the mask
    
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
