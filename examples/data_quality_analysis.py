import datetime
import logging

from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.util import get_logger
from parallelized_algorithmic_trader.data_management.data import get_candle_data


logger = get_logger('path.neat')
root_logger = get_logger('path')
root_logger.setLevel(logging.DEBUG)

n_days = 365

end = datetime.datetime(2022, 10, 13)
tickers = ['SPDN', 'SPY']
tickers = ['SPDN']
candle_data = get_candle_data(tickers, TemporalResolution.MINUTE, end-datetime.timedelta(days=n_days))

print(f'There are {len(candle_data.df)} candles in the data')
print(f'The total shape of the data is {candle_data.df.shape}')
dt = candle_data.df.index[-1] - candle_data.df.index[0]
print(f'The length of the time in the df is {dt.days} days')
print(f'The data points per day is {len(candle_data.df)/dt.days:.2f}')
print(f'There are: {len(candle_data.df)/(60*dt.days):.2f} data points per day/60')
assumed_n_trading_days = 252
print(f'There are: {len(candle_data.df)/(assumed_n_trading_days):.2f} data points per trading day')
print(f'There are: {len(candle_data.df)/(assumed_n_trading_days*60):.2f} data points per trading day/60')
# print(f'There are: {len(candle_data.df)/(60*dt.days):.2f} data points per day/60')

