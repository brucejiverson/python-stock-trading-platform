import pandas as pd
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]


# Analyze the S&P500 data by sector

# how many stocks are there for each sector
stocks_by_sector = df.groupby('GICS Sector')['Symbol'].count().to_dict()
print("Number of stocks by sector:")
print(stocks_by_sector)


import datetime
import os

# systems from this repo
from parallelized_algorithmic_trader.broker import TemporalResolution
import parallelized_algorithmic_trader.data_management.polygon_io as po


# for each stock in the S&P500, get the price history for the last 10 years for day resolution
end = datetime.datetime.now()
start = end-datetime.timedelta(days=365*2-2)
res = TemporalResolution.DAY

# get a list of the stocks sorted alphabetically
stocks = df['Symbol'].sort_values().tolist()


for stock in stocks:
    candle_data = po.get_candle_data(os.environ['POLYGON_IO'], [stock], start, end, res)
    
    
    