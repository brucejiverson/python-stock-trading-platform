import matplotlib.pyplot as plt
import datetime
import os
import logging

# systems from this repo
from parallelized_algorithmic_trader.strategy import StrategyConfig
from parallelized_algorithmic_trader.backtest import backtest
import parallelized_algorithmic_trader.indicators as indicators
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
from parallelized_algorithmic_trader.broker import TemporalResolution
import parallelized_algorithmic_trader.data_management.polygon_io as po

# the example strategy to run
from examples.strategies.two_ema_cross import TwoEMACross


# get the root logger and set to debug
rl = logging.getLogger('pat')
lvl = logging.INFO
rl.setLevel(lvl)


ticker = 'SPY'
end = datetime.datetime.now()
res = TemporalResolution.HOUR
start = end-datetime.timedelta(weeks=4*24)
candle_data = po.get_candle_data(os.environ['POLYGON_IO'], [ticker], start, end, res)


indicator_mapping:IndicatorMapping = [
    IndicatorConfig(ticker, indicators.EMA, args=(30,)), # fast
    IndicatorConfig(ticker, indicators.EMA, args=(90,)), # slow
]

config = StrategyConfig(
    indicator_mapping=indicator_mapping,
    strategy=TwoEMACross,
    tickers=[ticker],
    kwargs={'log_level': lvl},
    quantity=1
    )

results = backtest(
    market_data=candle_data,
    algorithm_configs=[config],
    verbose=True,
    plot=True,
    log_level=lvl
    )

# results[0].save_to_file('sample_for_testing.pkl')

plt.show()

