from __future__ import annotations
import matplotlib.pyplot as plt
import datetime
import os

# systems from this repo
from parallelized_algorithmic_trader.strategy import StrategyConfig
from parallelized_algorithmic_trader.backtest import backtest
import parallelized_algorithmic_trader.indicators as indicators
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
from parallelized_algorithmic_trader.broker import TemporalResolution

# the example strategy to run
from parallelized_algorithmic_trader.examples.strategies.two_ema_cross import TwoEMACross
import parallelized_algorithmic_trader.polygon_io as po


import logging
root_logger = logging.getLogger('pat')
root_logger.setLevel(logging.DEBUG)


def get_polygon_key():
    """Get Polygon API key from file."""
    f_path = os.path.join(os.path.dirname(__file__), "../../polygon_key.txt")
    # logger.info(F'install path: {f_path}')
    with open(f_path, 'r') as f:
        key = f.read()
    return key


n_days = 5
end = datetime.datetime.now()
res = TemporalResolution.MINUTE
start = end-datetime.timedelta(days=n_days)
candle_data = po.get_candle_data(get_polygon_key(), ['SPY'], start, end, res)


indicator_mapping:IndicatorMapping = [
    IndicatorConfig('SPY', indicators.EMA, args=(30,)), # fast
    IndicatorConfig('SPY', indicators.EMA, args=(90,)), # slow
]

config = StrategyConfig(
    indicator_mapping=indicator_mapping,
    strategy=TwoEMACross,
    tickers=['SPY'],
    kwargs={},
    quantity=1
    )


import logging
results = backtest(
    market_data=candle_data,
    algorithm_configs=[config],
    verbose=True,
    plot=True,
    timeit=True,
    log_level=logging.INFO
    )

plt.show()

