import datetime
import matplotlib.pyplot as plt

from parallelized_algorithmic_trader.data_management.data import get_candle_data
from parallelized_algorithmic_trader.backtest import build_features
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
import parallelized_algorithmic_trader.performance_analysis as pf_anal
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
import parallelized_algorithmic_trader.indicators as indicators
import parallelized_algorithmic_trader.visualizations as viz


if __name__ == '__main__':
    import os
    NEAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'neat_config.txt')

    tickers = ['SPY']

    indicator_mapping:IndicatorMapping= []
    for t in tickers:
        new_inds = [
            IndicatorConfig(t+'_close', indicators.MACD, args=[12, 26, 9]),
            IndicatorConfig(t+'_close', indicators.RSI, scaling_factor=0.05, bias=-50),
            IndicatorConfig(t+'_close', indicators.BB, args=(60,)),
        ]
        indicator_mapping.extend(new_inds)
        break   # only do it for the first ticker for now

    # set up the data to be used as global variables
    start = datetime.datetime.now() - datetime.timedelta(weeks=52)
    
    candle_data = get_candle_data(tickers, TemporalResolution.HOUR, start)
    data = build_features(candle_data, [indicator_mapping])

    # viz.plot_backtest_results(data.df, tickers=candle_data.tickers)
    viz.plot_price_history(data.df, tickers=candle_data.tickers)
    
    plt.show()
    