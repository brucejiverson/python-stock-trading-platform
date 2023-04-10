import matplotlib.pyplot as plt
import datetime
import logging
import click

# systems from this repo
from parallelized_algorithmic_trader.strategy import StrategyConfig
from parallelized_algorithmic_trader.backtest import backtest
import parallelized_algorithmic_trader.indicators as indicators
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.data_management.data import get_candle_data

# the example strategy to run
from examples.strategies.two_ema_cross import TwoEMACross


@click.command()
@click.option('--log-level', '-l', default=logging.INFO, help='Logging level')
def main(log_level:int):
    # get the root logger and set to debug
    rl = logging.getLogger('pat')
    rl.setLevel(log_level)

    ticker = 'SPY'
    end = datetime.datetime.now()
    res = TemporalResolution.HOUR
    start = end-datetime.timedelta(weeks=52*1)
    candle_data = get_candle_data([ticker], res, start, end)


    indicator_mapping:IndicatorMapping = [
        IndicatorConfig(ticker + '_close', None),
        IndicatorConfig(ticker, indicators.EMA, args=(10,)), # fast
        IndicatorConfig(ticker, indicators.EMA, args=(30,)), # slow
    ]

    config = StrategyConfig(
        indicator_mapping=indicator_mapping,
        strategy=TwoEMACross,
        tickers=[ticker],
        kwargs={})

    import time
    t0 = time.time()
    backtest(
        market_data=candle_data,
        algorithm_configs=[config],
        plot=True,
        folder_to_save_plots='tmp'
        )
    t1 = time.time()
    print(f'Backtest took {t1-t0:.3f} seconds')

    # results[0].save_to_file('sample_for_testing.pkl')

    plt.show()


if __name__ == '__main__':
    main()