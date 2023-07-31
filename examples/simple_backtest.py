import datetime
import logging
import click
import time
import matplotlib.pyplot as plt

# systems from this repo
from parallelized_algorithmic_trader.strategy import StrategyConfig
from parallelized_algorithmic_trader.backtest import *
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
    logger = logging.getLogger('pat.'+__name__)
    logger.setLevel(log_level)
    print(logger.name)

    ticker = 'SPY'
    end = datetime.datetime.now()
    res = TemporalResolution.MINUTE
    start = end-datetime.timedelta(weeks=52*1)
    candle_data = get_candle_data([ticker], res, start, end)

    col_name = ticker + '_close'
    indicator_mapping = IndicatorMapping(
        # IndicatorConfig(col_name, None),s
        IndicatorConfig(col_name, indicators.EMA, args=(50*60,)), # fast
        IndicatorConfig(col_name, indicators.EMA, args=(200*60,)), # slow
    )

    config = StrategyConfig(
        indicator_mapping=indicator_mapping,
        strategy=TwoEMACross,
        tickers=[ticker],
        kwargs={})

    build_features(candle_data, [s.indicator_mapping for s in [config]])
    global train_test_split_flag
    train_test_split_flag = False
    
    logger.info(f'Running to initialize...')
    run_simulation_on_candle_data(
        [config], 
        log_level=logging.WARNING)
    t0 = time.time()
    run_simulation_on_candle_data(
        [config], 
        log_level=log_level)
    logger.info(f'Full simulation time in simple_backtest {time.time()-t0:.3f} seconds')
    
    logger.info(f'Running to plot...')
    run_simulation_on_candle_data(
        [config], 
        plot=True,
        log_level=log_level)
    
    # t0 = time.time()
    # run_simulation_on_candle_data_old(
    #     [config], 
    #     None,
    #     log_level=log_level)
    # t1 = time.time()
    # print(f'Old backtest took {t1-t0:.3f} seconds')

    # results[0].save_to_file('sample_for_testing.pkl')
    plt.show()


if __name__ == '__main__':
    main()
    
    