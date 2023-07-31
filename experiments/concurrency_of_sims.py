import concurrent.futures
import matplotlib.pyplot as plt
import datetime
import logging
import click
import time
import matplotlib.pyplot as plt

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
    start = end-datetime.timedelta(weeks=52*10)
    candle_data = get_candle_data([ticker], res, start, end)


    indicator_mapping:IndicatorMapping = [
        IndicatorConfig(ticker + '_close', None),
        IndicatorConfig(ticker, indicators.EMA, args=(40,)), # fast
        IndicatorConfig(ticker, indicators.EMA, args=(40*4,)), # slow
    ]

    config = StrategyConfig(
        indicator_mapping=indicator_mapping,
        strategy=TwoEMACross,
        tickers=[ticker],
        kwargs={})

    n_algos = 256
    kwargs = {
        'market_data':candle_data,
        'algorithm_configs':[config],
    }
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    times = []
    for algo_batch_size in batch_sizes:
        t0 = time.time()
        n_futures = int(n_algos / algo_batch_size)
        print(f'Running {n_futures} sims with {algo_batch_size} algos per sim')
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(backtest, {'market_data':candle_data, 'algorithm_configs':[config]*algo_batch_size}) for _ in range(n_futures)]
            results = [f.result() for f in futures]
        
        
        
        t1 = time.time()
        print(f'Backtest with {algo_batch_size} algos per sim and {n_futures} sims took {t1-t0:.3f} seconds\n')
        times.append(t1-t0)
    # results[0].save_to_file('sample_for_testing.pkl')

    print(f'The best time and batch were {min(times):.4} and {batch_sizes[times.index(min(times))]}')
    # plot the results
    fig, ax = plt.subplots()
    ax.set_xlabel('Algos per sim')
    ax.set_ylabel('Time (s)')
    ax.plot(batch_sizes, times)
    plt.show()


if __name__ == '__main__':
    main()