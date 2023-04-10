
import matplotlib.pyplot as plt
import datetime
import hyperopt
import logging

from parallelized_algorithmic_trader.data_management.data import get_candle_data
from parallelized_algorithmic_trader.strategy import StrategyConfig                                                         # algorithm
from parallelized_algorithmic_trader.backtest import build_features, set_train_test_true, run_simulation_on_candle_data     # simulation 
import parallelized_algorithmic_trader.indicators as indicators                                                             # feature construction
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping                                    # feature construction
from parallelized_algorithmic_trader.trading.simulated_broker import TemporalResolution                                                     # tools for knowing current position

from examples.strategies.two_ema_cross import TwoEMACross                                   # algorithm
from parallelized_algorithmic_trader.performance_analysis import get_curve_fit_vwr                                          # scoring/fitness


# set up the data to be used as global variables
n_days = 30
now = datetime.datetime.now()
res = TemporalResolution.MINUTE
ticker = 'SPY'
candle_data = get_candle_data([ticker], res, now-datetime.timedelta(days=n_days), now)


def objective_func(hyper_params, test=False) -> float:
    
    global ticker
    global candle_data
    
    print(f'Running hyperparameter optimization test with hyper_params: {hyper_params}')
    short_period, long_period = hyper_params['short_period'], hyper_params['long_period']
    indicator_mapping:IndicatorMapping = [
        IndicatorConfig(ticker+'_close', None),
        IndicatorConfig(ticker, indicators.EMA, args=(short_period,)),   # fast
        IndicatorConfig(ticker, indicators.EMA, args=(long_period,)),    # slow\
    ]

    config = StrategyConfig(
        indicator_mapping=indicator_mapping,
        strategy=TwoEMACross,
        tickers=[ticker],
        kwargs={})
    
    # set up the data
    build_features(candle_data, [indicator_mapping])
    set_train_test_true(0.8)
    
    results = run_simulation_on_candle_data(
        algorithm_configs=[config],
        use_test_data=test,
        display_progress_bar=False,
        plot=test,
        log_level=logging.INFO
        )
    
    if test:
        plt.show()
    
    return -get_curve_fit_vwr(results[0].account.get_history_as_list())


if __name__ == '__main__':
    # decent example: https://www.programcreek.com/python/?project_name=jeongyoonlee%2FKaggler#

    space = {
        "short_period": hyperopt.hp.quniform("short_period", 5, 95, 5),
        "long_period": hyperopt.hp.quniform("long_period", 100, 240, 5),
    }
    
    # example space:
    #     space = {
    #     "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    #     "max_depth": hp.choice("num_leaves", [6, 8, 10]),
    #     "colsample_bytree": hp.quniform("colsample_bytree", .5, .9, 0.1),
    #     "subsample": hp.quniform("subsample", .5, .9, 0.1),
    #     "min_child_weight": hp.choice('min_child_weight', [10, 25, 100]),
    # }

    # minimize the objective over the space
    from hyperopt import fmin, tpe
    
    best = fmin(objective_func, space, algo=tpe.suggest, max_evals=20)

    print(f'\nBest hyperparams: {best}. Running simulation with best ')
    objective_func(best, test=True)
    
    
    
    