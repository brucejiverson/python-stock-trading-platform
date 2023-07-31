import concurrent.futures
import matplotlib.pyplot as plt
import datetime
import logging
import click
import time

# systems from this repo
from parallelized_algorithmic_trader.strategy import StrategyConfig
from parallelized_algorithmic_trader.backtest import *
import parallelized_algorithmic_trader.indicators as indicators
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.data_management.data import get_candle_data

# the example strategy to run
from examples.strategies.two_ema_cross import TwoEMACross


def run_simulation_on_candle_data_concurrent(
    algorithm_configs:List[StrategyConfig],
    slippage_model:SlippageModelsMarketOrders|None=None,
    use_test_data:bool = False,
    plot:bool=False,
    folder_to_save_plots:str|None=None,
    display_progress_bar:bool=True,
    log_level=logging.WARNING) -> TradingHistorySet:

    log_level = min(log_level, root_logger.getEffectiveLevel())
    logger.setLevel(log_level)

    # check if all the configs have the same state expectation. If any are unique, then the environment state will be reconstructed for each strategy. 
    # this is useful for things like genetic algorithms where the state will be the same for each bot in the population
    global compute_state_for_each_strategy
    if compute_state_for_each_strategy is None:
        compute_state_for_each_strategy = check_if_all_configs_have_the_same_state_expectation(algorithm_configs)
        logger.debug('compute_state_for_each_strategy: ' + str(compute_state_for_each_strategy))
            
    data = get_dataset(use_test_data)
    # print out the start and end dates
    logger.info(f'Backtesting from {data.start_date} to {data.end_date}')
    brokerage = SimulatedStockBrokerCandleData(data, market_order_slippage_model=slippage_model)

    # set up the algorithms
    algorithms:StrategySet = {}
    # Instantiates a strategy to the brain and adds a matching account to the brokerage
    accounts = brokerage.set_up_accounts(len(algorithm_configs))
    for algo_config, account_num in zip(algorithm_configs, accounts.keys()):
        algorithms[account_num] = algo_config.instantiate(account_num)     

    def step_one_algo(acc, alg, st):
        new_orders = alg.act(acc, st)
        if new_orders is not None:
            acc.submit_new_orders(new_orders)
                
    # Loop through the price history facilitating interaction between the algorithms and the brokerage
    # backtest with itterrows took: 20.058 s, using while loop took 19.1 s holding everything else constant
    obs = brokerage.reset()
    new_orders = []
    n_time_steps = len(data.df)
    i = 0
    for obs, reward, done, info in brokerage.step():
        state = None
        for account_num in accounts.keys():
            account, algo = accounts[account_num], algorithms[account_num]
            # generate new orders and submit them to the brokerage
            """This is broken out as a major efficiency upgrade for genetic algorithms where you need the same state hundreds of times for your population. Found that when state was 
            gotten for each strategy, backtesting for 1 day slowed from ~0.75 seconds to 20 seconds."""
            if compute_state_for_each_strategy or state is None:
                logger.debug(f'Getting state for {algo.indicator_mapping}')
                state = get_state_from_obs(algo.indicator_mapping, obs)
            
            # create futures for each agent
            with concurrent.futures.ThreadPoolExecutor() as executor:
                new_orders.append(executor.submit(step_one_algo, account, algo, state))    

        # wait for all the agents to finish
        concurrent.futures.wait(new_orders)
        if i%100 == 0 and display_progress_bar:
            printProgressBar(i, n_time_steps)
        i += 1
    print('')
            
    # catalog the outputs from the simulation
    results:TradingHistorySet = []
    for account, strategy in zip(accounts.values(), algorithms.values()):
        res = TradingHistory(strategy, account, data.tickers, data.resolution)
        results.append(res)
    
    if plot:
        # find the best performing agent
        best_result = get_best_strategy_and_account(results)
        best_strategy, best_account = best_result.strategy, best_result.account
        if len(results) > 1:
            logger.info(f'Printing stats for the best performing strategy only: {best_strategy}')

        performance_metrics = print_account_stats(
            best_account, 
            data, 
            brokerage.spread_percent, 
            brokerage._market_order_slippage, 
            brokerage.limit_order_slippage)
        
        create_pdf_performance_report(
            performance_metrics, 
            data.df,
            best_account, 
            data.tickers, 
            type(best_strategy).__name__, 
            folder_to_save_plots)
        
        plt.show()
    
    return results


def backtest_concurrent(
    market_data:CandleData,
    algorithm_configs:List[StrategyConfig],
    scaler:IndicatorConfig=None,
    slippage_model:SlippageModelsMarketOrders=None,
    plot:bool = False,
    folder_to_save_plots=None,
    display_progress_bar:bool=True,
    log_level:int = logging.WARNING) -> TradingHistorySet:
    """Backtest a set of algorithms on a set of market data

    Args:
        market_data (CandleData): [description]
        algorithm_configs (List[StrategyConfig]): [description]
        scaler (IndicatorConfig, optional): [description]. Defaults to None.
        SlippageModel (SlippageModelsMarketOrders, optional): [description]. Defaults to None.
        plot (bool, optional): [description]. Defaults to False.
        log_level (int, optional): [description]. Defaults to logging.WARNING.

    Returns:
        TradingHistorySet: [description]
    """

    # build the features
    build_features(market_data, [s.indicator_mapping for s in algorithm_configs])
    global train_test_split_flag
    train_test_split_flag = False
    
    return run_simulation_on_candle_data_concurrent(
        algorithm_configs, 
        slippage_model,
        plot=plot,
        folder_to_save_plots=folder_to_save_plots,
        display_progress_bar=display_progress_bar,
        log_level=log_level)


@click.command()
@click.option('--log-level', '-l', default=logging.WARNING, help='Logging level')
def main(log_level:int):
    # get the root logger and set to debug
    rl = logging.getLogger('pat')
    rl.setLevel(log_level)

    ticker = 'SPY'
    col_name = ticker + '_close'
    end = datetime.datetime.now()
    res = TemporalResolution.DAY
    start = end-datetime.timedelta(weeks=52*10)
    candle_data = get_candle_data([ticker], res, start, end)

    indicator_mapping:IndicatorMapping = [
        IndicatorConfig(col_name, None),
        IndicatorConfig(col_name, indicators.EMA, args=(40,)), # fast
        IndicatorConfig(col_name, indicators.EMA, args=(40*4,)), # slow
    ]

    algo_config = StrategyConfig(
        indicator_mapping=indicator_mapping,
        strategy=TwoEMACross,
        tickers=[ticker],
        kwargs={})

    t0 = time.time()
    backtest(
        market_data=candle_data,
        algorithm_configs=[algo_config]*256,
        plot=True,
        folder_to_save_plots='tmp'
        )
    t1 = time.time()
    print(f'Standard backtest took {t1-t0:.3f} seconds')
    
    t0 = time.time()
    backtest_concurrent(
        market_data=candle_data,
        algorithm_configs=[algo_config]*256,
        plot=True,
        folder_to_save_plots='tmp'
        )
    t1 = time.time()
    print(f'Backtest with all agent step calculations concurrent took {t1-t0:.3f} seconds')


if __name__ == '__main__':
    main()

