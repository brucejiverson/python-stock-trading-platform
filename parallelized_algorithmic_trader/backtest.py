from typing import List, Dict, Tuple, Any
import logging
import uuid
import pandas as pd
import matplotlib.pyplot as plt

from parallelized_algorithmic_trader.indicators import IndicatorMapping, aggregate_indicator_mappings, check_if_indicator_mappings_identical
from parallelized_algorithmic_trader.strategy import StrategySet, StrategyConfig
from parallelized_algorithmic_trader.data_management.data import CandleData
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.trading.simulated_broker import *
from parallelized_algorithmic_trader.performance_analysis import print_account_stats, get_best_strategy_and_account
from parallelized_algorithmic_trader.visualizations import *
from parallelized_algorithmic_trader.util import printProgressBar, get_logger


logger = get_logger(__name__ )
import logging
root_logger = logging.getLogger('pat')

# this is global so that we can make it once and then resuse it when doing consecutive backtests
main_data:CandleData = None         # all of the data including candles and any features or indicators

train_test_split_flag = False
train_test_split_index:int = None

train_data:CandleData = None
test_data:CandleData = None
compute_state_for_each_strategy:bool = None


def iter_df_subset(df:pd.DataFrame, start_i:int, end_i:int=None) -> Tuple[Any, pd.Series]:
    """Iterates over a subset of a dataframe.
    
    Parameters:
        df (pd.DataFrame): The dataframe to iterate over
        start_i (int): The positional index to start iterating at
        end_i (int|None): The positional index to stop iterating at. If None, will iterate to the end of the dataframe  
    
    Yields:
        Tuple[Any, pd.Series]: The index and the row of the dataframe
    """
    df_len = len(df)

    if end_i is None:
        end_i = df_len
    
    for i in range(start_i, end_i):
        # check the index to see if it is still in bounds
        if i >= df_len:
            break
        
        yield df.index[i], df.iloc[i]
        

def get_training_start_end_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
    if train_test_split_flag:
        return train_data.start_date, train_data.end_date
    else:
        return main_data.start_date, main_data.end_date


def build_features(candle_data:CandleData, indicator_mappings:List[IndicatorMapping]) -> CandleData:
    """Builds the features for the backtest. This is a global function so that we can make it once and then resuse it when doing consecutive backtests"""
    # this is done to prevent calculation of the features for consecutive backtests (can be somewhat time consuming)
    global main_data
    
    # initialize the main data frame
    main_data_needs_update = main_data is None or main_data.df.index[0] != candle_data.start_date or main_data.df.index[-1] != candle_data.end_date
    if main_data_needs_update:
        logger.debug('Computing features for backtest, overwriting global main_df')
        # the full dataframe with all candle data that we will build features on and then loop over
        main_df = candle_data.df.copy()
        
    else: main_df = main_data.df.copy()
        
    logger.debug(f'Building features for {len(main_df)} rows of data')
    # aggregate the indicator configs from all the algorithms and add them to the main_df
    comprehensive_indicator_mapping  = aggregate_indicator_mappings(indicator_mappings)
    for indicator_config in comprehensive_indicator_mapping:
        logger.debug(f'Start date of main_df before making indicator: {main_df.index[0]} for indicator {indicator_config.config_name}')
        # if len(indicator_config.names) == 0:
        #     raise ValueError(f'Indicator config {indicator_config} has no names. ')
        
        # check if this indicator has already been built by looking for names
        if len(indicator_config.names) > 0 and all([name in main_df.columns for name in indicator_config.names]):
            logger.debug(f'Indicator {indicator_config.names} has already been built. Skipping.')
            continue
        
        logger.debug(f'Making indicator {indicator_config.config_name}')
        indicator = indicator_config.make(main_df)
        if indicator is not None:
            main_df = pd.concat([main_df, indicator], axis=1)
        logger.info(f'Built indicator {indicator_config.config_name} with features {indicator_config.names}')
        
    logger.debug(f'Columns list after indicators build: {main_df.columns}')
    logger.debug(f'Main dataframe has {len(main_df)} rows of data. Starting at {main_df.index[0]} and ending at {main_df.index[-1]}')
    
    # now set up their data objects
    logger.debug(f'Do not forget that putting the dataframe into the CandleData object will automatically clean it, removing any rows that have nan values (indicator warm up)')
    logger.debug(f'Setting up main data frame')
    main_data = CandleData(main_df, candle_data.tickers, candle_data.resolution)
    return main_data


def set_train_test_true(fraction:float=0.8):
    """Informs that the data should be split for training and testing, and sets the fraction of data to be used for training"""
    global train_test_split_flag
    train_test_split_flag = True
    global train_data
    global test_data
    train_data, test_data = main_data.split(fraction)
    
    global train_test_split_index
    train_test_split_index = round((len(train_data.df)) * fraction)


def backtest(
    market_data:CandleData,
    algorithm_configs:List[StrategyConfig],
    slippage_model:SlippageModelsMarketOrders=None,
    plot:bool = False,
    folder_to_save_plots=None,
    display_progress_bar:bool=True,
    log_level:int = logging.WARNING) -> AccountHistoryFileHandlerSet:
    """Backtest a set of algorithms on a set of market data

    Args:
        market_data (CandleData): [description]
        algorithm_configs (List[StrategyConfig]): [description]
        SlippageModel (SlippageModelsMarketOrders, optional): [description]. Defaults to None.
        plot (bool, optional): [description]. Defaults to False.
        log_level (int, optional): [description]. Defaults to logging.WARNING.

    Returns:
        AccountHistoryFileHandlerSet: [description]
    """

    # build the features
    build_features(market_data, [s.indicator_mapping for s in algorithm_configs])
    global train_test_split_flag
    train_test_split_flag = False
    
    return run_simulation_on_candle_data(
        algorithm_configs, 
        slippage_model,
        plot=plot,
        folder_to_save_plots=folder_to_save_plots,
        display_progress_bar=display_progress_bar,
        log_level=log_level)


def get_state(ind_mapping:IndicatorMapping, row:pd.Series) -> Dict[str, float]:
    feature_names = []
    for indicator_conf in ind_mapping:
        feature_names.extend(indicator_conf.names)
    d = row[feature_names].to_dict()
    d['timestamp'] = row.name
    return d


def check_if_all_configs_have_the_same_state_expectation(algorithm_configs:List[StrategyConfig]) -> bool:
    """Checks if all the configs have the same state expectation. If they do, we can compute the state once and then pass it to all the algorithms"""

    first_config = algorithm_configs[0]
    for config in algorithm_configs:
        if not check_if_indicator_mappings_identical(config.indicator_mapping, first_config.indicator_mapping):
            return False
    return True


def run_simulation_on_candle_data(
    algorithm_configs:List[StrategyConfig],
    slippage_model:SlippageModelsMarketOrders|None=None,
    use_test_data:bool = False,
    plot:bool=False,
    folder_to_save_plots:str|None=None,
    display_progress_bar:bool=True,
    log_level=logging.WARNING) -> AccountHistoryFileHandlerSet:

    logger.setLevel(log_level)

    # check if all the configs have the same state expectation. If any are unique, then the environment state will be reconstructed for each strategy. 
    # this is useful for things like genetic algorithms where the state will be the same for each bot in the population
    global compute_state_for_each_strategy
    if compute_state_for_each_strategy is None:
        compute_state_for_each_strategy = check_if_all_configs_have_the_same_state_expectation(algorithm_configs)
        logger.debug('compute_state_for_each_strategy: ' + str(compute_state_for_each_strategy))
    
    # set up the data
    global train_test_split_flag
    if train_test_split_flag:
        if use_test_data:
            logger.info(f'Using test data')
            global test_data
            data = test_data
            start_idx = train_test_split_index
            end_idx = len(data.df)
        else:
            logger.info(f'Using train data')
            global train_data
            data = train_data
            start_idx = 0
            end_idx = train_test_split_index
    else:
        logger.debug(f'Using sim data, no train test split.')
        global main_data
        data = main_data
        start_idx = 0
        end_idx = len(data.df)
        
    # print out the start and end dates
    logger.info(f'Backtesting from {data.start_date} to {data.end_date}')

    brokerage = SimulatedStockBrokerCandleData()

    # add the market data to the brokerage
    assert isinstance(data, CandleData), f'Expected data to be a CandleData object, got {type(data)}'

    brokerage.set_expected_resolution(data.resolution)
    for t in data.tickers:
        # inform the brokerage that there is a new equity and to expect incoming candle_data about it 
        brokerage.add_ticker(t)

    if slippage_model is not None:
        logger.info(f'Using slippage model: {slippage_model} as dictacted by user')
        brokerage.set_slippage_model(slippage_model)
    elif data.resolution == TemporalResolution.MINUTE:
        logger.info(f'Using minute resolution data. Slippage model is set to NEXT_CLOSE')
        brokerage.set_slippage_model(SlippageModelsMarketOrders.NEXT_CLOSE)
    else:
        logger.info(f'The slippage model has been left at the default, {brokerage.market_order_slippage_model.name}')

    # set up the algorithms
    algorithms:StrategySet = {}
    accounts:Dict[uuid.UUID, SimulatedAccount] = {}
    # Instantiates a strategy to the brain and adds a matching account to the brokerage
    for algo_config in algorithm_configs:
        account_num = uuid.uuid4()
        algorithms[account_num] = algo_config.instantiate(account_num)

        a = SimulatedAccount(account_num, 10000)
        logger.debug(f'Created account {str(account_num)[:8]} with ${a.cash:.2f}')
        accounts[account_num] = a

    n_rows = len(data.df)
    # Loop through the price history facilitating interaction between the algorithms and the brokerage
    # backtest with itterrows took: 20.058 s, using while loop took 19.1 s holding everything else constant
    # for i, (ts, row) in enumerate(iter_df_subset(data.df.iterrows(), start_idx, end_idx)):
    for i, (ts, row) in enumerate(data.df.iterrows()):
        # process pending orders from the last loop and update the value history
        brokerage.set_prices(row)
        state = None
        for account_num in accounts.keys():
            account, algo = accounts[account_num], algorithms[account_num]
            brokerage.process_orders_for_account(account)  # processes any pending orders
            account.value_history[ts] = brokerage.get_account_value(account)

            # generate new orders and submit them to the brokerage
            """This is broken out as a major efficiency upgrade for genetic algorithms where you need the same state hundreds of times for your population. Found that when state was 
            gotten for each strategy, backtesting for 1 day slowed from ~0.75 seconds to 20 seconds."""
            if compute_state_for_each_strategy or state is None:
                logger.debug(f'Getting state for {algo.indicator_mapping}')
                state = get_state(algo.indicator_mapping, row)
            new_orders = algo.act(account, state)
            if new_orders is not None:
                accounts[account_num].submit_new_orders(new_orders)

        if logger.getEffectiveLevel() < 20 and display_progress_bar and i%10 == 0: 
            printProgressBar(i, n_rows)
        brokerage.clean_up()
    
    if logger.getEffectiveLevel() < 20 and display_progress_bar and i%10 == 0: 
        printProgressBar(n_rows+1, n_rows)
        print("")
        
    # catalog the outputs from the simulation
    results:AccountHistoryFileHandlerSet = []
    for account, strategy in zip(accounts.values(), algorithms.values()):
        res = AccountHistoryFileHandler(strategy, account, data.tickers, data.resolution)
        results.append(res)
    
    if plot:
        # find the best performing agent
        if len(results) > 1:
            best_strategy, best_account = get_best_strategy_and_account(results)
            print(f'Printing stats for the best performing strategy only: {best_strategy}')
        else:
            best_strategy = results[0].strategy
            best_account = results[0].account

    df = data.df

    if plot:
        performance_metrics = print_account_stats(best_account, data, brokerage.spread_percent, brokerage.market_order_slippage_model, brokerage.limit_order_slippage)
        
        create_pdf_performance_report(
            performance_metrics, 
            df, 
            best_account, 
            data.tickers, 
            type(best_strategy).__name__, 
            folder_to_save_plots)
        
        plt.show()
    
    return results

