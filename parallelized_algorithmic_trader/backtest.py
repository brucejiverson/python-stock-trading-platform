from typing import Tuple, Any
import logging
import pandas as pd
from numba import jit, types
from numba.typed import Dict, List
import matplotlib.pyplot as plt
import time

from parallelized_algorithmic_trader.indicators import IndicatorMapping, IndicatorConfig, aggregate_indicator_mappings, check_if_indicator_mappings_identical
from parallelized_algorithmic_trader.strategy import StrategyConfig, StrategyBase
from parallelized_algorithmic_trader.data_management.data import CandleData
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.trading.simulated_broker import *
from parallelized_algorithmic_trader.performance_analysis import print_account_stats, get_best_strategy_and_account
from parallelized_algorithmic_trader.visualizations import *
from parallelized_algorithmic_trader.util import get_logger


logger = get_logger(__name__ )
import logging
root_logger = logging.getLogger('pat')

# this is global so that we can make it once and then resuse it when doing consecutive backtests
main_data:CandleData = None                     # all of the data including candles and any features or indicators

train_test_split_flag = False
train_test_split_index:int = None

train_data:CandleData = None
test_data:CandleData = None
compute_state_for_each_strategy:bool = None


# global variables for speed in backtesting
TICKER_PRICE_IDXS = None

def get_dataset(use_test_data:bool) -> CandleData:
    
    # set up the data
    global train_test_split_flag
    if train_test_split_flag:
        if use_test_data:
            root_logger.info(f'Using test data')
            global test_data
            return test_data
        else:
            root_logger.info(f'Using train data')
            global train_data
            return train_data            
    else:
        root_logger.debug(f'Using sim data, no train test split.')
        global main_data
        return main_data


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


def build_features(candle_data:CandleData, indicator_mappings:list[IndicatorMapping]) -> CandleData:
    """Builds the features for the backtest. This is a global function so that we can make it once 
    and then resuse it when doing consecutive backtests
    
    Parameters:
        candle_data (CandleData): The candle data to build the features on
        indicator_mappings (list[IndicatorMapping]): The indicator mappings to build the features with
        
    Returns:
        CandleData: The candle data with the features built
    """
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
    logger.info(f'Received features: {comprehensive_indicator_mapping}')
    logger.debug(f'Type of comprehensive_indicator_mapping: {type(comprehensive_indicator_mapping)}')
    logger.info(f'Building {len(comprehensive_indicator_mapping)} features')
    for indicator_config in comprehensive_indicator_mapping:
        logger.debug(f'Type of indicator_config: {type(indicator_config)}')
        logger.debug(f'Start date of main_df before making indicator: {main_df.index[0]} for indicator {indicator_config.config_name}')
        # if len(indicator_config.names) == 0:
        #     raise ValueError(f'Indicator config {indicator_config} has no names. ')
        print(f'Building indicator {indicator_config.config_name} with features {indicator_config.names}')
        # check if this indicator has already been built by looking for names
        if len(indicator_config.names) > 0 and all([name in main_df.columns for name in indicator_config.names]):
            logger.debug(f'Indicator {indicator_config.names} has already been built. Skipping.')
            continue
        
        indicator = indicator_config.make(main_df)
        print(f'Made: {indicator_config.config_name}, {indicator_config.names}')
        if indicator is not None:
            main_df = pd.concat([main_df, indicator], axis=1)
        
        logger.info(f'Built indicator {indicator_config.config_name} with features {indicator_config.names}')
        
    # always build in the hour of the day and the timestamp as columns. The timestamp is now duplicated data...
    main_df['hour'] = main_df.index.hour
    
    # make the hour fractional
    if candle_data.resolution != TemporalResolution.DAY:
        main_df['hour'] = main_df['hour'] + main_df.index.minute / 60

    # force datatype to float64 for all column 
    main_df.drop(columns=['Source'], inplace=True)
    main_df = main_df.astype('float64')
    
    logger.info(f'Columns list after indicators build: {main_df.columns}')
    logger.debug(f'Main dataframe has {len(main_df)} rows of data. Starting at {main_df.index[0]} and ending at {main_df.index[-1]}')
    
    # now set up their data objects
    logger.debug(f'Do not forget that putting the dataframe into the CandleData object will automatically clean it, removing any rows that have nan values (indicator warm up)')
    logger.debug(f'Setting up main data frame')
    main_data = CandleData(main_df, candle_data.tickers, candle_data.resolution)
    logger.info(f'Ordered columns: {main_data.df.columns}')
    # main_data.df['idx'] = main_data.df.reset_index(drop=True).index.values.astype('int64')
    
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
    algorithm_configs:list[StrategyConfig],
    slippage_model:SlippageModelsMarketOrders=None,
    plot:bool = False,
    folder_to_save_plots=None,
    display_progress_bar:bool=True,
    log_level:int = logging.WARNING) -> TradingHistorySet:
    """Backtest a set of algorithms on a set of market data

    Args:
        market_data (CandleData): [description]
        algorithm_configs (list[StrategyConfig]): [description]
        scaler (IndicatorConfig, optional): [description]. Defaults to None.
        SlippageModel (SlippageModelsMarketOrders, optional): [description]. Defaults to None.
        plot (bool, optional): [description]. Defaults to False.
        log_level (int, optional): [description]. Defaults to logging.WARNING.

    Returns:
        TradingHistorySet: [description]
    """

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


AccountLite = types.float64[:]
oned_arr = types.Array(types.float64, 1, "C")
    

def check_if_all_configs_have_the_same_state_expectation(algorithm_configs:list[StrategyConfig]) -> bool:
    """Checks if all the configs have the same state expectation. If they do, we can compute the state once and then pass it to all the algorithms"""

    first_config = algorithm_configs[0]
    for config in algorithm_configs:
        if not check_if_indicator_mappings_identical(config.indicator_mapping, first_config.indicator_mapping):
            return False
    return True


class SimulatedStockBrokerCandleData(Brokerage):
    """This class is used to simulate an exchange. It is used for backtesting and for training Strategys.
    
    
    """
    
    def __init__(
        self,
        data:CandleData = None,                                         # the data to use for the simulation
        spread_percent:float=0.1,                                       # the spread of the exchange as a percentage
        limit_order_slippage:float=0.02,                                # the slippage of limit and stop orders in dollars, default to two ticks. 
        market_order_slippage_model:SlippageModelsMarketOrders=None,    # the commission of the exchange as a percentage
        commision_percent:float=0.0,                                    # the slippage of the exchange as a percentage
        log_level=None):
        
        super().__init__(__name__ + '.' + self.__class__.__name__)
        if log_level is not None: 
            self.logger.setLevel(log_level)
        # add the market data to the brokerage
        assert isinstance(data, CandleData), f'Expected data to be a CandleData object, got {type(data)}'
        
        self._data:CandleData = None
        self.ingest_data(data)
        self._current_ts:pd.Timestamp = None
        
        self._accounts:AccountSet = {}
        
        self.spread_percent:float = spread_percent
        self.commission_percent:float = commision_percent
        
        if market_order_slippage_model is not None:
            self.logger.info(f'Using slippage model: {market_order_slippage_model} as dictacted by user')
            self.set_slippage_model(market_order_slippage_model)
        elif data.resolution == TemporalResolution.MINUTE:
            self.logger.info(f'Using minute resolution data. Slippage model is set to NEXT_CLOSE')
            self.set_slippage_model(SlippageModelsMarketOrders.NEXT_CLOSE)
        else:
            self.logger.info(f'The slippage model has been left at the default, {SlippageModelsMarketOrders.NEXT_OPEN.name}')
            self.set_slippage_model(SlippageModelsMarketOrders.NEXT_OPEN)

        self.limit_order_slippage:float = limit_order_slippage
        self.logger.info(f'Initialized with spread: {self.spread_percent}%, commission: {self.commission_percent}, market order slippage: {self._market_order_slippage.name}, limit order slippage: ${self.limit_order_slippage}')

    def set_spread(self, spread:float):
        """Sets the spread (difference between bid and ask) of the exchange. Value should be a percentage
        
        :param spread: The spread of the exchange as a percentage
        """
        self.spread_percent = spread
        self.logger.debug(f'Spread set to {spread}')
    
    def set_commission(self, commission:float):
        """Sets the commission of the exchange. Value should be a percentage.
        
        :param commission: The commission of the exchange as a percentage
        """
        self.commission_percent = commission
        self.logger.debug(f'Commission set to {commission}')
    
    def set_slippage_model(self, model:SlippageModelsMarketOrders):
        """Sets the slippage of the exchange. Value should be a percentage
        
        :param model: The slippage model to use
        """
        self._market_order_slippage = model
        self.logger.debug(f'Slippage set to {model.name}, which means {model.value}')

    def ingest_data(self, data:CandleData):
        """Ingests the data into the exchange. This should be called once per tick.
        
        :param data: A pandas dataframe containing the data to ingest
        """
        assert isinstance(data.df, pd.DataFrame), f'Expected data to be a pandas dataframe, but got {type(data)}'
        self._data = data
        self.add_ticker(data.tickers)
        self._set_expected_resolution(data.resolution)


# @jit(nopython=True, cache=True)    
def process_orders_for_account(
    row_num:types.int64,
    cur_data:np.ndarray, 
    ticker_idxs, 
    spread:types.float64,
    market_slippage_name:types.unicode_type,
    account:AccountLite, 
    pending_orders:List[types.unicode_type],
    order_submissions:List[types.unicode_type],
    order_executions:List[np.float64],
    limit_order_slippage:float=.02
    ):
    
    """Implicitly relies on the ordering of accounts, account vals, and new orders to align as they are accessed by index."""

    tickers = list(set([t.split('_')[0] for t in ticker_idxs.keys()]))
    
    order_idx = 0
    while order_idx < len(pending_orders):
        order = pending_orders[order_idx]
        # handle cases for order types that may not have all ticker/price data properties first
        # if order[0] == 'CancelAllOrders':
        #     account.cancel_pending_orders        #     break
        # elif order[0] == 'CancelOrder':
        #     acc.cancel_order(order.order_to_cancel)
        #     logger.debug('Cancelling pending order...')
        #     break   # canceling modifies the list, so we need to break here. The while loop handles reattempting
        
        # high = get_cur_high(order[2], cur_data, ticker_idxs)
        # low = get_cur_low(order[2], cur_data, ticker_idxs)
        
        # now depending on the order type check to execute order, and/or update with current price info
        # if 'market' in order[0].lower():
        if order[0] == 'MarketOrder':
            # logger.debug(f'Executing market order: {order}')
            order = pending_orders.pop(order_idx)
            # get the index of the ticker from the list of tickers using idxs and stripping out candle suffixes
            acc_shares_idx = tickers.index(order[2]) + 1
            # logger.debug(f'Tickers: {tickers}. Account shares idx: {acc_shares_idx}. Account: {account}')
            ex_price = get_execution_price_for_market_order(order, cur_data, ticker_idxs, spread, market_slippage_name)
            if order[1] == 'BUY':
                execute_buy_order(account, ex_price, order, acc_shares_idx)
            else:
                execute_sell_order(account, ex_price, order, acc_shares_idx)
            # order.append(account[acc_shares_idx])
            # order.append(ex_price)
            # order.append(row_num)
            order_submissions.append(order)
            order_executions.append(List([account[acc_shares_idx], ex_price, row_num]))
            # logger.info(f'Order execution: {order}, shares: {account[acc_shares_idx]}, execution price: {ex_price}, row num: {row_num}')
            continue
        
        else:
            raise ValueError(f'Invalid order type: {order[0]} full order: {order}')
            # elif order[0] == 'TrailingStopOrder':
            #     if order[1] == 'BUY':
            #         order.update(high)
            #         if order.stop_triggered:
            #             price = order.stop_price + limit_order_slippage
            #             execute_pending_order(acc, price, idx)
            #             continue
            #     elif order[1] == 'SELL':
            #             order.update(low)
            #             if order.stop_triggered:
            #                 price = order.stop_price - limit_order_slippage
            #                 execute_pending_order(acc, price, idx)
            #                 continue
            
            # elif order[0] == 'LimitOrder':
            #     # check the price and see if this order should get filled
            #     if order[1] == 'BUY' and low <= order.limit_price:
            #         price = order.limit_price + limit_order_slippage
            #         execute_pending_order(acc, price)
            #         continue
            #     elif order[1] == 'SELL' and high >= order.limit_price:
            #         price = order.limit_price - limit_order_slippage
            #         execute_pending_order(acc, price)
            #         continue
            
            # elif order[0] == 'StopOrder':
            #     # check the price and see if this order should get filled
            #     if order[1] == 'BUY' and high >= order.stop_price:
            #         price = order.stop_price + limit_order_slippage
            #         execute_pending_order(acc, price)
            #         continue
            #     elif order[1] == 'SELL' and low <= order.stop_price:
            #         price = order.stop_price - limit_order_slippage
            #         execute_pending_order(acc, price)
            #         continue
            
            # elif order[0] == 'StopLimitOrder':
            #     # check the price and see if this order should get filled
            #     if order[1] == 'BUY':
            #         if not order.stop_triggered:
            #             order.update(high)
            #         else:
            #             order.update(low)
            #         if order.limit_triggered:
            #             price = order.limit_price + limit_order_slippage
            #             execute_pending_order(acc, price)
            #             continue
            #     elif order[1] == 'SELL':
            #         if not order.stop_triggered:
            #             order.update(low)
            #         else:
            #             order.update(high)
            #         if order.limit_triggered:
            #             price = order.limit_price - limit_order_slippage
            #             execute_pending_order(acc, price)
            #             continue


# @jit(nopython=True, cache=True)    
def fast_sim(
    raw_data:np.ndarray,
    algos:list[StrategyBase],
    accounts:list[AccountLite],
    ordered_cols:str,
    TICKER_PRICE_IDXS:Dict,
    market_slippage_model:str,
    spread_percent:float=0.1,
    limit_slippage:float=0.02) -> tuple[list[list[str]], np.ndarray]:
    
    idxs_for_algostate = np.array(
        [[ordered_cols.index(c) for c in a.indicator_mapping.feature_names] for a in algos], 
        dtype=np.int64)
    
    close_idxs = np.array([v for k, v in TICKER_PRICE_IDXS.items() if k.endswith('_close')])
    n_time_steps = len(raw_data)
    account_val_hists = np.zeros((len(accounts), n_time_steps), dtype=np.float64)
    order_submissions = [['remove me'] for _ in range(len(accounts))]  # jit throws errors with empty lists so none is added as first element and later ignored
    order_executions = [[1.] for _ in range(len(accounts))]
    pending_orders = [['remove me'] for _ in range(len(accounts))]
    [l.pop() for sublist in (order_submissions, order_executions, pending_orders) for l in sublist]
    
    for i, obs in enumerate(raw_data):
        ordered_obs = None

        # if not self.check_if_market_is_open():
        #     continue
        
        for j, (account, algo) in enumerate(zip(accounts, algos)):
            if compute_state_for_each_strategy or ordered_obs is None:
                ordered_obs = obs[idxs_for_algostate[0]]          # to do: make this work for multiple algo ordered_obs configs
            pending_orders[j].extend(algo.act(ordered_obs, account, pending_orders[j]))
            if len(pending_orders[j]) > 0:
                process_orders_for_account(
                    i, 
                    obs, 
                    TICKER_PRICE_IDXS, 
                    spread_percent, 
                    market_slippage_model, 
                    account, 
                    pending_orders[j], 
                    order_submissions[j],
                    order_executions[j],)
        account_val_hists[:, i] = (accounts[:, 0:1] + accounts[:, 1:] * obs[close_idxs])[:, 0]
    return order_submissions, order_executions, account_val_hists


HAS_RAN_FLAG = False
SLIPPAGE_MODEL = None

def run_simulation_on_candle_data(
    algorithm_configs:list[StrategyConfig],
    spread_percent:float=0.1,
    market_slippage_model:SlippageModelsMarketOrders|None=None,
    limit_slippage:float=0.02,
    use_test_data:bool = False,
    drop_unneeded_ohlcv_cols:bool=True,
    plot:bool=False,
    folder_to_save_plots:str|None=None,
    return_only_orders_and_history:bool=False,
    log_level=logging.WARNING) -> TradingHistorySet:

    log_level = min(log_level, root_logger.getEffectiveLevel())
    logger.setLevel(log_level)
    start_time = time.time()
    # check if all the configs have the same state expectation. If any are unique, then the environment state will be reconstructed for each strategy. 
    # this is useful for things like genetic algorithms where the state will be the same for each bot in the population
    
    data = get_dataset(use_test_data)
    logger.info(f'Backtesting from {data.start_date} to {data.end_date}')

    logger.debug(f'Instantiating algos... time: {time.time() - start_time:.3f} s')
    # set up the algorithms and accounts. Instantiates a strategy and adds a matching account
    for algo_config in algorithm_configs:
        algo_config.indicator_mapping.get_all_feature_names()
    algorithms:tuple[StrategyBase] = tuple(ac.instantiate(0) for ac in algorithm_configs)
    tkrs = np.array(data.tickers)
    accounts:types.float64[:,:] = np.array([[10000] + [0]*len(tkrs) for _ in algorithms], dtype=np.float64)
    
    logger.debug(f'Algos instantiated. Time: {time.time() - start_time:.3f} s')

    # these are global so they are persistanct across multiple simulation
    global TICKER_PRICE_IDXS
    global HAS_RAN_FLAG
    global SLIPPAGE_MODEL
    drop_cols = []
    if not HAS_RAN_FLAG:
        if market_slippage_model is None:
            if data.resolution == TemporalResolution.MINUTE:
                logger.info(f'Slippage model is set to NEXT_CLOSE as default for minute resolution candle data.')
                SLIPPAGE_MODEL = SlippageModelsMarketOrders.NEXT_CLOSE
            else: 
                logger.info(f'The slippage model has been left at the default, {SlippageModelsMarketOrders.NEXT_OPEN.name}')
                SLIPPAGE_MODEL = SlippageModelsMarketOrders.NEXT_OPEN

        global compute_state_for_each_strategy
        if compute_state_for_each_strategy is None:
            compute_state_for_each_strategy = check_if_all_configs_have_the_same_state_expectation(algorithm_configs)
            logger.debug('compute_state_for_each_strategy: ' + str(compute_state_for_each_strategy))
                
        if drop_unneeded_ohlcv_cols:
            if SLIPPAGE_MODEL == SlippageModelsMarketOrders.NEXT_OPEN:
                drop_cols = list(t + suffix for t in data.tickers for suffix in ('_high', '_low', '_volume'))
            elif SLIPPAGE_MODEL == SlippageModelsMarketOrders.NEXT_CLOSE:
                drop_cols = list(t + suffix for t in data.tickers for suffix in ('_open', '_high', '_low', '_volume'))
            data.df.drop(columns=drop_cols, inplace=True)
        HAS_RAN_FLAG = True
        
        if TICKER_PRICE_IDXS is None:
            # Sets the indexes of the open high low close columns for each ticker for easy lookup once everything is np arrays
            cols = data.df.columns
            TICKER_PRICE_IDXS = Dict.empty(
                key_type=types.unicode_type,
                value_type=types.int64
            )
            sfxs = ('open', 'high', 'low', 'close')
            [TICKER_PRICE_IDXS.__setitem__(f'{t}_{sfx}', cols.get_loc(f'{t}_{sfx}')) for t in main_data.tickers for sfx in sfxs if f'{t}_{sfx}' in cols]
    ORDERED_COLUMNS = [c for c in main_data.df.columns if c not in drop_cols]

    logger.debug(f'Time before data var assigned: {time.time() - start_time:.3f} s')
    raw_data = data.df.values.astype('float64')
    
    logger.debug(f'Initializing backtest took {time.time()-start_time:.3f} seconds')
    # logger.info(f'Parameters for fast sim: {raw_data.shape}, {len(algorithms)}, {len(accounts)}, {ORDERED_COLUMNS}, {TICKER_PRICE_IDXS}, {SLIPPAGE_MODEL.name}, {spread_percent}, {limit_slippage}')
    order_submis_hists, order_ex_hists, acc_val_hists = fast_sim(
        raw_data, 
        algorithms, 
        accounts, 
        ORDERED_COLUMNS, 
        TICKER_PRICE_IDXS,
        SLIPPAGE_MODEL.name, 
        spread_percent, 
        limit_slippage)
    start_time = time.time()

    # zip together the order submissions and executions
    full_order_historys = []
    for o_subs, o_exs in zip(order_submis_hists, order_ex_hists):
        hist = []
        for o, oinfo in zip(o_subs, o_exs): 
            if o is not None:
                order = [*o, *oinfo[:-1], data.df.index[int(oinfo[-1])]]
                hist.append(order)
        full_order_historys.append(hist)

    if return_only_orders_and_history:
        return full_order_historys, acc_val_hists

    reconstructed_val_historys = []
    for acc_val_hist in acc_val_hists:
        reconstructed_val_historys.append({data.df.index[n]:acc_val_hist[n] for n in range(len(acc_val_hist))})

    results:TradingHistorySet = []
    for acc, val_hist, order_hist, algo in zip(accounts, reconstructed_val_historys, full_order_historys, algorithms):
        new_account = SimulatedAccount.initialize_from_lite(data.tickers, acc, val_hist, order_hist)
        res = TradingHistory(algo, new_account, data.tickers, data.resolution)
        results.append(res)
    logger.debug(f'Post processing took {time.time()-start_time:.3f} seconds')
    
    if plot:
        best_result = get_best_strategy_and_account(results)
        best_strategy, best_account = best_result.strategy, best_result.account
        if len(results) > 1:
            logger.info(f'Printing stats for the best performing strategy only: {best_strategy}')

        performance_metrics = print_account_stats(
            best_account, 
            data, 
            spread_percent, 
            market_slippage_model, 
            limit_slippage)
        
        create_pdf_performance_report(
            performance_metrics, 
            data.df,
            best_account, 
            data.tickers, 
            type(best_strategy).__name__, 
            folder_to_save_plots)
        
        plt.show()
    return results
