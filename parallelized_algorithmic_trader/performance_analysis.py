from typing import Any, List, Tuple, Collection, Callable, Dict
from typing import List
import numpy as np
import logging
import pandas as pd
from scipy.optimize import curve_fit

from parallelized_algorithmic_trader.data_management.data import CandleData
from parallelized_algorithmic_trader.trading.simulated_broker import SimulatedAccount, OrderSide, AccountHistoryFileHandler, AccountHistoryFileHandlerSet
from parallelized_algorithmic_trader.util import get_logger


logger = get_logger(__name__)
# logger.setLevel(logging.INFO)


def get_best_strategy_and_account(results=AccountHistoryFileHandlerSet) -> AccountHistoryFileHandler:
    """find the account with the best performance. Note that this assumes that all accounts have cashed out."""
    highest_value = 0
    best_result = None
    for r in results:
        if AccountHistoryFileHandler.final_value > highest_value:
            highest_value = r.final_value
            best_result = r
    return best_result


def func_for_list_if_length(list:List, func) -> Any:
    """Returns the result of the function if the list is not empty, otherwise returns 0."""
    if len(list) == 0: return 0
    return func(list)


def print_account_stats(account:SimulatedAccount, underlying:CandleData=None, spread=None, market_slippage=None, limit_slippage=None) -> Dict[str, str]:
    """Calculates many performance metrics for the account history and prints them to the console.
    
    :param account: The account to calculate the performance metrics of.
    :param underlying: The underlying price history of the account. If this is not None, then alpha is not calculated.
    """
    
    analysis_data = {}
    
    def log_and_append(key, value):
        if not isinstance(value, str):
            value = str(value)
        logger.info(f'{key}: {value}')
        analysis_data[key] = value
    
    print(f'\n')
    logger.info(f'General performance metrics for account:')
    log_and_append('Start date', list(account.value_history.keys())[0])
    log_and_append('End date', list(account.value_history.keys())[-1])
    log_and_append('Data resolution', underlying.resolution.name)
    
    if spread is not None:
        log_and_append('Market order spread', f'{spread:.3f}%')
    if market_slippage is not None:
        log_and_append('Slippage for market orders', market_slippage.name)
    if limit_slippage is not None:
        log_and_append('Slippage for limit orders', f'{limit_slippage:.4f}')
        
    history = account.get_history_as_list()
    
    log_and_append('Max drawdown', f'{get_max_drawdown_as_percent(history):.2f}%')
    
    log_and_append('Starting value', '${}'.format(history[0]))
    log_and_append('Final value', '${:.0f}'.format(history[-1]))
    log_and_append('Total return', '{:.1f}%'.format(get_ROI(history)))
    log_and_append('Annualized return', '{:.1f}%'.format(get_annualized_ROI(account)))

    print("\n")
    logger.info(f'Trade stats:')
    # Get the buys and the sells
    for ticker in account._order_history.keys():
        n_buys = len(account._order_history[ticker][OrderSide.BUY])
        n_sells = len(account._order_history[ticker][OrderSide.SELL])
        logger.debug(f'{ticker} # of buys: {n_buys}, # sells: {n_sells}.')
    
    # parse the buys and sells into trades and do basic analysis on them
    trades = account.get_trades()
    if trades == []:
        account.parse_order_history_into_trades(underlying.df)
        trades = account.get_trades()
    
    log_and_append('Number of trades', len(trades))
    
    trade_profits_perc = [t.get_profit_percent() for t in trades]
    mean_profit_percent = 100*func_for_list_if_length(trade_profits_perc, np.mean)
    profit_std = 100*func_for_list_if_length(trade_profits_perc, np.std)
    log_and_append('Mean trade profit', f'{mean_profit_percent:.2f}%, stddev {profit_std:.2f}')
    
    trade_durations = [t.get_duration().total_seconds() for t in trades]
    mean_trade_duration = func_for_list_if_length(trade_durations, np.mean)/(60*60)
    trade_duration_std = func_for_list_if_length(trade_durations, np.std)/(60*60)
    log_and_append('Mean time to exit', f'{mean_trade_duration:.0f} hours, stddev {trade_duration_std:.0f}')
    
    buy_to_buy_times:List[pd.Timedelta] = [t1.buy.execution_timestamp - t2.buy.execution_timestamp for t1, t2 in zip(trades[1:], trades[:-1])]
    mean_time_between_trades = func_for_list_if_length([t.total_seconds() for t in buy_to_buy_times], np.mean)/(60*60)
    log_and_append('Mean time from buy to buy', f'{mean_time_between_trades:.0f} hours, {mean_time_between_trades/24:.2f} days')
    log_and_append('Win rate', f'{get_win_percentage(trades=trades):.1f} %')  
    
    # complex metrics
    if underlying is not None:
        ticker, alpha = get_alpha(account, underlying)
        log_and_append(f'Alpha relative to {ticker}', f'{alpha:.2f}%')
    
    log_and_append('VWR', f'{get_vwr(history):.3f}')
    log_and_append('VWR curve fit', f'{get_curve_fit_vwr(history):.3f}')

    # set the global variable for this
    ticker = underlying.tickers[0]
    set_benchmark_score(underlying.df[ticker+'_close'], get_curve_fit_vwr)
    vwr_diff = get_vwr_curve_fit_difference(history)
    log_and_append('VWR difference', f'{vwr_diff:.4}')
    return analysis_data

    
def get_win_percentage(account:SimulatedAccount=None, trades=None) -> float:
    """Returns the win percentage of the account.
    
    :param account: The account to calculate the win percentage of.
    :param trades: The trades to calculate the win percentage of. If this is None, then the trades are parsed from the account.
    :return: The win percentage of the account.
    """
    if trades is None:
        logger.debug(f'Calculating win percentage from account {account}.')
        trades = account.get_trades()
    
    n_wins = len([t for t in trades if t.get_net_profit() > 0])
    if len(trades) == 0:
        return 0
    return 100*n_wins / len(trades)


def get_ROI(data:Collection) -> float:
    """Returns the ROI of the data.
    
    :param data: The data to calculate the ROI of.
    :return: The ROI of the data as a percentage (ie 11 being 11%)
    """
    # Returns the percentage increase/decrease
    return (data[-1] / data[0]  - 1)*100


def get_annualized_ROI(account:SimulatedAccount) -> float:
    """Returns the annualized returns of the account. 
    
    Note that this is an approximation as it does not account for business days/weekends/holidays.
    
    :param account: The account to calculate the annualized ROI of.
    :return: The annualized ROI of the account as a percentage (ie 11 being 11%)
    """
    
    roi = get_ROI(account.get_history_as_list())

    end = list(account.value_history.keys())[-1]
    start = list(account.value_history.keys())[0]
    n_years = (end - start).total_seconds()/(60*60*24*365)
    annualized_roi = (1 + roi/100)**(1/n_years) - 1
    return annualized_roi*100


def get_expected_ROI(data:Collection) -> float:
    """Fits a curve to the account value history and returns the expected return.
    
    :param account: The account to calculate the expected ROI of.
    :return: The expected ROI of the account as a percentage (ie 11 being 11%)
    """
    
    start_value = data[0]
    
    mean_return_per_timestep = fit_exponential_curve_fixed_start(data)
    end_val = start_value*np.exp(mean_return_per_timestep*len(data))
    return (end_val - start_value)/start_value*100


def get_max_drawdown(data:Collection) -> float:
    """Returns the maximum drawdown of the account in dollars.
    
    :param account: The account to calculate the maximum drawdown of.
    :return: The maximum drawdown of the account in dollars.
    """
    latest_high = 0
    max_drawdown = 0
    for value in data:
        if value > latest_high:
            latest_high = value
        drawdown = latest_high - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def get_max_drawdown_as_percent(data:Collection) -> float:
    """Returns the maximum drawdown of the account as a percentage of the initial account value."""
    return 100*get_max_drawdown(data) / data[0]
 
    
def fit_exponential_curve_fixed_start(y:list[float]) -> float:
    """Fits an exponential curve to the data with the starting value being equal according to formula
    y = start_value * (1 + r)**t; or y = start_value * exp(r*t)
    The fitting is done in the log space in order to give equal weighting to data over time. 
    
    Returns the mean growth rate per timestep."""
    
    start_value = y[0]
    time = range(len(y))

    def exponential_growth(t, r):
        return np.log(start_value) + t*np.log(1 + r)
        # return np.log(start_value) + y*np.log(t)

    # check for the case where the value is constant over time
    if len(set(y)) == 1:
        return 0


    popt, pcov = curve_fit(exponential_growth, time, np.log(y))     # your data x, y to fit
    r_mean = popt[0]                                        # average growth per timestep
    return r_mean


def get_vwr(data:Collection, tau:float=4, stddev_max:float=0.2) -> float:
    """Calculates the variability weighted return for the account."""
    '''Variability-Weighted Return: Better SharpeRatio with Log Returns
    Alias:
      - VariabilityWeightedReturn
    See:
      - https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
    Params:
      - ``tau`` (default: ``4.0``)
         - rate at which weighting falls with increasing variability  (investor tolerance)
      - ``sdev_max`` (default: ``0.20``)
        maximum acceptable σP  (investor limit). The stddev of the difference between value and zero variance expected value
    '''

    logger.debug('Calculating VWR')
    start_val, end_val = data[0], data[-1]
    
    mean_return_per_timestep = (end_val/start_val)**(1/len(data)) - 1

    expected_end_val = start_val*(1 + mean_return_per_timestep)**len(data)
        
    # penalizing variance defined as account_value - mean_exponential growth
    zero_variability_returns = [start_val*np.exp(mean_return_per_timestep*i) for i in range(len(data))]
    logger.debug(f'vwr expected end val: {zero_variability_returns[-1]}')
    
    # difference between actual and zero-variability prices divided by zero-variability prices to normalize 
    log_difference = [v/v_zero_var - 1 for v, v_zero_var in zip(data, zero_variability_returns)]

    sigma_p = np.std(log_difference)    # *data[0]   # multiply by the starting value as we normalized this earlier, so sigma is in dollars
    if sigma_p > stddev_max:
        logger.warning(f'VWR sigma_p: {sigma_p} is greater than the max allowed: {stddev_max}. returning roi')
        return expected_end_val/start_val - 1
    
    log_return = np.log(zero_variability_returns[-1]/zero_variability_returns[0])
    logger.debug(f'log return: {log_return}')

    if mean_return_per_timestep < 0:
        logger.debug(f'Warning: mean return is less than 0. VWR will be modified to amplify the negative return.')
        variance_term = 1 + (sigma_p/stddev_max)**tau
    else: 
        variance_term = 1 - (sigma_p/stddev_max)**tau
    return log_return * variance_term


def get_curve_fit_vwr(data:Collection, tau:float=4, stddev_max:float=0.2) -> float:
    """Calculates the variability weighted return for the account. If less that 0 returns the expected ROI"""
    '''Variability-Weighted Return: Better SharpeRatio with Log Returns
    Alias:
      - VariabilityWeightedReturn
    See:
      - https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
    Params:
      - ``tau`` (default: ``4.0``)
         - rate at which weighting falls with increasing variability  (investor tolerance)
      - ``sdev_max`` (default: ``0.20``)
        maximum acceptable σP  (investor limit). The stddev of the difference between value and zero variance expected value
    '''
    logger.debug('calculating curve fit vwr')
    
    mean_return_per_timestep = fit_exponential_curve_fixed_start(data)
    
    start_value = data[0]
    # penalizing variance defined as account_value - mean_exponential growth
    zero_variability_returns = [start_value*np.exp(mean_return_per_timestep*i) for i in range(len(data))]
    logger.debug(f'Expected end val: {zero_variability_returns[-1]}') 
    
    # difference between actual and zero-variability prices divided by zero-variability prices to normalize
    normalized_differences = [v/v_zero_var - 1 for v, v_zero_var in zip(data, zero_variability_returns)]
    
    sigma_p = np.std(normalized_differences)    # * start_value
    if sigma_p > stddev_max: 
        logger.warning(f'VWR sigma_p is greater than the max allowed. Returning curve fit roi')
        return zero_variability_returns[-1]/start_value - 1
    logger.debug(f'VWR curve fit sigma_p: {sigma_p}')
    
    log_return = np.log(zero_variability_returns[-1]/start_value)
    logger.debug(f'log return: {log_return}')
    
    if mean_return_per_timestep < 0:
        logger.debug(f'Warning: mean return is less than 0. VWR will be modified to amplify the negative return.')
        variance_term = 1 + (sigma_p/stddev_max)**tau
    else: 
        variance_term = 1 - (sigma_p/stddev_max)**tau
        
    vwr = log_return * variance_term
    return vwr


def get_alpha(account:SimulatedAccount, underlying:CandleData, rf:float=0, specific_ticker:str|None=None) -> Tuple[str, float]:
    """Calculates the alpha of the account. If specific ticker is provided, returns the alpha of that ticker. 
    Otherwise defaults to the average alpha of all tickers in the underlying data."""
    # alpha = (r - r_f) - beta * (r_m - r_f)
    # r = account return
    # r_f = risk free rate
    # beta = beta of the account
    # r_m = market return
    # r_f = risk free rate
    beta = 1

    roi = get_ROI(account.get_history_as_list())
    
    n_tickers = len(underlying.tickers)
    if n_tickers == 1: specific_ticker = underlying.tickers[0]
    
    if specific_ticker is not None:
        assert specific_ticker in underlying.tickers, f'{specific_ticker} not in underlying tickers'
        underlying_return = get_ROI(underlying.df[specific_ticker+'_close'])
        alpha = roi - rf - beta*(underlying_return - rf)
        return specific_ticker, alpha
    else:
        underlying_returns = [get_ROI(underlying.df[ticker+'_close']) for ticker in underlying.tickers]
        average_underlying_return = np.mean(underlying_returns)
        alpha = roi - rf - beta*(average_underlying_return - rf)
        return 'equal weight basket all tickers', alpha


def geometric_sharpe(account:SimulatedAccount, underlying:CandleData, rf:float=0, specific_ticker:str|None=None) -> Tuple[str, float]:
    """Calculates the sharpe ratio of the account. If specific ticker is provided, returns the sharpe ratio of that ticker. 
    Otherwise defaults to the average sharpe ratio of all tickers in the underlying data."""
    
    history = account.get_history_as_list()
    geometric_mean = np.prod([1 + x for x in history])**(1/len(history)) - 1


def calmar_ratio(
    account:SimulatedAccount, 
    underlying:CandleData, 
    rf:float=0, 
    specific_ticker:str|None=None) -> float:
    """Calculates the calmar ratio of the account. If specific ticker is provided, returns the calmar ratio of that ticker. 
    Otherwise defaults to the average calmar ratio of all tickers in the underlying data."""
    
    history = account.get_history_as_list()
    max_drawdown = get_max_drawdown(history)
    if max_drawdown == 0:
        return 0
    roi = get_ROI(history)
    calmar = roi/max_drawdown
    return calmar


UNDERLYING_SCORE = None


def set_benchmark_score(benchmark_history:Collection, fitness_function:Callable):
    """Sets the benchmark score to be used in the score function."""
    global UNDERLYING_SCORE
    UNDERLYING_SCORE = fitness_function(benchmark_history)
    

def get_vwr_curve_fit_difference(account_values:Collection):
    """Returns the difference between the curve fit vwr and the benchmark score global variable."""    
    
    assert UNDERLYING_SCORE is not None, 'Underlying score must be set before calling this function. Call set_benchmark_score'
    
    account_vwr = get_curve_fit_vwr(account_values)
    return account_vwr - UNDERLYING_SCORE  


