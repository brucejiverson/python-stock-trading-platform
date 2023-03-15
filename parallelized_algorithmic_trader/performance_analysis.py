from typing import Any, List, Tuple, Collection, Callable
from typing import List
import numpy as np
import logging
import pandas as pd
from scipy.optimize import curve_fit

from parallelized_algorithmic_trader.data_management.market_data import CandleData
from parallelized_algorithmic_trader.broker import SimulatedAccount, OrderSide, AccountHistoryFileHandler, AccountHistoryFileHandlerSet
from parallelized_algorithmic_trader.util import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


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


def print_account_stats(account:SimulatedAccount, underlying:CandleData=None):
    """Calculates many performance metrics for the account history and prints them to the console.
    
    :param account: The account to calculate the performance metrics of.
    :param underlying: The underlying price history of the account. If this is not None, then alpha is not calculated.
    """
    
    print(f'\n')
    logger.info(f'General performance metrics:')
    history = account.get_history_as_list()
    
    logger.info('Max drawdown: ${:.0f}, {:.1f}%'.format(get_max_drawdown(account), get_max_drawdown_as_percent(account)))
    vh = account.get_history_as_list()
    logger.info('Starting value: ${}; final value: ${:.0f}'.format(vh[0], vh[-1]))
    logger.info('Total return: {:.1f}%, annualized return {:.1f}%'.format(get_ROI(history), get_annualized_ROI(account)))

    print("\n")
    logger.info(f'Trade stats:')
    # Get the buys and the sells
    for ticker in account._order_history.keys():
        n_buys = len(account._order_history[ticker][OrderSide.BUY])
        n_sells = len(account._order_history[ticker][OrderSide.SELL])
        
        logger.info(f'{ticker} # of buys: {n_buys}, # sells: {n_sells}.')
    
    # parse the buys and sells into trades and do basic analysis on them
    trades = account.get_trades()
    # trade_profits = [t.get_net_profit() for t in trades]
    # mean_profit = func_for_list_if_length(trade_profits, np.mean)
    trade_profits_perc = [t.get_profit_percent() for t in trades]
    mean_profit_percent = 100*func_for_list_if_length(trade_profits_perc, np.mean)
    profit_std = 100*func_for_list_if_length(trade_profits_perc, np.std)
    logger.info("Mean trade profit: {:.2f}%, stddev {:.2f}".format(mean_profit_percent, profit_std))
    
    trade_durations = [t.get_duration().total_seconds() for t in trades]
    mean_trade_duration = func_for_list_if_length(trade_durations, np.mean)/(60*60)
    trade_duration_std = func_for_list_if_length(trade_durations, np.std)/(60*60)
    logger.info('Mean time to exit: {:.0f} hours, stddev {:.0f}'.format(mean_trade_duration, trade_duration_std))
    
    buy_to_buy_times:List[pd.Timedelta] = [t1.buy.execution_timestamp - t2.buy.execution_timestamp for t1, t2 in zip(trades[1:], trades[:-1])]
    mean_time_between_trades = func_for_list_if_length([t.total_seconds() for t in buy_to_buy_times], np.mean)/(60*60)
    logger.info('Mean time from buy to buy: {:.0f} hours, {:.2f} days'.format(mean_time_between_trades, mean_time_between_trades/24))
    
    logger.info('Win rate: {:.1f} %'.format(get_win_percentage(trades=trades)))
    
    # vwr
    logger.info('VWR: {:.3f}, VWR curve fit: {:.3f}'.format(get_vwr(history), get_curve_fit_vwr(history)))
    if underlying is not None:
        logger.info('Alpha relative to ticker {} {:.2f}%'.format(*get_alpha(account, underlying)))

    # set the global variable for this
    # ticker = underlying.tickers[0]
    set_benchmark_score(underlying.df[ticker+'_close'], get_curve_fit_vwr)
    logger.info(f'VWR difference: {get_vwr_curve_fit_difference(account.get_history_as_list())}')
    
    
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


def get_expected_ROI(account:SimulatedAccount) -> float:
    """Fits a curve to the account value history and returns the expected return.
    
    :param account: The account to calculate the expected ROI of.
    :return: The expected ROI of the account as a percentage (ie 11 being 11%)
    """
    
    value_history = account.get_history_as_list()
    start_value = value_history[0]
    
    mean_return_per_timestep = fit_exponential_curve_fixed_start(value_history)
    end_val = start_value*np.exp(mean_return_per_timestep*len(value_history))
    return (end_val - start_value)/start_value*100


def get_expected_annualized_ROI(account:SimulatedAccount) -> float:
    """Fits a curve to the account value history and returns the expected annualized return.
    
    :param account: The account to calculate the expected annualized ROI of.
    :return: The expected annualized ROI of the account as a percentage (ie 11 being 11%)
    """
    value_history = account.get_history_as_list()
    start_value = value_history[0]
    
    end = list(account.value_history.keys())[-1]
    start = list(account.value_history.keys())[0]
    n_years = (end - start).total_seconds()/(60*60*24*365)
    
    mean_end_val = get_expected_ROI(account)
    mean_return = (mean_end_val -start_value)/start_value
    return 100*((1 + mean_return)**(1/n_years) - 1)
    

def get_max_drawdown(account:SimulatedAccount) -> float:
    """Returns the maximum drawdown of the account in dollars.
    
    :param account: The account to calculate the maximum drawdown of.
    :return: The maximum drawdown of the account in dollars.
    """
    latest_high = 0
    max_drawdown = 0
    for value in account.value_history.values():
        if value > latest_high:
            latest_high = value
        drawdown = latest_high - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def get_max_drawdown_as_percent(account:SimulatedAccount) -> float:
    """Returns the maximum drawdown of the account as a percentage of the initial account value."""
    return 100*get_max_drawdown(account) / account.get_history_as_list()[0]
 
    
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


def get_alpha(account:SimulatedAccount, underlying:CandleData, rf:float=0) -> Tuple[str, float]:
    """Calculates the alpha of the account."""
    # alpha = (r - r_f) - beta * (r_m - r_f)
    # r = account return
    # r_f = risk free rate
    # beta = beta of the account
    # r_m = market return
    # r_f = risk free rate
    beta = 1

    roi = get_ROI(account.get_history_as_list())
    ticker = underlying.tickers[0]
    underlying_return = get_ROI(underlying.df[ticker+'_close'])
    alpha = roi - rf - beta*(underlying_return - rf)
    return ticker, alpha


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