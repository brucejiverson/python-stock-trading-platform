from typing import Any, List, Tuple
import random
from dataclasses import dataclass
import datetime
from typing import List, Dict
import numpy as np
import logging
import pandas as pd
from scipy.optimize import curve_fit

from parallelized_algorithmic_trader.data_management.market_data import CandleData, TemporalResolution
from parallelized_algorithmic_trader.broker import Account, OrderSide, OrderBase, AccountHistory, AccountHistorySet
from parallelized_algorithmic_trader.util import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class Trade:
    """Holds all the information defining a trade, and provides some utility functions for analyzing performance."""
    buy:OrderBase
    sell:OrderBase
    max_draw_down:float|None = None
    max_percent_up:float|None = None

    @property
    def ticker(self) -> str:
        return self.buy.ticker

    def get_net_profit(self) -> float:
        """Returns the net profit of the trade, calculated as the difference between the buy and sell value.
        
        :return: float representing the net profit of the trade.
        """
        return self.sell.get_transaction_value() - self.buy.get_transaction_value()
    
    def get_profit_percent(self) -> float:
        """Returns the profit of the trade as a percentage gain or loss.
        
        :return: float representing the percentage gain or loss of the trade.
        """
        return (self.sell.execution_price - self.buy.execution_price) / self.buy.execution_price

    def get_duration(self) -> datetime.timedelta:
        """Returns the duration of the trade.
        
        :return: datetime.timedelta object representing the duration of the trade.
        """
        return self.sell.execution_timestamp - self.buy.execution_timestamp


TradeSet = List[Trade]      # the ticker mapped to a list of trades


def get_best_strategy_and_account(results=AccountHistorySet) -> AccountHistory:
    """find the account with the best performance. Note that this assumes that all accounts have cashed out."""
    highest_value = 0
    best_result = None
    for r in results:
        if AccountHistory.final_value > highest_value:
            highest_value = r.final_value
            best_result = r
    return best_result


def func_for_list_if_length(list:List, func) -> Any:
    """Returns the result of the function if the list is not empty, otherwise returns 0."""
    if len(list) == 0: return 0
    return func(list)


def parse_order_history_into_trades(account:Account, price_history:pd.DataFrame|None=None) -> TradeSet:
    """Compiles the order history of an account into a list of trades.
    
    :param account: The account to parse the order history of.
    :param price_history: The price history of the account. If this is not None, then the max drawdown and max percent up
    :return: A list of trades.
    """
    
    trades:TradeSet = []
    organized_orders:Dict[str, Dict[str, List[OrderBase]]] = {}
    for o in account._order_history:
        if o.ticker not in organized_orders:
            organized_orders[o.ticker] = {'buys':[], 'sells':[]}
        if o.side == OrderSide.BUY:
            organized_orders[o.ticker]['buys'].append(o)
        else:   
            organized_orders[o.ticker]['sells'].append(o)

    for ticker in organized_orders.keys():
        organized_orders[ticker]['buys'].sort(key=lambda o: o.execution_timestamp)
        organized_orders[ticker]['sells'].sort(key=lambda o: o.execution_timestamp)

        for i in range(len(organized_orders[ticker]['sells'])):
            trades.append(Trade(organized_orders[ticker]['buys'][i], organized_orders[ticker]['sells'][i]))

    # parse out the max drawdown and max percent up for each trade
    trade_index = 0
    flag = False
    max_draw_down = 0
    max_percent_up = 0

    if price_history is not None:
        for ts, row in price_history.iterrows():
            if trade_index >= len(trades):
                break
            trade = trades[trade_index]
            # monitor if we are in a trade
            if not flag and ts > trade.buy.execution_timestamp:
                flag = True
            if flag and ts > trade.sell.execution_timestamp:
                flag = False
                trade.max_percent_up = max_percent_up
                trade.max_draw_down = max_draw_down
                trade_index += 1
            
            if flag:
                cur_percent_trade_value = (trade.buy.execution_price - row[trade.ticker+'_close']) / trade.buy.execution_price
                if cur_percent_trade_value > max_percent_up:
                    max_percent_up = cur_percent_trade_value
                if cur_percent_trade_value < max_draw_down:
                    max_draw_down = cur_percent_trade_value
    return trades


def print_account_stats(account:Account, underlying:CandleData=None):
    """Calculates many performance metrics for the account history and prints them to the console.
    
    :param account: The account to calculate the performance metrics of.
    :param underlying: The underlying price history of the account. If this is not None, then alpha is not calculated.
    """
    
    print(f'\n')
    logger.info(f'General performance metrics:')
    
    logger.info('Max drawdown: ${:.0f}, {:.1f}%'.format(get_max_drawdown(account), get_max_drawdown_as_percent(account)))
    vh = list(account.value_history.values())
    logger.info('Starting value: ${}; final value: ${:.0f}'.format(vh[0], vh[-1]))
    logger.info('Total return: {:.1f}%, annualized return {:.1f}%'.format(get_ROI(account), get_annualized_ROI(account)))

    # Get the buys and the sells
    n_buys = len([o for o in account._order_history if o.side == OrderSide.BUY])
    n_sells = len([o for o in account._order_history if o.side == OrderSide.SELL])
    
    print("\n")
    logger.info(f'Trade stats:')
    logger.info(f'# of buys: {n_buys}, # sells: {n_sells}.')
    
    # parse the buys and sells into trades and do basic analysis on them
    trades = parse_order_history_into_trades(account)
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
    logger.info('VWR: {:.3f}, VWR curve fit: {:.3f}'.format(get_vwr(account), get_curve_fit_vwr(account)))
    if underlying is not None:
        logger.info('Alpha relative to ticker {} {:.2f}%'.format(*get_alpha(account, underlying)))


def get_win_percentage(account:Account=None, trades=None) -> float:
    """Returns the win percentage of the account.
    
    :param account: The account to calculate the win percentage of.
    :param trades: The trades to calculate the win percentage of. If this is None, then the trades are parsed from the account.
    :return: The win percentage of the account.
    """
    if trades is None:
        trades = parse_order_history_into_trades(account)
    
    n_wins = len([t for t in trades if t.get_net_profit() > 0])
    if len(trades) == 0:
        return 0
    return 100*n_wins / len(trades)


def get_ROI(account:Account) -> float:
    """Returns the ROI of the account.
    
    :param account: The account to calculate the ROI of.
    :return: The ROI of the account as a percentage (ie 11 being 11%)
    """
    # Returns the percentage increase/decrease
    initial:float = list(account.value_history.values())[0] 
    final:float = list(account.value_history.values())[-1]
    return (final / initial - 1)*100


def get_annualized_ROI(account:Account) -> float:
    """Returns the annualized returns of the account. 
    
    Note that this is an approximation as it does not account for business days/weekends/holidays.
    
    :param account: The account to calculate the annualized ROI of.
    :return: The annualized ROI of the account as a percentage (ie 11 being 11%)
    """
    
    roi = get_ROI(account)

    end = list(account.value_history.keys())[-1]
    start = list(account.value_history.keys())[0]
    n_years = (end - start).total_seconds()/(60*60*24*365)
    annualized_roi = (1 + roi/100)**(1/n_years) - 1
    return annualized_roi*100


def get_expected_ROI(account:Account) -> float:
    """Fits a curve to the account value history and returns the expected return.
    
    :param account: The account to calculate the expected ROI of.
    :return: The expected ROI of the account as a percentage (ie 11 being 11%)
    """
    
    value_history = list(account.value_history.values())
    start_value = value_history[0]
    
    mean_return_per_timestep = fit_exponential_curve_fixed_start(value_history)
    end_val = start_value*np.exp(mean_return_per_timestep*len(value_history))
    return (end_val - start_value)/start_value*100


def get_expected_annualized_ROI(account:Account) -> float:
    """Fits a curve to the account value history and returns the expected annualized return.
    
    :param account: The account to calculate the expected annualized ROI of.
    :return: The expected annualized ROI of the account as a percentage (ie 11 being 11%)
    """
    value_history = list(account.value_history.values())
    start_value = value_history[0]
    
    end = list(account.value_history.keys())[-1]
    start = list(account.value_history.keys())[0]
    n_years = (end - start).total_seconds()/(60*60*24*365)
    
    mean_end_val = get_expected_ROI(account)
    mean_return = (mean_end_val -start_value)/start_value
    return 100*((1 + mean_return)**(1/n_years) - 1)
    

def get_max_drawdown(account:Account) -> float:
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


def get_max_drawdown_as_percent(account:Account) -> float:
    """Returns the maximum drawdown of the account as a percentage of the initial account value."""
    return 100*get_max_drawdown(account) / list(account.value_history.values())[0]
 
    
def fit_exponential_curve_fixed_start(y:list[float]) -> np.ndarray:
    """Fits an exponential curve to the data with the starting value being equal according to formula
    y = start_value * (1 + r)**t
    The fitting is done in the log space in order to give equal weighting to data over time. 
    
    Returns the mean growth rate per timestep."""
    
    start_value = y[0]
    time = range(len(y))

    def exponential_growth(t, r):
        return np.log(start_value) + t*np.log(1 + r)
        # return np.log(start_value) + y*np.log(t)

    popt, pcov = curve_fit(exponential_growth, time, np.log(y))     # your data x, y to fit
    r_mean = popt[0]                                        # average growth per timestep
    return r_mean


def get_vwr(account:Account, tau:float=4, stddev_max:float=0.2) -> float:
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
    value_history = list(account.value_history.values())
    start_val, end_val = value_history[0], value_history[-1]
    
    mean_return_per_timestep = (end_val/start_val)**(1/len(value_history)) - 1

    expected_end_val = start_val*(1 + mean_return_per_timestep)**len(value_history)
        
    # penalizing variance defined as account_value - mean_exponential growth
    zero_variability_returns = [start_val*np.exp(mean_return_per_timestep*i) for i in range(len(value_history))]
    logger.debug(f'vwr expected end val: {zero_variability_returns[-1]}')
    
    # difference between actual and zero-variability prices divided by zero-variability prices to normalize 
    log_difference = [v/v_zero_var - 1 for v, v_zero_var in zip(value_history, zero_variability_returns)]

    sigma_p = np.std(log_difference)    # *value_history[0]   # multiply by the starting value as we normalized this earlier, so sigma is in dollars
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


def get_curve_fit_vwr(account:Account, tau:float=4, stddev_max:float=0.2) -> float:
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
    
    mean_return_per_timestep = fit_exponential_curve_fixed_start(list(account.value_history.values()))
    
    value_history = list(account.value_history.values())
    start_value = value_history[0]
    # penalizing variance defined as account_value - mean_exponential growth
    zero_variability_returns = [start_value*np.exp(mean_return_per_timestep*i) for i in range(len(value_history))]
    logger.debug(f'Expected end val: {zero_variability_returns[-1]}') 
    
    # difference between actual and zero-variability prices divided by zero-variability prices to normalize
    normalized_differences = [v/v_zero_var - 1 for v, v_zero_var in zip(value_history, zero_variability_returns)]
    
    sigma_p = np.std(normalized_differences)    # * start_value
    if sigma_p > stddev_max: 
        logger.warning(f'VWR sigma_p is greater than the max allowed. Returning roi')
        return get_expected_ROI(account)
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


def get_alpha(account:Account, underlying:CandleData, rf:float=0) -> Tuple[str, float]:
    """Calculates the alpha of the account."""
    # alpha = (r - r_f) - beta * (r_m - r_f)
    # r = account return
    # r_f = risk free rate
    # beta = beta of the account
    # r_m = market return
    # r_f = risk free rate
    beta = 1

    roi = get_ROI(account)
    ticker = underlying.tickers[0]
    underlying_return = 100*(underlying.df[ticker+'_close'][-1] - underlying.df[ticker+'_close'][0]) / underlying.df[ticker+'_close'][0]
    alpha = roi - rf - beta*(underlying_return - rf)
    return ticker, alpha


def get_vwr_curve_alpha(account:Account, underlying:CandleData):
    vwr = get_curve_fit_vwr(account)
    