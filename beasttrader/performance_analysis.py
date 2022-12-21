from __future__ import annotations
import random
from dataclasses import dataclass
import datetime
from typing import List, Dict
import numpy as np
import logging
import pandas as pd
from scipy.optimize import curve_fit

from beasttrader.market_data import CandleData, TemporalResolution
from beasttrader.exchange import Account, OrderSide, OrderBase
from beasttrader.backtest_tools import SimulationResult, SimulationResultSet


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Trade:
    buy:OrderBase
    sell:OrderBase
    max_draw_down:float|None = None
    max_percent_up:float|None = None

    @property
    def ticker(self) -> str:
        return self.buy.ticker

    def get_net_profit(self) -> float:
        return self.sell.get_transaction_value() - self.buy.get_transaction_value()
    
    def get_profit_percent(self) -> float:
        return (self.sell.execution_price - self.buy.execution_price) / self.buy.execution_price

    def get_duration(self) -> datetime.timedelta:
        return self.sell.execution_timestamp - self.buy.execution_timestamp


TradeSet = List[Trade]      # the ticker mapped to a list of trades


def get_best_strategy_and_account(results=SimulationResultSet) -> SimulationResult:
    """find the account with the best performance. Note that this assumes that all accounts have cashed out."""
    highest_value = 0
    best_result = None
    for r in results:
        if SimulationResult.final_value > highest_value:
            highest_value = r.final_value
            best_result = r
    return best_result


def compile_order_history_into_trades(account:Account, price_history:pd.DataFrame|None=None) -> TradeSet:
    """Compiles the order history of an account into a list of trades."""
    
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


def print_account_general_stats(account:Account, underlying:CandleData=None):
    print(f'\n')
    logger.info(f'General performance metrics:')
    logger.info('Max drawdown: ${:.2f}'.format(get_max_drawdown(account)))
    logger.info('Max drawdown percent: {:.2f} %'.format(get_max_drawdown_percent(account)))
    vh = list(account.value_history.values())
    logger.info('Starting value: ${}; final value: ${:.2f}'.format(vh[0], vh[-1]))
    logger.info('Total return: {:.2f} %'.format(get_ROI(account)))
    if underlying is not None:
        logger.info('Annualized return: {:.2f} %'.format(get_annualized_returns(account)))

    # Get the buys and the sells
    buys = [o for o in account._order_history if o.side == OrderSide.BUY]
    sells = [o for o in account._order_history if o.side == OrderSide.SELL]
    
    print(f'\n')
    logger.info(f'Trade stats:')
    n_buys = len(buys)
    n_sells = len(sells)
    logger.info(f'Number of buys: {n_buys}.')
    logger.info(f'Number of sells: {n_sells}.')
    delta = list(account.value_history.keys())[-1] - list(account.value_history.keys())[0]
    n_days = delta.total_seconds() / (60*60*24)
    logger.info(f'Mean # buys per day {n_buys/n_days:.2f} over {n_days:.1f} days.')
    # parse the buys and sells into trades and do basic analysis on them
    trades = compile_order_history_into_trades(account)

    trade_profits = [t.get_net_profit() for t in trades]
    mean_profit = np.mean(trade_profits) if len(trade_profits) > 0 else 0
    logger.info("Mean trade profit: ${:.2f}".format(mean_profit))
    trade_profits_perc = [t.get_profit_percent() for t in trades]
    mean_profit_percent = np.mean(trade_profits_perc) if len(trade_profits_perc) > 0 else 0
    logger.info("Mean trade profit percent: {:.2f}%".format(mean_profit_percent*100))
    trade_durations = [t.get_duration().total_seconds() for t in trades]
    mean_duration = np.mean(trade_durations)/60 if len(trade_durations) > 0 else 0
    logger.info('Mean time in the market duration: {:.2f} minutes'.format(mean_duration))
    buy_to_buy_times:List[pd.Timedelta] = [t1.buy.execution_timestamp - t2.buy.execution_timestamp for t1, t2 in zip(trades[1:], trades[:-1])]
    mean_time_from_buy_to_buy = np.mean([t.total_seconds() for t in buy_to_buy_times])/60 if len(buy_to_buy_times) > 0 else 0
    logger.info('Mean time from buy to buy: {:.0f} minutes, {:.2f} days'.format(mean_time_from_buy_to_buy, mean_time_from_buy_to_buy/(60*24)))
    
    logger.info('Win rate: {:.1f} %'.format(get_win_percentage(trades=trades)))
    if underlying is not None:
        logger.info('Alpha: {:.2f}%'.format(get_alpha(account, underlying)))
    logger.info('')


def get_win_percentage(account:Account=None, trades=None) -> float:
    """Returns the win percentage of the account."""
    if trades is None:
        trades = compile_order_history_into_trades(account)
    
    n_wins = len([t for t in trades if t.get_net_profit() > 0])
    if len(trades) == 0:
        return 0
    return 100*n_wins / len(trades)


def score_trading_frequency(account: Account, min_n_trades_to_penalize:float=40, scale:float=100) -> float:
    """Scores an account based on how frequently it trades on a scale from 0 to 100, penalizing infrequent trades.
    
    min_n_trades_to_penalize: should be in units of trades/unit time of the simulation data
    
    The score is calculated linearly from 0 to 100, with 0 being 0 trades/time and 100 being >= min_n_trades_to_penalize.

    The idea here is that if youre algorithm only trades a few times, it is almost certainly overfitting the data, not learning 
    generalization. It doesn't actually matter how often we are trading, just so long as we have enough trades happen in our 
    training dataset. Therefore, agents who trade infrequently are penalized, but agents who trade very frequently are scored
    similarly to agents who trade moderately frequently."""
    orders = account._order_history
    n_orders = len(orders)
    return scale if n_orders >= min_n_trades_to_penalize else scale*(n_orders / min_n_trades_to_penalize)


def _score_return_on_scale(self, days_in_play_through:int, best_expected_ann_return:float=0.25, scale:float=100) -> float:
    ann_ret = self._get_est_annual_return(days_in_play_through=days_in_play_through)
    logger.info(f'annualized return: {ann_ret}')
    return scale * ann_ret / best_expected_ann_return 

    
def get_random_fitness_func_type(weights:dict[function,float]) -> function:
    return random.choices(list(weights.keys()), list(weights.values()))[0]


def get_ROI(account:Account) -> float:
    # Returns the percentage increase/decrease
    initial:float = list(account.value_history.values())[0] 
    final:float = list(account.value_history.values())[-1]
    return (final / initial - 1)*100


def get_max_drawdown(account:Account) -> float:
    """Returns the maximum drawdown of the account in dollars."""
    latest_high = 0
    max_drawdown = 0
    for value in account.value_history.values():
        if value > latest_high:
            latest_high = value
        drawdown = latest_high - value
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def get_max_drawdown_percent(account:Account) -> float:
    return 100*get_max_drawdown(account) / list(account.value_history.values())[0]


def count_business_days_in_range(start:datetime.datetime, end:datetime.datetime) -> int:
    """Parse the amount of trading time between two dates.

    Args:
        start (datetime.datetime): The start date.
        end (datetime.datetime): The end date.

    Returns:
        float: The amount of trading time in days.
    """

    # assumes that the dates are full and the end and start are included
    day_counter = 0
    while 1:
        if start.weekday() < 5:
            day_counter += 1
        start += datetime.timedelta(days=1)
        if start > end:
            break
    return day_counter


def get_annualized_returns(account:Account, business_days_only=True) -> float:
    """Returns the annualized returns of the account"""
        # units per year
    if business_days_only:
        _TANN = {
            TemporalResolution.MINUTE: 252*(24-8)*60,
            TemporalResolution.HOUR: 252*(24-8),
            TemporalResolution.DAY: 252,
            TemporalResolution.WEEK: 52,
            TemporalResolution.MONTH: 12,
        }
        n_per_day = {
            TemporalResolution.MINUTE: 60*16,
            TemporalResolution.HOUR: 16,
            TemporalResolution.DAY: 1,
            TemporalResolution.WEEK: 1/5,
            TemporalResolution.MONTH: 1/21,
        }    

    else:
        _TANN = {
            TemporalResolution.MINUTE: 365*24*60,
            TemporalResolution.HOUR: 365*24,
            TemporalResolution.DAY: 365,
            TemporalResolution.WEEK: 52,
            TemporalResolution.MONTH: 12,
        }

    # tau is - rate at which weighting falls with increasing variability  (investor tolerance)
    # r_norm - normalized return (avg log return, normalized to simple annualized return)
    # stddev_max - maximum acceptable σP  (investor limit)
    # timesteps_per_year = _TANN[data_res]
    hist = list(account.value_history.values())

    if business_days_only:
        trading_days = count_business_days_in_range(list(account.value_history.keys())[0], list(account.value_history.keys())[-1])     # days
        n_years:float = trading_days/252
    else:
        delta = list(account.value_history.keys())[-1] - list(account.value_history.keys())[0]
        n_years:float = delta.days / 365

    annualized_return = (hist[-1] / hist[0])**(1/n_years) - 1
    annualized_return_log = np.log(hist[-1] / hist[0]) / n_years

    # trading_days = trading_days * n_per_day[data_res]
    # r_mean = np.log(hist[-1] / hist[0]) / trading_days     # average log return per timestep
    # r_norm = 100*(np.exp(r_mean*timesteps_per_year) - 1)                # annualized return
    
    return annualized_return


def fit_exponential_curve_fixed_start(y:list[float]) -> np.ndarray:
    """Fits an exponential curve to the data with the starting value being equal. 
    The fitting is done in the log space in order to give equal weighting to data over time. 
    
    Returns the mean growth rate per timestep."""
    start_value = y[0]
    t = range(len(y))

    def exponential_growth(x, y):
        return y*np.log(x) + np.log(start_value)

    popt, pcov = curve_fit(exponential_growth, t, y)    # your data x, y to fit
    r_mean = popt[0]                                    # average growth per timestep
    return r_mean


def get_custom_vwr(account:Account, data_res:TemporalResolution, tau:float=4, stddev_max:float=0.2, business_days_only=True) -> float:
    """Calculates the variability weighted return for the account."""
    '''Variability-Weighted Return: Better SharpeRatio with Log Returns
    Alias:
      - VariabilityWeightedReturn
    See:
      - https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
    Params:
      - ``tau`` (default: ``2.0``)
         - rate at which weighting falls with increasing variability  (investor tolerance)
      - ``sdev_max`` (default: ``0.20``)
        maximum acceptable σP  (investor limit). The stddev of the difference between value and zero variance expected value
    '''

    value_history = list(account.value_history.values())
    start_value = value_history[0]
    
    mean_return_per_timestep = fit_exponential_curve_fixed_start(value_history)
    mean_end_val = start_value*np.exp(mean_return_per_timestep*len(value_history))
    
    n_years = (account.value_history.keys()[-1] - account.value_history.keys()[0]).years
    r_mean_per_yr = (mean_end_val/start_value)**(1/n_years) - 1
    print(f'n_years: {n_years}, annualized: {r_mean_per_yr}')
    
    # penalizing variance defined as account_value - mean_exponential growth
    zero_variability_returns = [start_value*np.exp(mean_return_per_timestep*i) for i in range(len(value_history))]
    # difference between actual and zero-variability prices divided by zero-variability prices to normalize 
    difference = [v/v_zero_var - 1 for v, v_zero_var in zip(value_history, zero_variability_returns)]

    sigma_p = np.std(difference)
    if sigma_p > stddev_max: return 0
    
    vwr = start_value*np.exp(mean_return_per_timestep*len(value_history)) * ((1 + sigma_p/stddev_max)**tau)
    return vwr if vwr > 0 else start_value*np.exp(mean_return_per_timestep*len(value_history)) 


def get_alpha(account:Account, underlying:CandleData, rf:float=0) -> float:
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
    logger.info(F'Using ticker: {ticker} for alpha calculation')
    underlying_return = 100*(underlying.df[ticker+'_close'][-1] - underlying.df[ticker+'_close'][0]) / underlying.df[ticker+'_close'][0]
    alpha = roi - rf - beta*(underlying_return - rf)
    return alpha

