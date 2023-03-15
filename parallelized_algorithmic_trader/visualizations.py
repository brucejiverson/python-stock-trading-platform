import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import List, Optional
import numpy as np
import pandas as pd

from parallelized_algorithmic_trader.broker import SimulatedAccount, OrderSide
from parallelized_algorithmic_trader.data_management.market_data import CandleData
from parallelized_algorithmic_trader.util import get_logger

logger = get_logger(__name__)


def plot_backtest_results_mpf(candle_datas:List[CandleData], account:SimulatedAccount):
    """Plots the account history"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    df = candle_datas[0].df
    # now plot the orders from the account
    df['buys'] = np.nan
    df['sells'] = np.nan
    for o in account._order_history:
        if o.side == OrderSide.BUY:
            df.loc[o.execution_timestamp, 'buys'] = o.execution_price
        elif o.side == OrderSide.SELL:
            df.loc[o.execution_timestamp, 'sells'] = o.execution_price

    # scale the buys and sells up/down cause its prettier
    # df['buys'] = df['buys']*.999
    # df['sells'] = df['sells']*1.001
    apd = mpf.make_addplot(df['buys'], ax=ax1, type='scatter', marker='v', color='g')
    apd2 = mpf.make_addplot(df['sells'], ax=ax1, type='scatter', marker='^', color='r')

    # now plot the candle close data
    for candle_data in candle_datas:
        mpf.plot(candle_data.df, # the dataframe containing the OHLC (Open, High, Low and Close) data
            ax = ax1, # the matplotlib axes object to plot on
            # type='candle', # use candlesticks 
            type='line',
            volume=ax1.twinx(), # also show the volume
            style='yahoo',  # choose the yahoo style
            addplot=[apd, apd2],
        )
            # warn_too_much_data=len(df) + 1,

    ax1.title.set_text('Price History and Features With Order Overlay')

    # plot the account value over time
    ax2.plot(account.value_history.keys(), account.value_history.values(), label='account value', color='black')
    ax2.title.set_text(f'Account Value Over Time')
    fig.suptitle(f'parallelized_algorithmic_trader Backtest Results', fontsize=14, fontweight='bold')
    fig.autofmt_xdate()


def plot_backtest_results(
    feature_df:pd.DataFrame, 
    account:SimulatedAccount=None, 
    tickers:List[str]=None, 
    strategy_name:str='', 
    use_datetimes:bool=False,
    save_folder:str|None=None):
    """Plots the account history"""
    
    # figure out if we need 3 axes or 2
    features_for_special_axis = []
    features_for_price_axis = []
    candle_items = ('_low', '_high', '_open', '_close', '_volume')
    cols_to_ignore = [t+c for t in tickers for c in candle_items]
    cols_to_ignore.append('Source')
    equity_range = feature_df[f'{tickers[0]}_close'].max(), feature_df[f'{tickers[0]}_close'].min()

    for feature_name in feature_df.columns:
        if feature_name not in cols_to_ignore:
            feature_range = feature_df[feature_name].max(), feature_df[feature_name].min()
            # search for the first valid number of the feature
            for i in range(len(feature_df)):
                if not np.isnan(feature_df[feature_name].iloc[i]):
                    feature_start = feature_df[feature_name].iloc[i]
                    break
            feature_end = feature_df[feature_name].iloc[-1]
            
            feature_range_spans = feature_range[0] > equity_range[0] and feature_range[1] < equity_range[1]
            feature_range_contained = equity_range[0] > feature_start > equity_range[1] or equity_range[0] > feature_end > equity_range[1]
            # now get the config relating to this feature
            if feature_range_spans or feature_range_contained:
                features_for_price_axis.append(feature_name)
            else:
                features_for_special_axis.append(feature_name)

    n_axis = 1
    if len(features_for_special_axis) > 0:
        n_axis += 1
    if account is not None:
        n_axis += 1
    
    if n_axis == 1:
        fig, ax1 = plt.subplots()
    elif n_axis == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)


    if len(features_for_special_axis) > 0:
        if n_axis == 2:
            spec_ax = ax2
        else:
            spec_ax = ax3
        # plot these special features first
        for feature_name in features_for_special_axis:
            if 'OBV' in feature_name:
                # special case
                print(f'Warning: OBV is not supported for plotting')
                continue
            if use_datetimes:
                feature_df.reset_index().plot(y=feature_name, x='timestamp', kind='line', ax=spec_ax, label=feature_name)
            else:
                feature_df.reset_index().plot(y=feature_name, kind='line', ax=spec_ax, label=feature_name)
    
    if account is not None:
        # now plot the trades from the account
        buys = account.get_all_buys()
        sells = account.get_all_sells()
        
        buy_times = [o.execution_timestamp for o in buys]
        buy_values = [o.execution_price for o in buys]        
        
        sell_times = [o.execution_timestamp for o in sells]
        sell_values = [o.execution_price for o in sells]
        
        if not use_datetimes:
            sell_times = [feature_df.index.get_loc(t) for t in sell_times]
            buy_times = [feature_df.index.get_loc(t) for t in buy_times]
        
        # scale the buys and sells up/down cause its prettier
        buy_values = [b*.9995 for b in buy_values]
        sell_values = [s*1.0005 for s in sell_values]
    
        # now plot the orders
        ax1.scatter(sell_times, sell_values, marker='v', c='r', zorder=4, label='sell')
        ax1.scatter(buy_times, buy_values, marker='^', c='g', zorder=4, label='buy')

    # now plot the candle close data for each symbol
    for i, t in enumerate(tickers):
        if i == 0: kwargs = {'color':'black'}
        else: kwargs = {}
        
        # col_name = f'{t}_close'
        # # sort through the data for jumps, plotting every time.
        # res = resolution.get_as_minutes()
        # previous_break = feature_df.index[0]
        # for i, (ts, row) in enumerate(feature_df.iterrows()):
        #     if ts == 0: continue
        #     last_timestamp = feature_df.index[i-1]
        #     delta = ts - last_timestamp
        #     if delta > pd.Timedelta(f'{2*res} minute'):
        #         # now plot up to the previous timestamp
        #         section = feature_df.loc[previous_break:last_timestamp]
        #         ax1.plot(section.index, section[col_name], label=t, **kwargs)

        if use_datetimes:
            feature_df[f'{t}_close'].plot(ax=ax1, label=t, **kwargs)
        else:
            feature_df.reset_index()[f'{t}_close'].plot(ax=ax1, label=t, **kwargs)

    for feature_name in features_for_price_axis:
        if use_datetimes:
            if 'minima' in feature_name or 'maxima' in feature_name:
                feature_df.reset_index().plot.scatter(x='timestamp', y=feature_name, ax=ax1, label=feature_name, s=25, c='b')
            else:
                feature_df.plot(y=feature_name, label=feature_name, ax=ax1)
        else:
            if 'minima' in feature_name or 'maxima' in feature_name:
                print(f'Plotting feature {feature_name} as scatter')
                feature_df.reset_index().reset_index().plot.scatter(x='index', y=feature_name, ax=ax1, label=feature_name, s=25, c='b')
            else:
                feature_df.reset_index()[feature_name].plot(ax=ax1, label=feature_name)

    ax1.legend()
    ax1.set_ylabel('Price (USD)')
    ax1.title.set_text('Price History and Features With Order Overlay')

    if account is not None:
        # plot the account value over time
        if use_datetimes:
            ax2.plot(account.value_history.keys(), account.value_history.values(), label='account value', color='black')
        else:
            ax2.plot(range(len(account.value_history)), account.value_history.values(), label='account value', color='black')
        ax2.title.set_text(f'Account Value Over Time')
        ax2.set_ylabel('Account Value (USD)')
    ax2.tick_params(
        bottom=use_datetimes,
        labelbottom=use_datetimes)
    fig.suptitle(f'Backtest Results for {strategy_name.upper()}', fontsize=14, fontweight='bold')
    fig.autofmt_xdate()
    
    if save_folder:
        fig.savefig(f'{save_folder}/{strategy_name}_backtest.png')
    return fig, (ax1, ax2)


def plot_underwater(account:SimulatedAccount, use_datetimes:bool=False, save_folder:str=None):
    """Plots the underwater curve"""
    # get the underwater curve
    latest_high = 0
    underwater = []
    times = []
    for time, value in account.value_history.items():
        if value > latest_high:
            latest_high = value
        underwater.append((value - latest_high)/latest_high)
        times.append(time)

    df = pd.DataFrame({'underwater': underwater, 'timestamp': times}, index=times)
    # plot the underwater curve
    if use_datetimes:
        df.plot(y='underwater', x='timestamp', kind='line', label='underwater', color='black')
    else:
        df.plot(y='underwater', kind='line', label='underwater', color='black')
    plt.title('Underwater Curve')
    plt.ylabel('Underwater (%)')

    # fill in the area under the curve
    plt.fill_between(df.index, df['underwater'], color='red', alpha=0.5)

    if save_folder:
        plt.savefig(f'{save_folder}/under_water_curve.png')


def plot_cumulative_returns(account:SimulatedAccount, underlying:Optional[pd.DataFrame]=None, use_datetimes:bool=False, save_folder:str=None):
    """Plots the cumulative returns of the account"""
    # get the cumulative returns
    cumulative_returns = []
    times = []
    starting_cash = list(account.value_history.values())[0]
    for time, value in account.value_history.items():
        cumulative_returns.append(value / starting_cash)
        times.append(time)

    fig, ax = plt.subplots(1, 1)  # Create the figure
    df = pd.DataFrame({'cumulative_returns': cumulative_returns, 'timestamp': times})
    # plot the cumulative returns
    if use_datetimes:
        df.plot(y='cumulative_returns', x='timestamp', ax=ax, kind='line', label='cum. returns', color='black')
    else:
        df.plot(y='cumulative_returns', ax=ax, kind='line', label='cum. returns', color='black')

    plt.title('Cumulative Returns')
    plt.ylabel('Cumulative Returns (%)')

    # fill in the area under the curve
    # plt.fill_between(df.index, df['cumulative_returns'], color='green', alpha=0.5)

    # plot the underlying cumulative returns
    if underlying is not None:
        # figure out the close column name
        close_cols = [c for c in underlying.columns if '_close' == c.lower()[-6:]]
        for col in close_cols:
            underlying_cum_returns = underlying[col] / underlying[col].iloc[0]
            df = pd.DataFrame({'cumulative_returns': underlying_cum_returns, 'timestamp': underlying.index})
            if use_datetimes:
                df.plot(y='cumulative_returns', x='timestamp', ax=ax, kind='line', label=f'{col} cum. returns')
            else:
                df.drop(columns=['timestamp']).reset_index().reset_index().plot(x='index', y='cumulative_returns', ax=ax, kind='line', label=f'{col} cum. returns')
    # fill the area between the two plots
    # plt.fill_between(df.index, df['cumulative_returns'], df[t+'cumulative_returns'], color='green', alpha=0.5)

    if save_folder:
        plt.savefig(f'{save_folder}/cumulative_returns.png')

def visual_analysis_of_trades(account:SimulatedAccount, price_history:pd.DataFrame, save_folder:str=None):
    plot_trade_durations_vs_profits(account, save_folder)
    # plot_trade_profit_hist(account)
    # plot_trade_max_theoretical_profit_vs_drawdown(account, price_history)

    
def plot_trade_profit_hist(account:SimulatedAccount) -> None:
    """Makes a histogram of the profits of the trades in an account."""
    trades = account.get_trades()
    profits = [t.get_profit_percent() for t in trades]
    if len(trades) > 10:
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.hist(profits, bins=20)
        ax1.set_title('Trade Profits')
        ax1.set_xlabel('Profit and loss (%)')
        ax1.set_ylabel('# trades in this profit range')

        # now plot the trade durations as a histogram on ax2
        durations = [t.get_duration().total_seconds()/(60*60*24) for t in trades]
        ax2.hist(durations, bins=20)
        ax2.set_title('Histogram of Trade Durations')
        ax2.set_xlabel('Duration (days)')
        ax2.set_ylabel('# trades in this duration range')
    else:
        logger.warning('Not enough trades to plot a histogram of trade profits')
    return profits


def plot_trade_max_theoretical_profit_vs_drawdown(account:SimulatedAccount, price_history:pd.DataFrame) -> None:
    """Makes a histogram of the profits of the trades in an account."""
    trades = account.get_trades()
    theo_profits = [t.max_percent_up for t in trades]
    if len(trades) > 10:
        # scatter plot the theoretical max profits with perfect exit within the trade timeframe vs max drawdown
        fig, ax = plt.subplots(1,1)
        ax.scatter(theo_profits, [t.max_draw_down for t in trades])
        ax.set_title('Theoretical Max Profit vs Max Drawdown')
        ax.set_xlabel('Theoretical Max Profit (%)')
        ax.set_ylabel('Max Drawdown (%)')
        
        # fig, (ax1, ax2) = plt.subplots(2,1)
        # ax1.hist(theo_profits, bins=20)
        # ax1.set_title('Trade Max Percent Up')
        # ax1.set_xlabel('Profit and loss (%)')
        # ax1.set_ylabel('# trades in this profit range')

        # # now plot the max drawdowns as a histogram on ax2
        # max_drawdowns = [t.max_draw_down for t in trades]
        # ax2.hist(max_drawdowns, bins=20)
        # ax2.set_title('Histogram of Trade Max Drawdowns')
        # ax2.set_xlabel('Max Drawdown (%)')
        # ax2.set_ylabel('# trades in this max drawdown range')

        # # now make a new one with a ratio of the profits and drawdowns
        # fig, ax = plt.subplots(1,1)
        # ratios = [-t.max_percent_up / t.max_draw_down for t in trades if t.max_draw_down != 0]
        # ax.hist(ratios, bins=20)
        # ax.set_title('Trade Max Percent Up / Max Drawdown')
        # ax.set_xlabel('Profit and loss / Max Drawdown (%)')
        # ax.set_ylabel('# trades in this profit range')


def plot_trade_durations_vs_profits(account:SimulatedAccount, save_folder:str|None=None) -> None:
    """Plots the trade durations vs the trade profits"""
    trades = account.get_trades()
    profits = [t.get_profit_percent() for t in trades]
    durations = [t.get_duration().total_seconds()/(60*60*24) for t in trades]
    fig, ax = plt.subplots(1,1)
    ax.scatter(durations, profits)
    ax.set_title('Trade Profits vs Trade Durations')
    ax.set_xlabel('Duration (days)')
    ax.set_ylabel('Profit and loss (%)')

    if save_folder:
        plt.savefig(f'{save_folder}/trade_durations_vs_profits.png')