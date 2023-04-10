from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import Formatter
import matplotlib.backends.backend_pdf as pdf_backend
import os

from parallelized_algorithmic_trader.trading.simulated_broker import SimulatedAccount
from parallelized_algorithmic_trader.util import get_logger

logger = get_logger(__name__)


class TimeAxisFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m-%d %H:%M'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''

        return self.dates[ind].strftime(self.fmt)


def create_pdf_performance_report(
    performance_metrics: Dict[str, str], 
    df: pd.DataFrame, 
    account: SimulatedAccount,
    tickers: List[str] = None, 
    strategy_name: str = '', 
    save_folder: str | None = None
) -> None:
    
    images = [
        create_table_for_performance_metrics(performance_metrics),
        plot_backtest_results(df, account, tickers, strategy_name),
        plot_trade_profit_hist(account),
        plot_underwater(account)
    ]
    
    fname = f'{strategy_name}_performance_report.pdf'
    fpath = os.path.join(save_folder, fname) if save_folder is not None else fname
    
    with pdf_backend.PdfPages(fpath) as pdf:
        pdf.infodict()['Title'] = fname
        
        for fig, ax in images:
            pdf.savefig(fig)
            

def create_table_for_performance_metrics(performance_metrics:Dict[str, str], input_ax:plt.Axes=None) -> Tuple[plt.Figure, plt.Axes]:
    """Create a plot of the performance metrics text. Performance metrics is a dictionary with the row labels as the keys and the cell contents are the values."""
    if input_ax is None:
        fig, ax = plt.subplots()
        fig.tight_layout()
    else:
        ax = input_ax
        
    ax.axis('off')
    ax.axis('tight')
    cell_text = [[val] for val in performance_metrics.values()]
    ax.table(cellText=cell_text, rowLabels=list(performance_metrics.keys()), loc='center')
    # add a title
    ax.set_title(f'Performance Metrics')
    
    if input_ax is None:
        fig.tight_layout()
    return fig, ax

    
def parse_columns_by_feature_range(df:pd.DataFrame, tickers:List[str]) -> Tuple[List[str], List[str]]:
    """Returns the names of the features that should be plotted on the price axis"""
    
    candle_suffixes = ('_low', '_high', '_open', '_close', '_volume')
    cols_to_ignore = [t+c for t in tickers for c in candle_suffixes]
    cols_to_ignore.append('Source')

    features_for_price_axis = []
    features_for_special_axis = []
    
    # parse out the features that should be plotted on the price axis and the features that are in different ranges
    equity_range = df[f'{tickers[0]}_close'].max(), df[f'{tickers[0]}_close'].min()
    for feature_name in df.columns:
        if feature_name in cols_to_ignore:
            continue
        
        if any([txt in feature_name.lower() for txt in ('minima', 'maxima')]):
            features_for_price_axis.append(feature_name)
            continue
        
        feature_range = df[feature_name].max(), df[feature_name].min()
        # search for the first valid number of the feature
        for i in range(len(df)):
            if not np.isnan(df[feature_name].iloc[i]):
                feature_start = df[feature_name].iloc[i]
                break
        feature_end = df[feature_name].iloc[-1]
        
        feature_range_spans = feature_range[0] > equity_range[0] and feature_range[1] < equity_range[1]
        feature_range_contained = equity_range[0] > feature_start > equity_range[1] or equity_range[0] > feature_end > equity_range[1]
        # now get the config relating to this feature
        if feature_range_spans or feature_range_contained:
            features_for_price_axis.append(feature_name)
        else:
            features_for_special_axis.append(feature_name)
            
    return features_for_price_axis, features_for_special_axis


def plot_backtest_results(df:pd.DataFrame, account:SimulatedAccount, tickers:List[str]=None, strategy_name:str='') -> Tuple[plt.Figure, plt.Axes]:
    """Plots the account history, price history of the assets being traded, features, and performance metrics"""
    
    features_for_special_axis = []
    features_for_price_axis = []
    features_for_price_axis, features_for_special_axis = parse_columns_by_feature_range(df, tickers)

    layout = [["price history", "price history"]]
    if len(features_for_special_axis) > 0:
        layout.append(['special features', 'special features'])
    layout.extend([
        ["cumulative returns", "cumulative returns"], 
        ])
        # ["trade hist", "unused"]
    
    fig, axes = plt.subplot_mosaic(layout)
    
    plot_price_history(df, tickers, features_for_price_axis, axes['price history'])
    plot_cumulative_returns(account, df, ax=axes['cumulative returns'])
    # plot_trade_profit_hist(account, ax=axes['trade hist'])

    # plot features not on the price axis
    if len(features_for_special_axis) > 0:
        # plot these special features first
        for feature_name in features_for_special_axis:
            if 'OBV' in feature_name:
                logger.warning(f'OBV is not supported for plotting')
                continue
            df.reset_index().plot(y=feature_name, kind='line', ax=axes['special features'], label=feature_name)

    fig.suptitle(f'Backtest Results for {strategy_name.upper()}', fontsize=14, fontweight='bold')
    fig.autofmt_xdate()
    fig.tight_layout()
    
    return fig, axes
    

def plot_underwater(account:SimulatedAccount) -> Tuple[plt.Figure, plt.Axes]:
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
    fig, ax = plt.subplots()
    formatter = TimeAxisFormatter(df.index)
    ax.xaxis.set_major_formatter(formatter)
    
    df.plot(y='underwater', kind='line', label='underwater', color='black', ax=ax)
    plt.title('Underwater Curve')
    plt.ylabel('Underwater (%)')

    # fill in the area under the curve
    plt.fill_between(df.index, df['underwater'], color='red', alpha=0.5)

    # if save_folder:
    #     fig = plt.gcf()
    #     fig.set_size_inches(12, 8)
    #     plt.savefig(f'{save_folder}/under_water_curve.png')
    return fig, ax


def plot_price_history(df:pd.DataFrame, tickers:List[str], features_to_plot:List[str]=[], input_ax:plt.Axes=None):
    
    if input_ax is None:
        fig, ax = plt.subplots()
    else: 
        ax = input_ax
    formatter = TimeAxisFormatter(df.index)
    ax.xaxis.set_major_formatter(formatter)
    # axes['cumulative returns'].xaxis.set_major_formatter(formatter)

    # plot the candle close data for each symbol
    for i, t in enumerate(tickers):
        if i == 0: kwargs = {'color':'black'}
        else: kwargs = {}
        
        df.reset_index()[f'{t}_close'].plot(ax=ax, label=t, **kwargs)

    # plot the features that are on the price axis
    for feature_name in features_to_plot:
        # special case for scatter plots
        if any([txt in feature_name.lower() for txt in ('minima', 'maxima')]):
            df[feature_name] = df[feature_name].replace(0, np.nan)
            ax.scatter(x=range(len(df)), y=df[feature_name], label=feature_name, s=25)
        else:
            df.reset_index()[feature_name].plot(ax=ax, label=feature_name)

    ax.legend()
    ax.set_ylabel('Price (USD)')
    ax.title.set_text('Price History and Features With Order Overlay')
    if input_ax is None:
        fig.autofmt_xdate()
        

def plot_cumulative_returns(
    account:SimulatedAccount, 
    underlying:Optional[pd.DataFrame]=None, 
    ax:Optional[plt.Axes]=None, 
    save_folder:str=None) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the cumulative returns of the account"""
            
    account_cum_returns = []
    times = []
    starting_cash = list(account.value_history.values())[0]
    for time, value in account.value_history.items():
        account_cum_returns.append(100*value / starting_cash)
        times.append(time)

    if ax is None:
        fig, ax = plt.subplots(1, 1)  # Create the figure

    formatter = TimeAxisFormatter(times)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title('Cumulative Returns')
    # ax.set_ylabel('Cumulative Returns (%)')
    
    df = pd.DataFrame({'account_cum_returns': account_cum_returns, 'timestamp': times})
    df['account_cum_returns'].plot(ax=ax, label='account_cum_returns', color='black')

    # plot the underlying cumulative returns
    if underlying is not None:
        close_col_names = [c for c in underlying.columns if '_close' == c.lower()[-6:]]
        for col in close_col_names:
            col_name = f'{col}_cum_returns'
            if len(df) != len(underlying[col]):
                logger.warning(f'Length of underlying price history does not match length of account value history. Skipping {col}.')
                continue
            
            df[col_name] = [100*u/underlying[col].iloc[0] for u in underlying[col]]
            df[col_name].plot(ax=ax, label=f'{col} cum. returns')

        # plot the mean of the underlying cumulative returns
        if len(close_col_names) > 1:
            df['mean_underlying_cum_returns'] = df[[f'{c}_cum_returns' for c in close_col_names]].mean(axis=1)
            df['mean_underlying_cum_returns'].plot(ax=ax, label='Equal weight cum. returns')
            
    if ax is None:
        fig.autofmt_xdate()
    ax.legend()
    

def plot_trade_profit_hist(account:SimulatedAccount, ax:Optional[plt.Axes]=None) -> Tuple[plt.Figure, plt.Axes]:
    """Makes a histogram of the profits of the trades in an account."""
    trades = account.get_trades()
    profits = [t.get_profit_percent() for t in trades]
    
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.hist(profits, bins=20)
    ax.set_title('Distribution of Trade Profits')
    ax.set_xlabel('Profit and loss (%)')
    ax.set_ylabel('# trades')
    return fig, ax


def plot_trade_duration_hist(account:SimulatedAccount, ax:Optional[plt.Axes]=None):
    trades = account.get_trades()
    durations = [t.get_duration().total_seconds()/(60*60*24) for t in trades]
    ax.hist(durations, bins=20)
    ax.set_title('Histogram of Trade Durations')
    ax.set_xlabel('Duration (days)')
    ax.set_ylabel('# trades in this duration range')


def plot_trade_durations_vs_profits(account:SimulatedAccount):
    """Plots the trade durations vs the trade profits"""
    trades = account.get_trades()
    profits = [t.get_profit_percent() for t in trades]
    durations = [t.get_duration().total_seconds()/(60*60*24) for t in trades]
    fig, ax = plt.subplots(1,1)
    ax.scatter(durations, profits)
    ax.set_title('Trade Profits vs Trade Durations')
    ax.set_xlabel('Duration (days)')
    ax.set_ylabel('Profit and loss (%)')

        
def distribution_of_monthly_returns(account:SimulatedAccount):
    """Plots the distribution of monthly returns as a histogram"""
    # calculate the monthly returns from the account history
    
    monthly_returns = []
    month_start = None
    for timestamp, value in account.value_history.items():
        if month_start is None:
            month_start = timestamp
            continue
        if timestamp.month != month_start.month:
            # calculate the monthly return
            monthly_returns.append((value - account.value_history[month_start])/account.value_history[month_start])
            month_start = timestamp

    if len(monthly_returns) < 12:
        logger.warning('Not enough data to plot a distribution of monthly returns')
        return

    fig, ax = plt.subplots(1,1)
    ax.hist(monthly_returns, bins=20)
    ax.set_title('Distribution of Monthly Returns')
    ax.set_xlabel('Monthly Return (%)')
    ax.set_ylabel('# months in this return range')

        