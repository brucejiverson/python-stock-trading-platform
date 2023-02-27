import logging
import datetime
import matplotlib.pyplot as plt
import numpy as np

from parallelized_algorithmic_trader import polygon_io as po
from parallelized_algorithmic_trader.broker import TemporalResolution
from parallelized_algorithmic_trader.performance_analysis import *

if __name__ == "__main__":
    
    root_logger = get_logger('pat')
    root_logger.setLevel(logging.DEBUG)
    
    # get some data
    n_days = 30*12*.5
    end = datetime.datetime(2022, 12, 15)
    tickers = ['SPY']
    candle_data = po.get_candle_data(tickers, end-datetime.timedelta(days=n_days), end, TemporalResolution.MINUTE)
    
    logger = get_logger('pat.performance_analysis')
    value_history = candle_data.df['SPY_close'].values
    t = range(len(value_history))
    
    logger.info(f'Starting value of asset: ${candle_data.df.iloc[0]["SPY_close"]}. Ending value: ${candle_data.df.iloc[-1]["SPY_close"]}')

    # percentage change
    logger.info(f'Percentage change: {100*(value_history[-1]/value_history[0]- 1):.3f}%')
    r_mean = fit_exponential_curve_fixed_start(value_history)
    logger.info(f'Mean return per day: {100*r_mean:.4f}%')
    
    # annualized returns
    mean_exponential = [value_history[0]*(1 + r_mean)**(t_i) for t_i in t]
    mean_end_val = mean_exponential[-1]
    n_years = (candle_data.df.index[-1] - candle_data.df.index[0]).total_seconds()/(60*60*24*365)
    mean_return = (mean_end_val - value_history[0])/value_history[0]
    mean_return_annualized = (1 + mean_return)**(1/n_years) - 1
    
    r_mean_per_yr = (mean_end_val/value_history[0])**(1/n_years) - 1
    logger.info(f'n_years: {n_years:.2f}, mean return: {100*mean_return}%, annualized returns: {100*r_mean_per_yr:.3f}%')
    
    # volatility
    stddev_max = 0.15
    tau = 4
    # penalizing variance defined as account_value - mean_exponential growth
    zero_variance_returns = [value_history[0]*np.exp(r_mean*i) for i in range(len(value_history))]
    # difference between actual and zero-variability prices divided by zero-variability prices to normalize 
    difference = [v/v_zero_var - 1 for v, v_zero_var in zip(value_history, zero_variance_returns)]

    sigma_p = np.std(difference)
    
    vwr = mean_return * ((1 + sigma_p/stddev_max)**tau)
    logger.info(f'sigmap: {sigma_p:.3f}, vwr: {vwr:.3f}')
    
    # plot the original data
    fig, ax = plt.subplots(1,1)
    candle_data.df.plot(y=f'SPY_close', ax=ax, label='SPY')
    candle_data.df['mean_exponential'] = mean_exponential
    candle_data.df['mean_exponential_continuous'] = value_history[0]*np.exp(r_mean*t)

    # now plot the exponential curve
    candle_data.df.plot(y='mean_exponential', ax=ax, label='Exponential Curve')
    candle_data.df.plot(y='mean_exponential_continuous', ax=ax, label='Exponential Curve (continuous)')
    
    ax.set_ylabel('Price (USD)')
    ax.title.set_text('Price History By Index')
    fig.suptitle(f'Data inspection for {tickers}', fontsize=14, fontweight='bold')
    plt.show()

