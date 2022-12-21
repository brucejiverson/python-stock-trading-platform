import logging
import datetime
import matplotlib.pyplot as plt


if __name__ == "__main__":
    from beasttrader import polygon_io as po
    from beasttrader.exchange import TemporalResolution
    from beasttrader.performance_analysis import *
    
    root_logger = logging.getLogger('beasttrader')
    root_logger.setLevel(logging.DEBUG)
    
    # get some data
    n_days = 30*24
    now = datetime.datetime.now()
    tickers = ['SPY']
    candle_data = po.get_candles(tickers, now-datetime.timedelta(days=n_days), now, TemporalResolution.MINUTE)
    
    logger = logging.getLogger('beasttrader.performance_analysis')
    logger.info(f'Starting value of asset: {candle_data.df.iloc[0]["SPY_close"]}')
    logger.info(f'Ending value of asset: {candle_data.df.iloc[-1]["SPY_close"]}')
    # percentage change
    logger.info(f'Percentage change: {candle_data.df.iloc[-1]["SPY_close"]/candle_data.df.iloc[0]["SPY_close"]}')
    logger.info(f'')
        
    # plot the original data
    fig, ax = plt.subplots(1,1)
    for t in tickers:
        candle_data.df.reset_index().plot(y=f'{t}_close', ax=ax, label=t)
        
    ax.set_ylabel('Price (USD)')
    ax.title.set_text('Price History Plotted without Dates')
    fig.suptitle(f'Data inspection for {tickers}', fontsize=14, fontweight='bold')
    plt.show()

