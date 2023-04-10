
import matplotlib.pyplot as plt
import pandas as pd

from parallelized_algorithmic_trader.performance_analysis import print_account_stats, get_vwr, get_best_strategy_and_account
from parallelized_algorithmic_trader.visualizations import plot_trade_profit_hist, plot_backtest_results, plot_cumulative_returns, plot_underwater
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping



if __name__ == '__main__':
    
    # open the exchange and get the order history

    if verbose or plot:
        # find the best performing agent
        best_strategy = results[0].strategy
        account_history = results[0].account

    print_account_stats(account_history, market_data)
    vwr = get_vwr(account_history, market_data.resolution)
    print('VWR: {:.2f}'.format(vwr))
    if 1:
        # create the nested indicators the strategy to provide insights in plotting
        for indicator_conf in best_strategy.indicator_mapping:
            if isinstance(indicator_conf.target, IndicatorConfig and indicator_conf.target.name not in df.columns:

                s = indicator_conf.target.make(df)
                df = pd.concat([df, s], axis=1)
    plot_backtest_results(df, account_history, market_data.tickers)
    plot_underwater(account_history)
    plot_cumulative_returns(account_history, market_data)
    
    plot_trade_profit_hist(account_history)
    plt.show()
