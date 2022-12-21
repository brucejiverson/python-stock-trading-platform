from __future__ import annotations
from time import monotonic
from typing import List, Dict
import logging
import pandas as pd
import logging
import traceback
import time

from beasttrader.market_data import CandleData, TemporalResolution
from beasttrader.my_td_ameritrade import td_ameritrade_exchange as td
from beasttrader.indicators import IndicatorConfig, IndicatorMapping, aggregate_indicator_mappings
from beasttrader.strategy import StrategySet, StrategyConfig
from beasttrader.backtest_tools import *


logger = logging.getLogger('beasttrader.neat')
root_logger = logging.getLogger('beasttrader')
root_logger.setLevel(logging.DEBUG)


def run(strategy_config:StrategyConfig, tickers, res:TemporalResolution, log_level=None):
    logger = logging.getLogger(__name__ )
    if log_level is not None: logger.setLevel(log_level)

    ### td ameritrade testing
    exchange = td.TDAmeritradeBroker(TemporalResolution.MINUTE, ['SPY', 'SPDN'])

    exchange.set_expected_resolution(res)
    for t in tickers:
        # inform the exchange that there is a new equity and to expect incoming candle_data about it 
        exchange.add_ticker(t)

    # instantiate the strategy class
    strategy = strategy_config.instantiate(exchange._account_number)

    def get_state(indicators:IndicatorMapping, row:pd.Series) -> Dict[str, float]:
        feature_names = []
        for indicator_conf in indicators:
            feature_names.extend(indicator_conf.names)
        return row[feature_names].to_dict()

    try:
        # Loop through the price history facilitating interaction between the strategies and the exchange
        # backtest with itterrows took: 20.058 s, using while loop took 19.1 s holding everything else constant
        while 1:
            loop_start = monotonic()
            # the full dataframe with all candle data that we will build features on and then loop over
            ph_conf = td.PriceHistoryConfig(
                tickers=tickers,
            )
            df = exchange.get_candle_data(ph_conf)
            logger.debug(f'Building features for {len(df)} rows of data')
            # aggregate the indicator configs from all the strategies and add them to the df
            for indicator_conf in strategy_config.indicator_mapping:
                s = indicator_conf.make(df)
                df = pd.concat([df, s], axis=1)
            logger.info(f'Built following indicators: {[conf.names for conf in strategy_config.indicator_mapping]}')

            """This is broken out as a major efficiency upgrade for genetic algorithms where you need the same state hundreds of times for your population. Found that when state was 
            gotten for each strategy, backtesting for 1 day slowed from ~0.75 seconds to 20 seconds."""
            # check the current position, see if pending orders got executed
            actions_executed_in_last_candle:List = exchange.update_current_position()

            # inform the strategy if they had an order get filled or if a pending order got cancelled
            if len(actions_executed_in_last_candle) > 0:
                if 'ORDER' in type(actions_executed_in_last_candle[0]).__name__.upper():
                    strategy.order_filled(actions_executed_in_last_candle[0])
                elif 'CANCEL' in type(actions_executed_in_last_candle[0]).__name__.upper():
                    strategy.order_cancelled()
            
            # check to make new orders
            state = get_state(strategy.indicator_mapping, exchange._data.df.iloc[-1])
            new_order = strategy.act(state)
            if new_order is not None:
                strategy.set_pending_order(new_order)
                exchange.submit_order(new_order)

            # sleep until the next evaluation time
            elapsed = monotonic() - loop_start
            st = res.get_as_minutes()*60 - elapsed
            logger.debug(f'Loop took {elapsed} seconds, sleeping for {st:.2f} seconds')
            time.sleep(st)

    except KeyboardInterrupt:
        logger.info('Keyboard interrupt detected, exiting')
    except:
        logger.exception('Exception detected, exiting')
        logger.error(traceback.format_exc())
            

if __name__ == "__main__":
    pass
    
    # run()