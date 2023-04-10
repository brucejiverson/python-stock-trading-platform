
from time import monotonic
from typing import List, Dict
import logging
import pandas as pd
import logging
import traceback
import time

from parallelized_algorithmic_trader.data_management.data import CandleData
from parallelized_algorithmic_trader.data_management.data_utils import TemporalResolution
from parallelized_algorithmic_trader.indicators import IndicatorConfig, IndicatorMapping, aggregate_indicator_mappings
from parallelized_algorithmic_trader.strategy import StrategySet, StrategyConfig
from parallelized_algorithmic_trader.trading.alpaca_broker import AlpacaBroker
from parallelized_algorithmic_trader.data_management.data import get_candle_data, CandleData

logger = logging.getLogger('parallelized_algorithmic_trader.neat')
root_logger = logging.getLogger('parallelized_algorithmic_trader')
root_logger.setLevel(logging.DEBUG)


def get_state(indicators:IndicatorMapping, row:pd.Series) -> Dict[str, float]:
    feature_names = []
    for indicator_conf in indicators:
        feature_names.extend(indicator_conf.names)
    return row[feature_names].to_dict()


def build_features(df:pd.DataFrame, indicator_mapping:IndicatorMapping) -> pd.DataFrame:

    logger.debug(f'Building features for {len(df)} rows of data')

    for indicator_conf in indicator_mapping:
        if len(indicator_conf.names) == 0:
            raise ValueError(f'Indicator config {indicator_conf} has no names')
        
        if all([name in df.columns for name in indicator_conf.names]): 
            logger.warning(f'Skipping indicator {indicator_conf.names} because it is already in the dataframe')
            continue  # skip if already in df
        logger.debug(f'Making indicator {indicator_conf.names}')
        indicator = indicator_conf.make(df)
        df = pd.concat([df, indicator], axis=1)
        logger.info(f'Built indicator {indicator_conf.names}')
        
    logger.debug(f'Columns list after indicators build: {df.columns}')

    # that has had the indicators warm up and is ready to be used for the backtest
    logger.debug(f'Backtest dataframe has {len(df)} rows of data. Starting at {df.index[0]} and ending at {df.index[-1]}')
    
    return df


def run(
    strategy_config:StrategyConfig, 
    tickers:List[str], 
    res:TemporalResolution, 
    sub_period=1,
    log_level=None):
    
    logger = logging.getLogger(__name__ )
    if log_level is not None: logger.setLevel(log_level)

    broker = AlpacaBroker()
    
    broker.set_expected_resolution(res)
    for t in tickers:
        # inform the broker that there is a new equity and to expect incoming candle_data about it 
        broker.add_ticker(t)

    agent = strategy_config.instantiate(broker.get_account_number())

    try:
        # Loop through the price history facilitating interaction between the strategies and the broker
        # backtest with itterrows took: 20.058 s, using while loop took 19.1 s holding everything else constant
        while 1:
            loop_start = monotonic()
            
            # the full dataframe with all candle data that we will build features on and then loop over
            df = broker.get_current_candle_data()
            df = build_features(df, strategy_config.indicator_mapping)
            
            # update the state by pulling all info on pending orders, holdings, cash, etc
            broker.update_state()
            
            # check the current position, see if pending orders got executed
            actions_executed_in_last_candle:List = broker.update_current_position()

            # inform the strategy if they had an order get filled or if a pending order got cancelled
            if len(actions_executed_in_last_candle) > 0:
                if 'ORDER' in type(actions_executed_in_last_candle[0]).__name__.upper():
                    strategy.order_filled(actions_executed_in_last_candle[0])
                elif 'CANCEL' in type(actions_executed_in_last_candle[0]).__name__.upper():
                    strategy.order_cancelled()
            
            # check to make new orders
            state_for_agent = get_state(strategy_config.indicator_mapping, df.iloc[-1])
            new_orders = agent.act(state_for_agent)
            
            if new_orders is not None:
                agent.set_pending_order(new_orders)
                broker.submit_order(new_orders)

            # sleep until the next evaluation time
            elapsed = monotonic() - loop_start
            st = res.get_as_minutes()*60 - elapsed
            logger.debug(f'Loop took {elapsed} seconds, sleeping for {st:.2f} seconds')
            time.sleep(st)

    except KeyboardInterrupt:
        logger.warning('Keyboard interrupt detected, exiting')
    except:
        logger.exception('Exception detected, exiting')
        logger.error(traceback.format_exc())


if __name__ == "__main__":

    tickers = ['SPY']
    # I've confirmed that these get served to the bot in order. Types of features should get grouped together
    indicator_mapping:IndicatorMapping= []
    for t in tickers:
        new_inds = [
            IndicatorConfig(t+'_close', indicators.BB, args=(30,), desired_output_name_keywords=['BBB']),
            IndicatorConfig(t+'_close', indicators.PercentBB, args=(30,)),
        ]
        
        indicator_mapping.extend(new_inds)
        break   # only do it for the first ticker for now


    bot_config = StrategyConfig(
        indicator_mapping=indicator_mapping,
        strategy=NEATEqualAllocation,
        tickers=tickers,
        args=[],
        kwargs={'net':net, 'log_level':logging.WARNING}
        )
    
    run(
        bot_config,
        tickers,
        TemporalResolution.HOUR,
        sub_period=4
        )