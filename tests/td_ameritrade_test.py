
from parallelized_algorithmic_trader.broker import TemporalResolution
import parallelized_algorithmic_trader.td_ameritrade_wrapper.td_ameritrade_brokerage as td 

import logging
root_logger = logging.getLogger("pat")
root_logger.setLevel(logging.DEBUG)


##### NOTE: THESE HIT REAL API ENDPOINTS, BE CAREFUL WITH TESTING TOO OFTEN. WILL BE UPDATED>   
def get_candle_data():
    n_days = 10

    res = TemporalResolution.MINUTE
    td_broker = td.TDAmeritradeBroker(res, ['AAPL'])
    price_history_config = td.PriceHistoryConfig(
        symbol='AAPL',
        period=n_days
        )

    # candle_data = td_broker.get_candle_data(price_history_config)
    candle_data = td_broker.get_recent_price_history()
