from dataclasses import dataclass
from typing import List
import datetime
import os
import pickle

from beasttrader.market_data import TemporalResolution

from beasttrader.strategy import StrategyBase
from beasttrader.exchange import Account
from beasttrader.util import maybe_make_dir, RESULTS_DIRECTORY


@dataclass
class SimulationResult:
    strategy:StrategyBase
    account:Account
    start:datetime.datetime
    end:datetime.datetime
    tickers:List[str]
    resolution:TemporalResolution

    @property
    def final_value(self):
        return list(self.account.value_history.values())[-1]


SimulationResultSet = List[SimulationResult]


def save_backtest_results(result:SimulationResult):
    
    maybe_make_dir(RESULTS_DIRECTORY)
    file_name = os.path.join(RESULTS_DIRECTORY, 'latest_result.pkl')
    with open(file_name, 'wb') as f:
        pickle.dump(result, f)


def load_backtest_results() -> SimulationResult:
    file_name = os.path.join(RESULTS_DIRECTORY, 'latest_result.pkl')
    with open(file_name, 'rb') as f:
        return pickle.load(f)


