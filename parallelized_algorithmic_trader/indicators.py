from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, List, Set
from dataclasses import dataclass, field
import pandas as pd
import pandas_ta as ta
from enum import Enum
from parallelized_algorithmic_trader.util import get_logger


logger = get_logger(__name__)


@dataclass
class IndicatorConfig:
    """A container for all of the information for consturcting an indicator.
    If you want the simulation to return an existing column, such as the current close price, then set the target to the column name and the construction_func to None.
    
    Parameters:
        target: str|IndicatorConfig, the df column or IndicatorConfig that the Indicator is constructed on top of
        contruction_func: function, the class that will be used to construct the Indicator
        args: Tuple[Any, ...], the arguments to pass to the construction_func
        kwargs: Dict[str, Any], the keyword arguments to pass to the construction_func
        scaling_factor: float, the factor to scale the Indicator by
        desired_output_name_keywords: List[str]|None, if the Indicator has multiple outputs, this is a list of keywords that will keep only specified outputs
        
    """
    target: str|IndicatorConfig                                         # this is the data that the Indicator is constructed on top of. Symbol or another Indicator
    construction_func: function|None                                    # the class that will be used to construct the Indicator
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    scaling_factor: float = field(default=1.0)
    bias: float = field(default=0.0)                                    # a bias to add to the Indicator
    desired_output_name_keywords:List[str]|None = field(default=None)
    names:List[str] = field(default_factory=list)
    config_name:str = field(default=None)                               # initialized in post init

    def __post_init__(self):
        self.args = tuple(self.args)
        if len(self.kwargs) == 0:
            kwargs_str = ""
        else:
            kwargs_str = f", {self.kwargs}"
        if isinstance(self.target, str):
            target_str = self.target
            constructor_name = '' if self.construction_func is None else self.construction_func.__name__
        elif isinstance(self.target, IndicatorConfig):
            target_str = self.target.config_name
            constructor_name = "_" + self.construction_func.__name__
        else:
            logger.debug(f'type of self.target: {type(self.target)} self.target: {self.target}')
            raise ValueError(f'IndicatorConfig.target must be a string or another IndicatorConfig. It is {type(self.target)}')
        
        self.config_name = f'{target_str}_{constructor_name}{self.args}{kwargs_str}'
        
        # self.names:List[str] = [f'{target_str}_{constructor_name}{self.args}{kwargs_str}']
        # if self.construction_func.__name__ == 'BB':
        #     col_identifiers = ('BBL', 'BBM', 'BBU', 'BBP', 'BBB')
        #     self.names = [f'{target_str}_{col_id}{self.args}{kwargs_str}' for col_id in col_identifiers]
        # elif self.construction_func.__name__ == 'MACD':
        #     col_identifiers = ('MACD', 'MACDh', 'MACDs')
        #     self.names = [f'{target_str}_{col_id}{self.args}{kwargs_str}' for col_id in col_identifiers]
        # elif self.construction_func.__name__ == 'SUPERTREND':
        #     col_identifiers = ('SUPERTREND', 'SUPERTRENDd', 'SUPERTRENDl', 'SUPERTRENDs')
        #     self.names = [f'{target_str}_{col_id}{self.args}{kwargs_str}' for col_id in col_identifiers]
        # elif self.construction_func.__name__ == "ZBB":
        #     col_identifiers = ('ZBBU', 'ZBBM', 'ZBBL', 'ZBBP', 'ZBBB')
        #     self.names = [f'{target_str}_{col_id}{self.args}{kwargs_str}' for col_id in col_identifiers]
            
        # logger.debug(f'Indicator names in __post_init__ before filtering: {self.names}')
        # if self.desired_output_name_keywords is not None:
        #     self.names = [n for n in self.names if any([k.lower() in n.lower() for k in self.desired_output_name_keywords])]
            
        # logger.debug(f'IndicatorConfig.__post_init__ names: {self.names}')

    def make(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame | None:
        """Construct the Indicator and return it as a DataFrame."""

        if self.construction_func is None:
            # then this config is likely being used to tell the backtest that an existing column should be included in the state on state delivery to the bot. 
            # no construction needed.
            logger.debug(f'No construction function. Returning None')
            self.config_name = self.target
            self.names = [self.target]
            return None

        target_str = self.target if isinstance(self.target, str) else self.target.config_name
        logger.debug(f'Constructing Indicator {self.construction_func.__name__} on {target_str} with args: {self.args} and kwargs: {self.kwargs}')

        # make a copy of the dataframe so that we can add features that may not be a part of the final dataframe 
        temp_df = df[self.get_root_targets()]
        
        if isinstance(self.target, str):
            logger.debug(f'Target is a string. Target: {self.target}')
            target_col_name = self.target
        elif isinstance(self.target, IndicatorConfig):
            logger.debug(f'Target is an IndicatorConfig. Constructing it first.')
            
            if not all([name in df.columns for name in self.target.names]) or 1:
                logger.debug(f'Constructing target indicator {self.target.names}')
                target_indicator = self.target.make(df)
                temp_df = pd.concat([temp_df, target_indicator], axis=1)
                
            target_col_name = self.target.names[0]
        else:
            raise ValueError(f'IndicatorConfig.target must be a string or another IndicatorConfig. It is {type(self.target)}')

        logger.debug(f'Making indicator on {target_col_name} args: {self.args}, temp_dfhead: \n{temp_df.head()}')
        ind:pd.Series|pd.DataFrame = self.construction_func(temp_df, target_col_name, *self.args, **self.kwargs)
        logger.debug(f'Indicator constructed. ind.head(): \n{ind.head()} \n ind.tail(): \n{ind.tail()}')
        
        # only include the desired columns
        if isinstance(ind, pd.DataFrame):
            if self.desired_output_name_keywords is not None:
                logger.debug(f'Using keywords: {self.desired_output_name_keywords}')
                names_to_keep = []
                for col_name in ind.columns:
                    for key_word in self.desired_output_name_keywords:
                        if key_word.lower() in col_name.lower():
                            names_to_keep.append(col_name)
                            break
                        
                logger.debug(f'Names to keep: {names_to_keep}. All names: {ind.columns.to_list()}')
                ind = ind[names_to_keep]
                
            self.names = ind.columns.to_list()
        elif isinstance(ind, pd.Series):   # there should only be one name, this is a series
            ind.name = self.config_name
            self.names = [self.config_name]
        else:
            raise TypeError(f'IndicatorConfig.make() must return a pd.Series or pd.DataFrame. It returned {type(ind)}')
        
        if self.bias != 0.0:
            ind = ind + self.bias
        if self.scaling_factor != 1.0:
            ind = ind * self.scaling_factor
        return ind

    def get_root_targets(self) -> List[str]:
        """Recursively looks through the target and possible subtargets to get the root columns of the data frame."""
        if isinstance(self.target, str):
            # if its a ticker, get OHLCV
            if not any(word in self.target for word in ('_close', '_open', '_high', '_low', '_volume')):
                return [
                    f'{self.target}_close', 
                    f'{self.target}_open', 
                    f'{self.target}_high', 
                    f'{self.target}_low', 
                    f'{self.target}_volume']             
            
            return [self.target]
        elif isinstance(self.target, IndicatorConfig):
            return self.target.get_root_targets()
        else:
            raise ValueError(f'IndicatorConfig.target must be a string or another IndicatorConfig. It is {type(self.target)}')


def SMA(df: pd.DataFrame, ticker:str, period: int) -> pd.Series:
    sma = ta.sma(df[ticker+'_close'], length=period)
    return sma


def EMA(df: pd.DataFrame, ticker:str, period: int) -> pd.Series:
    ema = ta.ema(df[ticker+'_close'], length=period)
    return ema


def DIFF(df: pd.DataFrame, target_col:str, tool_col:str) -> pd.Series:
    """The difference between the tool column and the target column"""
    logger.debug(f'DIFF: target_col: {target_col}, tool_col: {tool_col}')
    
    # check every row for type 
    diff = df[tool_col] - df[target_col]
    return diff


def VWAP(df: pd.DataFrame, ticker:str) -> pd.Series:
    suffix = ('_high', '_low', '_close', '_volume')
    vwap = ta.vwap(*[df[ticker + s] for s in suffix], anchor='D')
    return vwap


def RSI(df: pd.DataFrame, target:str, length:Any|None=None, scalar:Any|None=None) -> pd.Series:
    rsi = ta.rsi(df[target], length=length, scalar=scalar)
    return rsi


def OBV(df:pd.DataFrame, ticker:str) -> pd.Series:
    obv = ta.obv(df[ticker+'_close'], df[ticker+'_volume'])
    return obv


def BB(df: pd.DataFrame, target_col: str, period: int, stdd:int=2) -> pd.DataFrame:
    bb = ta.bbands(df[target_col], length=period, std=stdd)

    logger.debug(f'BB ta lib columns: {bb.columns}')
    # rename the columns
    ta_lib_col_names = bb.columns
    new_cols = {}
    keywords = ('BBU', 'BBB', 'BBL', 'BBM')
    for c in ta_lib_col_names:
        for kw in keywords:
            if kw in c:
                new_name = '_'.join(c.split('_')[:-2])
                new_cols[c] = f'{target_col}_{new_name}({period},)'
                break
        
    bb = bb.rename(columns=new_cols)
    return bb


def PercentBB(df:pd.DataFrame, target_col:str, period:int, stdd:int=2, vertical_bias=-0.5) -> pd.Series:
    bb = BB(df, target_col, period, stdd)
    
    price = df[target_col]
    
    for col in bb.columns:
        if 'BBU' in col:
            upper = bb[col]
        elif 'BBL' in col:
            lower = bb[col]
            
    bb_percent:pd.Series = (price - lower) / (upper - lower)
    bb_percent += vertical_bias
    bb_percent.name = f'{target_col}_BBPercent({period},)'
    return bb_percent


def MACD(df: pd.DataFrame, target_col:str, fast_period: int, slow_period: int, signal_period: int) -> pd.DataFrame:
    macd:pd.DataFrame = ta.macd(df[target_col], fast_period, slow_period, signal_period)
    
    # rename the columns
    new_names = []
    for c in macd.columns:
        if 'MACD_' in c:
            new_names.append(f'{target_col}_MACD{(fast_period, slow_period, signal_period)}')
        elif 'MACDh_' in c:
            new_names.append(f'{target_col}_MACDh{(fast_period, slow_period, signal_period)}')
        elif 'MACDs_' in c:
            new_names.append(f'{target_col}_MACDs{(fast_period, slow_period, signal_period)}')

    macd.rename(columns=dict(zip(macd.columns, new_names)), inplace=True)
    return macd


def DERIVATIVE(df: pd.DataFrame, target_col:str, shift_val:int=1) -> pd.Series:
    """If shift_val > 0, converts the column to the percentage change.
    if < 0, adds a new column with the future change"""
    if shift_val > 0:
        deriv = df[target_col].pct_change(shift_val)
        deriv = 100*deriv
        return deriv
    
    # if shift_val < 0:
    #     names = '% Change ' + str(-shift_val) + ' steps in future'
    #     df[names] = (df[target_col].shift(shift_val, fill_value=None) - df[target_col])/df[target_col]
    #     df[names] = 100*df[names]
    #     df.replace([np.inf, -np.inf], np.nan)
    #     df.dropna(inplace=True)
    #     return df, names
    else: 
        raise NotImplementedError('shift_val < 0 is not implemented as it gives knowledge of the future.')
    

def ZEMA(df: pd.DataFrame, target_col:str, period:int) -> pd.Series:
    zema = ta.zlma(df[target_col], length=period)
    return zema


def ZBB(df: pd.DataFrame, target_col:str, period:int, stddev:float) -> pd.Series:
    zema = ta.zlma(df[target_col], length=period)

    # where in a typical bollinger calculation the stddev would be taken straight up on the price data, here we take 
    # on the difference between the price and the zema
    price_zema_diff = df[target_col] - zema
    bbb = stddev*ta.stdev(price_zema_diff, length=period)

    bbu = zema + bbb/2
    bbl = zema - bbb/2
    bbp = (df[target_col] - bbl)/(bbu - bbl)
    
    # merge all the series into a singule dataframe
    data = {
        f'{target_col}_ZBBU({period},{stddev})': bbu,
        f'{target_col}_ZBBM({period},{stddev})': zema,
        f'{target_col}_ZBBL({period},{stddev})': bbl,
        f'{target_col}_ZBBP({period},{stddev})': bbp,
        f'{target_col}_ZBBB({period},{stddev})': bbb,
    }    

    zbb = pd.DataFrame(data, index=df.index)
    return zbb


def SUPERTREND(df: pd.DataFrame, ticker:str, period:int, multiplier:float) -> pd.DataFrame:
    st = ta.supertrend(df[ticker+'_high'], df[ticker+'_low'], df[ticker+'_close'], period, multiplier)
        
    # rename the columns  
    new_names = []
    keywords = ('SUPERT_', 'SUPERTd', 'SUPERTl', 'SUPERTs')
    formated_keywords = ('SUPERTREND', 'SUPERTRENDd', 'SUPERTRENDl', 'SUPERTRENDs')
    for c in st.columns:
        for k, fk in zip(keywords, formated_keywords):
            new_names.append(f'{ticker}_{fk}{(period, multiplier)}') if k in c else None

    st.rename(columns=dict(zip(st.columns, new_names)), inplace=True)
    for col in st.columns:
        st[col] = (st[col] + 1)/2 if 'SUPERTRENDd' in col else st[col]
    return st


def PRICE_HISTORY(df: pd.DataFrame, ticker:str, *mask) -> pd.DataFrame:
    """mask is a list of bools ordered from most recent backwards in time indicating 
    whether to include that timestep or not"""

    price_history = pd.DataFrame()
    
    for i, b in enumerate(mask):
        if not isinstance(b, bool):
            raise TypeError('mask must be a list of bools')
        if b and not f'price_history_{i}' in df.columns:
            price_history[f'price_history_{i}'] = df[ticker+'_close'].shift(i)
    return price_history
    

def TIMEOFDAY(df:pd.DataFrame, _) -> pd.Series:
    """The hour of the day"""
    logger.debug(f'Adding feature for the hour of the day called TIMEOFDAYm    ')
    
    # hr = pd.Series([i.hour for i in df.index])
    hr = df[_]
    for i in range(len(hr)):
        hr.iloc[i] = df.index[i].hour + df.index[i].minute/60
    
    return hr


def MEAN(df:pd.DataFrame, targets=List[str]) -> pd.DataFrame:
    """Takes the mean of the specified features"""
    means = df[targets].mean(axis=1)
    return means


def MIN_MAX_SCALER(df:pd.DataFrame, target:str, min_val:float, max_val:float) -> pd.Series:
    """Scales the target column to be between min_val and max_val"""
    return (df[target] - min_val)/(max_val - min_val)


def STANDARD_SCALER(df:pd.DataFrame, target:str) -> pd.Series:
    """Scales the target column to have a mean of 0 and a standard deviation of 1"""
    return (df[target] - df[target].mean())/df[target].std()


IndicatorMapping = List[IndicatorConfig] 


# class IndicatorMapping(list):
#     """A list of IndicatorConfigs"""
#     def get_all_names(self) -> List[str]:
#         return [n for config in self for n in config.names]
    
#     def get_n_features(self) -> int:
#         return len(self.get_all_names())

#     def get_all_tickers_from_mapping(self) -> Set[str]:
#         tickers = set()
#         for m in self:
#             # search down through the targets until we find a ticker
#             target = m.target
#             while not isinstance(target, str):
#                 target = target.target
#             tickers.add(target.split('_')[0])
#         print(f'Found following tickers feature mapping associated with training: {tickers}')
#         return tickers

"""Defines the structure of the state. Each element of the tuple is a DataStreamAddress. When state gets built it will be a 1D vector of the data from each address. The order of the vector is the same as the order of the tuple"""


def check_if_indicators_identical(config1:IndicatorConfig, config2:IndicatorConfig) -> bool:
    """Checks the target recursively, the construction func, args and kwargs to see if they are identical
    
    """
    # check the target
    if isinstance(config1.target, str) and isinstance(config2.target, str):
        if config1.target != config2.target: return False
    elif type(config1.target) != type(config2.target):
        return False
    else:   # both are IndicatorConfigs
        assert isinstance(config1.target, IndicatorConfig) and isinstance(config2.target, IndicatorConfig)
        check_if_indicators_identical(config1.target, config2.target)
    
    # check the other parameters
    if any([
        config1.construction_func != config2.construction_func,
        config1.args != config2.args,
        config1.kwargs != config2.kwargs,]):
        return False

    return True


def check_if_indicator_mappings_identical(m1:IndicatorMapping, m2:IndicatorMapping) -> bool:
    """Checks if two IndicatorMappings are identical"""
    if len(m1) != len(m2): return False

    for c1, c2 in zip(m1, m2):
        if not check_if_indicators_identical(c1, c2): 
            return False
        
    return True


def aggregate_indicator_mappings(mappings: List[IndicatorMapping]) -> IndicatorMapping:
    """Aggregate multiple IndicatorMappings into a single IndicatorMapping with no duplicates"""
    aggregated:IndicatorMapping = []
    names_of_indicators_weve_found:List[str] = []

    def check_if_any_name_has_been_found(names:List[str]) -> bool:
        for name in names:
            if name in names_of_indicators_weve_found:
                return True
        return False
    
    for m in mappings:
        for indicator_config in m:
            # logger.debug(f'indicator_config.names: {indicator_config.names}')
            # logger.debug(f'names_of_indicators_weve_found: {names_of_indicators_weve_found}')
            
            # if not check_if_any_name_has_been_found(indicator_config.names):
            #     [names_of_indicators_weve_found.append(n) for n in indicator_config.names]
            if not any([check_if_indicators_identical(indicator_config, a) for a in aggregated]):
                aggregated.append(indicator_config)
        
    return aggregated

