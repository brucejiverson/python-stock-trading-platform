from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pandas_ta as ta
from enum import Enum
from parallelized_algorithmic_trader.util import get_logger


logger = get_logger(__name__)


class IndicatorConstructionType(Enum):
    """The type of Indicator construction to use"""
    PANDAS = "PANDAS"         # use a column from the data frame, for backtesting
    ON_EACH_DATA_POINT = "on_each_data_point"       # for live trading?


class NumericalIndicatorSpaces(Enum):
    """The spaces that numerical indicators can be in"""
    PRICE_ACTION = "The same space as the price data"
    BOUNDED = "A bounded space, such as a percentage or a ratio"
    INDEPENDANT = "An independant space, such fully scaled different data ie renko, OBV"


@dataclass
class IndicatorConfig:
    """By default I'm making these all use TA-Lib."""
    target: str | IndicatorConfig                      # this is the data that the Indicator is constructed on top of. Symbol or another Indicator
    construction_func: function                      # the class that will be used to construct the Indicator
    construction_type: IndicatorConstructionType = field(default=IndicatorConstructionType.PANDAS)         
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    scaling_factor: float = field(default=1.0)
    desired_output_name_keywords:List[str]|None = field(default=None)

    def __post_init__(self):
        self.args = tuple(self.args)
        if len(self.kwargs) == 0:
            k = ""
        else:
            k = f", {self.kwargs}"
        if type(self.target) == str:
            t = self.target
            constructor_name = self.construction_func.__name__
        elif type(self.target) == IndicatorConfig:
            t = self.target.names[0]
            constructor_name = "_" + self.construction_func.__name__
        else:
            logging.debug(f'type of self.target: {type(self.target)} self.target: {self.target}')
            raise ValueError(f'IndicatorConfig.target must be a string or another IndicatorConfig. It is {type(self.target)}')
        self.names:List[str] = [f'{t}_{constructor_name}{self.args}{k}']
        if self.construction_func.__name__ == 'BB':
            col_kwards = ('BBL', 'BBM', 'BBU', 'BBP', 'BBB')
            self.names = [f'{t}_{i}{self.args}{k}' for i in col_kwards]
            if self.desired_output_name_keywords is not None:
                self.names = [n for n in self.names if any([k.lower() in n.lower() for k in self.desired_output_name_keywords])]
        elif self.construction_func.__name__ == 'MACD':
            col_kwards = ('MACD', 'MACDh', 'MACDs')
            self.names = [f'{t}_{i}{self.args}{k}' for i in col_kwards]
            if self.desired_output_name_keywords is not None:
                self.names = [n for n in self.names if any([k.lower() in n.lower() for k in self.desired_output_name_keywords])]
        elif self.construction_func.__name__ == 'SUPERTREND':
            col_kwards = ('SUPERTREND', 'SUPERTRENDd', 'SUPERTRENDl', 'SUPERTRENDs')
            self.names = [f'{t}_{i}{self.args}{k}' for i in col_kwards]
            if self.desired_output_name_keywords is not None:
                self.names = [n for n in self.names if any([k.lower() in n.lower() for k in self.desired_output_name_keywords])]
        
        # parse to see what math space the construction func is in
        price_space_func_names = (
            'SMA', 'EMA', 'DIFF', 'VWAP', 'BB'
        )
        bounded_space_func_names = (
            'RSI', 'MACD', 'STOCH',
        )
        independant_space_func_names = (
            'OBV', 'RENKO'
        )

        if self.construction_func.__name__ in price_space_func_names:
            self.math_space = NumericalIndicatorSpaces.PRICE_ACTION
        elif self.construction_func.__name__ in bounded_space_func_names:
            self.math_space = NumericalIndicatorSpaces.BOUNDED
        elif self.construction_func.__name__ in independant_space_func_names:
            self.math_space = NumericalIndicatorSpaces.INDEPENDANT

    def make(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Construct the Indicator and return it as a DataFrame."""

        logger.debug(f'Constructing Indicator {self.names[0]}')

        # make a copy of the dataframe so that we can add features that may not be a part of the final dataframe 
        if self.construction_type == IndicatorConstructionType.PANDAS:
            temp_df = df.copy()
            if type(self.target) == str:
                target_col_name = self.target
            elif type(self.target) == IndicatorConfig:
                for name in [n for n in self.target.names if not n in df.columns]:
                    logger.debug(f'Could not find {name} in the dataframe. Trying to construct it so that subsequently the parent can be constructed.')
                    target_indicator = self.target.make(df)
                    # temp_df[self.target.names] = target_indicator
                    temp_df = pd.concat([temp_df, target_indicator], axis=1)
                target_col_name = self.target.names[0]
            else: raise ValueError(f'IndicatorConfig.target must be a string or another IndicatorConfig. It is {type(self.target)}')

            ind:pd.Series|pd.DataFrame = self.construction_func(temp_df, target_col_name, *self.args, **self.kwargs)
           
            # only include the desired columns
            if type(ind) == pd.DataFrame and self.desired_output_name_keywords is not None:
                names_to_use = [c for c in ind.columns if any([k.lower() in c.lower() for k in self.desired_output_name_keywords])]
                ind = ind[names_to_use]
                self.names = ind.columns
            else:   # there should only be one name, this is a series
                ind.name = self.names[0]
            
            # do scaling
            if self.scaling_factor != 1.0:
                if type(ind) == pd.DataFrame:
                    for col in ind.columns:
                        ind[col] = ind[col] * self.scaling_factor
                else: ind = ind * self.scaling_factor
            return ind
        else:
            raise NotImplementedError(f'{self.construction_type} is not implemented')


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


def RSI(df: pd.DataFrame, ticker:str, length:Any|None=None, scalar:Any|None=None) -> pd.Series:
    rsi = ta.rsi(df[ticker+'_close'], length=length, scalar=scalar)
    return rsi


def OBV(df:pd.DataFrame, ticker:str) -> pd.Series:
    obv = ta.obv(df[ticker+'_close'], df[ticker+'_volume'])
    return obv


def BB(df: pd.DataFrame, target_col: str, period: int, stdd:int=2) -> pd.DataFrame:
    bb = ta.bbands(df[target_col], length=period, std=stdd)

    # rename the columns
    cols = bb.columns
    new_cols = {}
    for c in cols:
        if 'BBU' in c:
            new_cols[c] = f'{target_col}_BBU({period},)'
        elif 'BBB' in c:
            new_cols[c] = f'{target_col}_BBB({period},)'
        elif 'BBL' in c:
            new_cols[c] = f'{target_col}_BBL({period},)'
        elif 'BBM' in c:
            new_cols[c] = f'{target_col}_BBM({period},)'
        elif 'BBP' in c:
            new_cols[c] = f'{target_col}_BBP({period},)'
        
    bb = bb.rename(columns=new_cols)
    return bb


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
        deriv = df[target_col]/df[target_col].shift(shift_val, fill_value=0) - 1
        deriv = 100*df[target_col].fillna(0)
        # df.replace([np.inf, -np.inf], np.nan)
        # df.dropna(inplace=True)
        return deriv
    # if shift_val < 0:
    #     names = '% Change ' + str(-shift_val) + ' steps in future'
    #     df[names] = (df[target_col].shift(shift_val, fill_value=None) - df[target_col])/df[target_col]
    #     df[names] = 100*df[names]
    #     df.replace([np.inf, -np.inf], np.nan)
    #     df.dropna(inplace=True)
    #     return df, names
    else: raise NotImplementedError('shift_val < 0 is not implemented')
    

def ZEMA(df: pd.DataFrame, target_col:str, period:int) -> pd.Series:
    zema = ta.zlma(df[target_col], length=period)
    return zema


def ZBB(df: pd.DataFrame, target_col:str, period:int, stddev:float) -> pd.Series:
    zema = ta.zlema(df[target_col], length=period)

    # where in a typical bollinger calculation the stddev would be taken straight up on the price data, here we take 
    # on the difference between the price and the zema
    price_zema_diff = df[target_col] - zema
    bbb = stddev*ta.stdev(price_zema_diff, length=period)

    bbu = zema + bbb/2
    bbl = zema - bbb/2
    bbp = (df[target_col] - bbl)/(bbu - bbl)
    
    # merge all the series into a singule dataframe
    cols = [
        f'{target_col}_ZBBU({period},{stddev})', 
        f'{target_col}_ZBBL({period},{stddev})', 
        f'{target_col}_ZBBP({period},{stddev})', 
        f'{target_col}_ZBBB({period},{stddev})']

    zbb = pd.concat([bbu, bbl, bbp, bbb], axis=1, columns=cols)
    
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


IndicatorMapping = Tuple[IndicatorConfig] 
"""Defines the structure of the state. Each element of the tuple is a DataStreamAddress. When state gets built it will be a 1D vector of the data from each address. The order of the vector is the same as the order of the tuple"""


def aggregate_indicator_mappings(mappings: List[IndicatorMapping]) -> IndicatorMapping:
    """Aggregate multiple IndicatorMappings into a single IndicatorMapping with no duplicates"""

    aggregated:IndicatorMapping = []
    appended_names = []
    for m in mappings:
        for config in m:
            if config.names not in appended_names:
                aggregated.append(config)
                appended_names.append(config.names)

    return aggregated
