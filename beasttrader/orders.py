from __future__ import annotations
from enum import Enum
import uuid
import pandas as pd


class OrderSide(Enum):  # TD AMERITRADE calls this "instruction". 
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


# 'BUY', 'SELL', 'BUY_TO_COVER', or 'SELL_SHORT'. Options instructions include 'BUY_TO_OPEN', 'BUY_TO_CLOSE', 'SELL_TO_OPEN', or 'SELL_TO_CLOSE'


"""For now I'm making it so that all orders are executed for the maximum amount"""


# all of the order classes again except with an additional field for the account number uuid
class OrderBase:
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, dollar_amount_to_use:float=None):
        self.account_number:uuid.UUID = account_number
        self.ticker = ticker
        self.side = side
        self.dollar_amount_to_use = dollar_amount_to_use        # if this is left at None, then the order will be executed for the maximum amount

        self.quantity:float|None = None
        self.open = True
        self.execution_timestamp = None
        self.execution_price:float|None = None  # this maybe needs to be a list, dict or other data structure to represent market orders that get filled at mulitple prices, depneding on how the exhanges give data 
        self.exchange:str|None = None
        self.commision:float = 0    # this is the commision paid to the exchange in dollars 

    def get_transaction_value(self) -> float | None:
        if self.execution_price is None or self.quantity is None:
            return None
        return self.execution_price * self.quantity

    def update(self, current_price_for_ticker:float):
        """This method gets overwritten by some of the subclasses but not all of them."""
        pass
    
    def set_commision_paid(self, commision:float):
        self.commision = commision

    
class MarketOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, dollars:float=None):
        super().__init__(account_number, ticker, side, dollars)


class LimitOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, limit_price:float, dollars:float=None):
        super().__init__(account_number, ticker, side, dollars)
        self.limit_price:float = limit_price


class StopOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, stop_price:float):
        super().__init__(account_number, ticker, side)
        self.stop_price:float = stop_price


class TrailingStopOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, trailing_percentage:float):
        super().__init__(account_number, ticker, side)
        self._trailing_percentage:float = trailing_percentage
        self._highest_value:float|None = None
        self.stop_price:float|None = None

    def update(self, current_price:float):
        """This method should be called every time the price changes. It returns True if the stop price has been hit, and False otherwise."""
        # update the highest value
        if self._highest_value is None:
            self._highest_value = current_price
        else:
            self._highest_value = max(self._highest_value, current_price)
        
        # update the stop value and execute the order if it has been hit
        self.stop_price = self._highest_value * (1 - self._trailing_percentage)


class StopLimitOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, stop_price:float, limit_price:float):
        super().__init__(account_number, ticker, side)
        self.stop_price:float = stop_price
        self.limit_price:float  = limit_price