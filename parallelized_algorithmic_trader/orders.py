
from enum import Enum
import uuid


class OrderSide(Enum):  # TD AMERITRADE calls this "instruction". 
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


# 'BUY', 'SELL', 'BUY_TO_COVER', or 'SELL_SHORT'. Options instructions include 'BUY_TO_OPEN', 'BUY_TO_CLOSE', 'SELL_TO_OPEN', or 'SELL_TO_CLOSE'
"""For now I'm making it so that all orders are executed for the maximum amount"""


# all of the order classes again except with an additional field for the account number uuid
class OrderBase(object):
    """This is the base class for all orders. It is not meant to be instantiated directly.
    
    A note on slots and inheritance: Now, regarding inheritance: for an instance to be dict-less, all classes up its inheritance 
    chain must also have dict-less instances. Classes with dict-less instances are those which define __slots__, plus most 
    built-in types (built-in types whose instances have dicts are those on whose instances you can set arbitrary attributes, 
    such as functions). Overlaps in slot names are not forbidden, but they're useless and waste some memory, since slots are inherited:"""
    __slots__ = "account_number", "ticker", "side", "dollar_amount_to_use", "quantity", "open", "execution_timestamp", \
    "execution_price", "exchange", "commision"
    
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
        """Returns the value of the transaction in dollars, product of execution price and quantity."""
        if self.execution_price is None or self.quantity is None:
            return None
        return self.execution_price * self.quantity

    def update(self, current_price_for_ticker:float):
        """This method gets overwritten by some of the subclasses but not all of them."""
        pass
    
    def set_commision_paid(self, commision:float):
        self.commision = commision

    
class MarketOrder(OrderBase):
    __slots__ = ()
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, dollars:float=None):
        super().__init__(account_number, ticker, side, dollars)


class LimitOrder(OrderBase):
    __slots__ = "limit_price"
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, limit_price:float, dollars:float=None):
        super().__init__(account_number, ticker, side, dollars)
        self.limit_price:float = limit_price


class StopOrder(OrderBase):
    __slots__ = "stop_price"
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, stop_price:float):
        super().__init__(account_number, ticker, side)
        self.stop_price:float = stop_price


class TrailingStopOrder(OrderBase):
    __slots__ = "_trailing_percentage", "_highest_value", "stop_price"
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
    __slots__ = "stop_price", "limit_price"
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, stop_price:float, limit_price:float):
        super().__init__(account_number, ticker, side)
        self.stop_price:float = stop_price
        self.limit_price:float  = limit_price