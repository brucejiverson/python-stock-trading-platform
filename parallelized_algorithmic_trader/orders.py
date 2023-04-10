from enum import Enum
import uuid


class OrderSide(Enum):
    # Note: this matches the alpaca class...
    BUY = "buy"
    SELL = "sell"


# all of the order classes again except with an additional field for the account number uuid
class OrderBase(object):
    """This is the base class for all orders. It is not meant to be instantiated directly.
    
    A note on slots and inheritance: Now, regarding inheritance: for an instance to be dict-less, all classes up its inheritance 
    chain must also have dict-less instances. Classes with dict-less instances are those which define __slots__, plus most 
    built-in types (built-in types whose instances have dicts are those on whose instances you can set arbitrary attributes, 
    such as functions). Overlaps in slot names are not forbidden, but they're useless and waste some memory, since slots are inherited:"""
    
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, dollar_amount_to_use:float=None, shares:float=None):
        self.account_number:uuid.UUID = account_number
        self.ticker = ticker
        self.side = side
        self.dollar_amount_to_use = dollar_amount_to_use        # if this is left at None, then the order will be executed for the maximum amount

        self.shares:float|None = shares
        self.open:bool = True
        self.execution_timestamp = None
        self.execution_price:float|None = None  # this maybe needs to be a list, dict or other data structure to represent market orders that get filled at mulitple prices, depneding on how the exhanges give data 
        self.exchange:str|None = None
        self.commision:float = 0    # this is the commision paid to the exchange in dollars 

    def __repr__(self):
        s = f"{self.__class__.__name__.upper()}"
        if self.side is not None:
            s+= f' to {self.side.name}'
        if self.ticker is not None:
            s+= f' ticker {self.ticker}'
        if self.dollar_amount_to_use is not None:
            s+= f' for ${self.dollar_amount_to_use}'
        
        return s

    def get_transaction_value(self) -> float | None:
        """Returns the value of the transaction in dollars, product of execution price and shares."""
        if self.execution_price is None or self.shares is None:
            return None
        return self.execution_price * self.shares

    def update(self, current_price_for_ticker:float):
        """This method gets overwritten by some of the subclasses but not all of them."""
        pass
    
    def set_commision_paid(self, commision:float):
        self.commision = commision

    
class MarketOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, dollars:float=None, shares=None):
        super().__init__(account_number, ticker, side, dollars, shares)


class LimitOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, limit_price:float, dollars:float=None, shares:float=None):
        super().__init__(account_number, ticker, side, dollars, shares)
        self.limit_price:float = limit_price


class StopOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, stop_price:float, dollars:float=None, shares:float=None):
        super().__init__(account_number, ticker, side, dollars, shares)
        self.stop_price:float = stop_price


class TrailingStopOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, trailing_percentage:float):
        super().__init__(account_number, ticker, side)
        self._trailing_percentage:float = trailing_percentage
        self._highest_value:float|None = None
        self.stop_price:float|None = None
        self.stop_triggered:bool = False

    def update(self, current_price:float):
        """This method should be called every time the price changes."""
        # update the highest value
        if self._highest_value is None:
            self._highest_value = current_price
        elif current_price > self._highest_value:
            self._highest_value = current_price
            self.stop_price = self._highest_value * (1 - self._trailing_percentage)
        elif current_price < self.stop_price:
            self.stop_triggered = True
            

class StopLimitOrder(OrderBase):
    def __init__(self, account_number:uuid.UUID, ticker:str, side:OrderSide, stop_price:float, limit_price:float, shares=None, dollars=None):
        super().__init__(account_number, ticker, side)
        self.stop_price:float = stop_price
        self.limit_price:float  = limit_price
        self.shares = shares
        self.dollars = dollars
        self.stop_triggered:bool = False
        self.limit_triggered:bool = False
        
    def update(self, current_price:float):
        """Checks the stop and limit prices against the current value"""
        
        if self.side == OrderSide.BUY:
            if current_price < self.stop_price:
                self.stop_triggered = True
            
            if self.stop_triggered and current_price > self.limit_price:
                self.limit_triggered = True
        else:
            if current_price > self.stop_price:
                self.stop_triggered = True
            
            if self.stop_triggered and current_price < self.limit_price:
                self.limit_triggered = True


class CancelAllOrders(OrderBase):
    """A dummy to pass from the agent that indicates to the broker that all orders should be cancelled"""
    def __init__(self, account_number: uuid.UUID):
        super().__init__(account_number, None, None)
        
        
class CancelOrder(OrderBase):
    """This will cause the broker to look through pending orders and cancels the one with matching id(object). Holds on to order itself as well."""
    def __init__(self, account_number: uuid.UUID, order:OrderBase):
        super().__init__(account_number, order.ticker, order.side)
        self.order_to_cancel = order
        