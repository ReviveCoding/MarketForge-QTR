from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OwnOrder:
    side: str
    price: float
    size: int
    arrival_ns: int
    queue_ahead: int = 0
    filled: int = 0
    active: bool = False
    cancelled: bool = False


class QueueSimulator:
    def __init__(self) -> None:
        self.depth: dict[tuple[str, float], int] = {}
        self.orders: list[OwnOrder] = []
        self.cash = 0.0
        self.position = 0
        self.fill_times: list[int] = []

    def submit(self, side: str, price: float, size: int, send_ns: int, latency_ns: int) -> OwnOrder:
        order = OwnOrder(side, price, size, send_ns + latency_ns)
        self.orders.append(order)
        return order

    def _activate(self, now_ns: int) -> None:
        for order in self.orders:
            if not order.active and not order.cancelled and order.arrival_ns <= now_ns:
                order.active = True
                order.queue_ahead = self.depth.get((order.side, order.price), 0)

    def market_add(self, side: str, price: float, size: int, now_ns: int) -> None:
        self._activate(now_ns)
        key = (side, price)
        self.depth[key] = self.depth.get(key, 0) + size

    def market_cancel(
        self, side: str, price: float, size: int, now_ns: int, ahead_of_us: bool = True
    ) -> None:
        self._activate(now_ns)
        key = (side, price)
        self.depth[key] = max(0, self.depth.get(key, 0) - size)
        if ahead_of_us:
            for order in self.orders:
                if order.active and not order.cancelled and (order.side, order.price) == key:
                    order.queue_ahead = max(0, order.queue_ahead - size)

    def market_trade(self, resting_side: str, price: float, size: int, now_ns: int) -> None:
        self._activate(now_ns)
        remaining = size
        key = (resting_side, price)
        market_depth = self.depth.get(key, 0)
        consumed_market = min(remaining, market_depth)
        self.depth[key] = market_depth - consumed_market
        remaining -= consumed_market
        for order in self.orders:
            if not remaining:
                break
            if order.active and not order.cancelled and (order.side, order.price) == key:
                consume_ahead = min(remaining, order.queue_ahead)
                order.queue_ahead -= consume_ahead
                remaining -= consume_ahead
                fill = min(remaining, order.size - order.filled)
                if fill:
                    order.filled += fill
                    remaining -= fill
                    self.position += fill if order.side == "B" else -fill
                    self.cash += -price * fill if order.side == "B" else price * fill
                    self.fill_times.extend([now_ns] * fill)

    def nav(self, mid: float) -> float:
        return self.cash + self.position * mid
