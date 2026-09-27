from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

BookKey = tuple[str, int, str, str, str]


@dataclass(frozen=True, slots=True)
class MBOEvent:
    source: str
    instrument_id: int
    contract: str
    trading_date: str
    session_id: str
    ts_event_ns: int
    sequence: int
    action: str
    side: str
    order_id: int
    price: float | None
    size: int
    flags: int = 0

    @property
    def key(self) -> BookKey:
        return (self.source, self.instrument_id, self.contract, self.trading_date, self.session_id)


@dataclass(slots=True)
class RestingOrder:
    side: str
    price: float
    size: int
    entered_ns: int


class FIFOBook:
    def __init__(self) -> None:
        self.orders: dict[int, RestingOrder] = {}
        self.levels: dict[str, dict[float, OrderedDict[int, None]]] = {"B": {}, "A": {}}
        self.reset_generation = 0

    def clear(self) -> None:
        self.orders.clear()
        self.levels = {"B": {}, "A": {}}
        self.reset_generation += 1

    def _remove(self, order_id: int, quantity: int | None = None) -> int:
        order = self.orders.get(order_id)
        if order is None:
            return 0
        removed = order.size if quantity is None else min(order.size, quantity)
        order.size -= removed
        if order.size == 0:
            level = self.levels[order.side][order.price]
            level.pop(order_id, None)
            if not level:
                self.levels[order.side].pop(order.price)
            self.orders.pop(order_id)
        return removed

    def apply(self, event: MBOEvent) -> None:
        if event.action in {"T", "F", "N"}:
            return
        if event.action == "R":
            self.clear()
            return
        if event.side not in {"B", "A"}:
            raise ValueError("book-changing event requires B/A side")
        if event.action == "A":
            if event.price is None:
                raise ValueError("add requires price")
            if event.order_id in self.orders:
                raise ValueError("duplicate order id")
            order = RestingOrder(event.side, event.price, event.size, event.ts_event_ns)
            self.orders[event.order_id] = order
            self.levels[event.side].setdefault(event.price, OrderedDict())[event.order_id] = None
        elif event.action == "C":
            self._remove(event.order_id, event.size)
        elif event.action == "M":
            self._remove(event.order_id)
            if event.price is None:
                raise ValueError("modify requires price")
            order = RestingOrder(event.side, event.price, event.size, event.ts_event_ns)
            self.orders[event.order_id] = order
            self.levels[event.side].setdefault(event.price, OrderedDict())[event.order_id] = None
        else:
            raise ValueError(f"unsupported action {event.action}")

    def queue_at(self, side: str, price: float) -> list[tuple[int, int]]:
        return [
            (order_id, self.orders[order_id].size) for order_id in self.levels[side].get(price, {})
        ]

    def top(self, side: str, depth: int = 10) -> list[tuple[float, int, int]]:
        prices = sorted(self.levels[side], reverse=side == "B")[:depth]
        return [
            (
                price,
                sum(self.orders[order_id].size for order_id in self.levels[side][price]),
                len(self.levels[side][price]),
            )
            for price in prices
        ]


class MultiInstrumentBooks:
    def __init__(self) -> None:
        self.books: dict[BookKey, FIFOBook] = {}

    def apply(self, event: MBOEvent) -> FIFOBook:
        book = self.books.setdefault(event.key, FIFOBook())
        book.apply(event)
        return book

    def get(self, key: BookKey) -> FIFOBook:
        return self.books[key]


@dataclass(slots=True)
class SimulatedPassiveOrder:
    key: BookKey
    side: str
    price: float
    quantity: int
    decision_time_ns: int
    activation_time_ns: int
    horizon_end_ns: int
    queue_ahead: OrderedDict[int, int] = field(default_factory=OrderedDict)
    filled_quantity: int = 0
    first_fill_time_ns: int | None = None
    completion_time_ns: int | None = None
    queue_trajectory: list[tuple[int, int]] = field(default_factory=list)
    active: bool = False
    invalidated_by_reset: bool = False

    @property
    def fill_fraction(self) -> float:
        return self.filled_quantity / self.quantity

    @property
    def time_to_first_fill_ns(self) -> int | None:
        return (
            None
            if self.first_fill_time_ns is None
            else self.first_fill_time_ns - self.activation_time_ns
        )

    def activate(self, book: FIFOBook, ts_event_ns: int) -> None:
        if ts_event_ns < self.activation_time_ns or self.active:
            return
        self.queue_ahead = OrderedDict(book.queue_at(self.side, self.price))
        self.active = True
        self.queue_trajectory.append((ts_event_ns, sum(self.queue_ahead.values())))

    def _reduce_ahead(self, order_id: int, quantity: int) -> int:
        ahead = self.queue_ahead.get(order_id, 0)
        removed = min(ahead, quantity)
        if removed:
            remaining = ahead - removed
            if remaining:
                self.queue_ahead[order_id] = remaining
            else:
                self.queue_ahead.pop(order_id)
        return removed

    def observe(self, event: MBOEvent, book_before: FIFOBook) -> None:
        if event.key != self.key or event.ts_event_ns > self.horizon_end_ns:
            return
        self.activate(book_before, event.ts_event_ns)
        if not self.active or self.filled_quantity >= self.quantity:
            return
        queue_before = sum(self.queue_ahead.values())
        filled_before = self.filled_quantity
        if event.action == "R":
            self.invalidated_by_reset = True
            self.active = False
            self.queue_trajectory.append((event.ts_event_ns, queue_before))
            return
        if event.action in {"C", "M"}:
            self._reduce_ahead(event.order_id, event.size if event.action == "C" else 2**63 - 1)
        elif event.action == "F" and event.side == self.side and event.price == self.price:
            residual = event.size - self._reduce_ahead(event.order_id, event.size)
            if not self.queue_ahead and residual > 0:
                fill = min(self.quantity - self.filled_quantity, residual)
                if fill:
                    self.filled_quantity += fill
                    self.first_fill_time_ns = self.first_fill_time_ns or event.ts_event_ns
                    if self.filled_quantity == self.quantity:
                        self.completion_time_ns = event.ts_event_ns
        queue_after = sum(self.queue_ahead.values())
        if queue_after != queue_before or self.filled_quantity != filled_before:
            self.queue_trajectory.append((event.ts_event_ns, queue_after))


@dataclass(frozen=True, slots=True)
class SimulatedL3Label:
    label_type: str
    side: str
    quote_price: float
    quantity: int
    queue_ahead_at_entry: int
    filled: bool
    partial_fill: bool
    fill_fraction: float
    first_fill_time_ns: int | None
    completion_time_ns: int | None
    time_to_first_fill_ns: int | None
    invalidated_by_reset: bool
    queue_trajectory: tuple[tuple[int, int], ...]


def finalize_simulated_l3(order: SimulatedPassiveOrder) -> SimulatedL3Label:
    initial = order.queue_trajectory[0][1] if order.queue_trajectory else 0
    return SimulatedL3Label(
        label_type="SIMULATED_L3",
        side=order.side,
        quote_price=order.price,
        quantity=order.quantity,
        queue_ahead_at_entry=initial,
        filled=order.filled_quantity == order.quantity,
        partial_fill=0 < order.filled_quantity < order.quantity,
        fill_fraction=order.fill_fraction,
        first_fill_time_ns=order.first_fill_time_ns,
        completion_time_ns=order.completion_time_ns,
        time_to_first_fill_ns=order.time_to_first_fill_ns,
        invalidated_by_reset=order.invalidated_by_reset,
        queue_trajectory=tuple(order.queue_trajectory),
    )
