from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    expected_return: float
    expected_markout: float
    volatility: float
    fill_probability: float
    adverse_selection_probability: float
    uncertainty: float
    healthy: bool


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float


def quote_controller(
    mid: float, inventory: float, prediction: Prediction, base_spread: float = 0.25
) -> Quote:
    expected_move = max(-1.0, min(1.0, mid * prediction.expected_return))
    volatility = max(0.0, min(0.01, prediction.volatility))
    reservation = (
        mid + expected_move - 0.02 * inventory - 0.1 * prediction.adverse_selection_probability
    )
    half = (
        base_spread / 2
        + 5 * volatility
        + 0.1 * prediction.adverse_selection_probability
        - 0.05 * prediction.fill_probability
        + 0.01 * abs(inventory)
    )
    half = max(half, 0.125)
    return Quote(reservation - half, reservation + half)


def risk_gate(
    quote: Quote,
    prediction: Prediction,
    inventory: float,
    latency_ms: float,
    pnl: float,
    mid: float | None = None,
) -> tuple[bool, str]:
    if not prediction.healthy or not all(
        math.isfinite(value) for value in vars(prediction).values() if not isinstance(value, bool)
    ):
        return False, "MODEL_HEALTH"
    if not math.isfinite(quote.bid) or not math.isfinite(quote.ask) or quote.bid >= quote.ask:
        return False, "INVALID_QUOTE"
    if mid is not None and (quote.bid < mid - 5 or quote.ask > mid + 5):
        return False, "PRICE_COLLAR"
    if abs(inventory) >= 10:
        return False, "INVENTORY_LIMIT"
    if latency_ms > 50:
        return False, "LATENCY_LIMIT"
    if pnl < -1000:
        return False, "DAILY_LOSS_LIMIT"
    return True, "ALLOW"


def smoke_accounting(
    mid: float, future_mid: float, quote: Quote, inventory: float, cash: float
) -> tuple[float, float, float]:
    if future_mid <= quote.bid:
        inventory += 1
        cash -= quote.bid
    if future_mid >= quote.ask:
        inventory -= 1
        cash += quote.ask
    return inventory, cash, cash + inventory * future_mid
