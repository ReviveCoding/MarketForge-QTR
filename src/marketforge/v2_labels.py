from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SideMarkouts:
    bid_quote_markout: float
    ask_quote_markout: float


def side_specific_quote_markouts(
    future_mid: float, bid_quote: float, ask_quote: float
) -> SideMarkouts:
    return SideMarkouts(
        bid_quote_markout=future_mid - bid_quote,
        ask_quote_markout=ask_quote - future_mid,
    )


def post_fill_markout(side: str, future_mid_after_fill: float, fill_price: float) -> float:
    if side == "B":
        return future_mid_after_fill - fill_price
    if side == "A":
        return fill_price - future_mid_after_fill
    raise ValueError("fill side must be B or A")


def endpoint_crossing_proxy(side: str, future_mid: float, quote_price: float) -> bool:
    """Historical endpoint-only diagnostic; explicitly not an L3 fill label."""
    if side == "B":
        return future_mid <= quote_price
    if side == "A":
        return future_mid >= quote_price
    raise ValueError("quote side must be B or A")
