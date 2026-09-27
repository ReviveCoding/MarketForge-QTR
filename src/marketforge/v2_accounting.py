from __future__ import annotations

from dataclasses import dataclass

from .specs import InstrumentSpec


@dataclass(slots=True)
class AccountingBreakdown:
    price_pnl_usd: float = 0.0
    spread_capture_usd: float = 0.0
    fees_usd: float = 0.0
    slippage_usd: float = 0.0
    liquidation_cost_usd: float = 0.0
    hedge_cost_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    nav_usd: float = 0.0


def futures_round_trip(
    spec: InstrumentSpec,
    entry_price: float,
    exit_price: float,
    signed_contract_quantity: float,
    reference_mid_at_entry: float,
    fee_per_contract_per_side_usd: float,
    entry_slippage_ticks: float = 0.0,
    exit_slippage_ticks: float = 0.0,
    liquidation_cost_usd: float = 0.0,
    hedge_cost_usd: float = 0.0,
) -> AccountingBreakdown:
    quantity = abs(signed_contract_quantity)
    price_pnl = (exit_price - entry_price) * spec.contract_multiplier * signed_contract_quantity
    spread_capture = (
        (reference_mid_at_entry - entry_price) * spec.contract_multiplier * signed_contract_quantity
    )
    fees = 2 * fee_per_contract_per_side_usd * quantity
    slippage = spec.ticks_to_usd(entry_slippage_ticks + exit_slippage_ticks, quantity)
    realized = price_pnl - fees - slippage - liquidation_cost_usd - hedge_cost_usd
    return AccountingBreakdown(
        price_pnl_usd=price_pnl,
        spread_capture_usd=spread_capture,
        fees_usd=fees,
        slippage_usd=slippage,
        liquidation_cost_usd=liquidation_cost_usd,
        hedge_cost_usd=hedge_cost_usd,
        realized_pnl_usd=realized,
        unrealized_pnl_usd=0.0,
        nav_usd=realized,
    )
