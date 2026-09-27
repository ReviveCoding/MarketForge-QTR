from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .state import ROOT


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    key: str
    source: str
    venue: str
    symbol: str
    asset_class: str
    currency: str
    tick_size: float
    contract_multiplier: float
    tick_value: float
    quantity_unit: str
    timezone: str
    session: dict[str, str]
    fee_convention: str
    fee_value: float

    def ticks_to_price(self, ticks: float) -> float:
        return ticks * self.tick_size

    def ticks_to_usd(self, ticks: float, quantity: float = 1.0) -> float:
        return ticks * self.tick_value * quantity

    def price_move_to_usd(self, price_move: float, quantity: float = 1.0) -> float:
        return price_move * self.contract_multiplier * quantity


def load_registry(path: Path | None = None) -> dict[str, InstrumentSpec]:
    registry_path = path or ROOT / "config" / "instruments_v2.yaml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if payload["schema_version"] != 2:
        raise ValueError("instrument registry schema must be v2")
    output = {}
    for key, value in payload["instruments"].items():
        spec = InstrumentSpec(
            key=key,
            source=value["source"],
            venue=value["venue"],
            symbol=value["symbol"],
            asset_class=value["asset_class"],
            currency=value["currency"],
            tick_size=float(value["tick_size"]),
            contract_multiplier=float(value["contract_multiplier"]),
            tick_value=float(value["tick_value"]),
            quantity_unit=value["quantity_unit"],
            timezone=value["timezone"],
            session=value["session"],
            fee_convention=value["fee"]["convention"],
            fee_value=float(value["fee"]["value"]),
        )
        if abs(spec.tick_size * spec.contract_multiplier - spec.tick_value) > 1e-9:
            raise ValueError(f"dimensionally inconsistent registry entry: {key}")
        output[key] = spec
    return output
