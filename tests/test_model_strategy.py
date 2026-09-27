import torch

from marketforge.model import MarketForgeM3T
from marketforge.strategy import Prediction, Quote, risk_gate


def test_causal_mask_prevents_future_mutation() -> None:
    torch.manual_seed(1)
    model = MarketForgeM3T(18, d_model=16, layers=1, heads=4).eval()
    action = torch.ones(1, 4, dtype=torch.long)
    side = torch.ones(1, 4, dtype=torch.long)
    event = torch.zeros(1, 4, 3)
    state = torch.zeros(1, 4, 18)
    book = torch.zeros(1, 4, 10, 6)
    before = model.encode(action, side, event, state, book)
    action[:, -1] = 6
    event[:, -1] = 99
    state[:, -1] = 99
    book[:, -1] = 99
    after = model.encode(action, side, event, state, book)
    assert torch.equal(before[:, :-1], after[:, :-1])
    assert not torch.equal(before[:, -1], after[:, -1])


def test_risk_gate_kills_invalid_quote() -> None:
    prediction = Prediction(0, 0, 0, 0.5, 0.5, 0, True)
    assert risk_gate(Quote(101, 100), prediction, 0, 1, 0) == (False, "INVALID_QUOTE")


def test_risk_gate_price_collar() -> None:
    prediction = Prediction(0, 0, 0, 0.5, 0.5, 0, True)
    assert risk_gate(Quote(90, 91), prediction, 0, 1, 0, mid=100) == (False, "PRICE_COLLAR")
