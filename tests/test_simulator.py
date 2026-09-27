from marketforge.simulator import QueueSimulator


def test_latency_and_accounting_invariants() -> None:
    sim = QueueSimulator()
    own = sim.submit("B", 100.0, 1, 10, 10)
    sim.market_trade("B", 100.0, 1, 15)
    assert own.filled == 0
    sim.market_trade("B", 100.0, 1, 20)
    assert own.filled == 1
    assert sim.position == 1 and sim.cash == -100 and sim.nav(101) == 1
