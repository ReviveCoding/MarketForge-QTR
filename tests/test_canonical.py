from marketforge.canonical import _epoch_ns


def test_epoch_ns_expression_preserves_fraction() -> None:
    expression = _epoch_ns("ts_event")
    assert "substr(ts_event, 21, 9)" in expression
    assert "epoch_ns" in expression
