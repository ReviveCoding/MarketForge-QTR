from marketforge.state import content_hash


def test_content_hash_is_order_independent() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
