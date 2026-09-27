from marketforge.features import Book


def test_book_official_action_semantics() -> None:
    book = Book()
    book.apply("A", "B", 1, 100.0, 5)
    book.apply("A", "A", 2, 101.0, 3)
    book.apply("F", "A", 2, 101.0, 2)
    assert book.top("A")[0] == (101.0, 3, 1)
    book.apply("C", "B", 1, 100.0, 2)
    assert book.top("B")[0] == (100.0, 3, 1)
    book.apply("M", "A", 2, 102.0, 4)
    assert book.top("A")[0] == (102.0, 4, 1)
    book.apply("R", "N", 0, None, 0)
    assert not book.orders
