import dreamer


def test_dreamer_has_version() -> None:
    assert hasattr(dreamer, "__version__")
    assert isinstance(dreamer.__version__, str)
    assert dreamer.__version__
