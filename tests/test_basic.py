def test_import_and_version():
    import vega
    assert hasattr(vega, "__version__")
