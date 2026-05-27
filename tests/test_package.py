def test_package_imports():
    import ccnss_helpers
    assert hasattr(ccnss_helpers, "data")
    assert hasattr(ccnss_helpers, "plotting")
    assert hasattr(ccnss_helpers, "save_checkpoint")
    assert hasattr(ccnss_helpers, "load_or_compute")
    assert ccnss_helpers.__version__ == "0.1.0"
