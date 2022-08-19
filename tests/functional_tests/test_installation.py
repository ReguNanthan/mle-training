def test_pkg_installation():
    try:
        import HousePricePrediction
    except Exception as e:
        assert (
            False
        ), f"Error : {e}. HousePricePrediction pacakage is not intsalled correctly"

    try:
        import pandas, numpy
    except Exception as e:
        assert False, f"Error : {e}. Numpy/Pandas pacakage is not intsalled correctly"

    # with pytest.raises(ImportError):
    #   import pandas
