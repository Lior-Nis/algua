def test_fundamentals_enum_values():
    from algua.data.models import Dataset, Kind

    assert Dataset.FUNDAMENTALS.value == "fundamentals"
    assert Kind.FUNDAMENTALS.value == "fundamentals"
