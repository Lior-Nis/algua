def test_fundamentals_enum_values():
    from algua.data.models import Dataset, Kind

    assert Dataset.FUNDAMENTALS.value == "fundamentals"
    assert Kind.FUNDAMENTALS.value == "fundamentals"


def test_news_dataset_and_kind():
    from algua.data.models import Dataset, Kind

    assert Dataset.NEWS.value == "news"
    assert Kind.NEWS.value == "news"
