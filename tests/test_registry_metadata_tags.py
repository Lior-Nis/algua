from algua.registry.metadata import canonicalize_tags, dump_tags, load_tags


def test_canonicalize_trims_lowercases_dedupes_sorts():
    assert canonicalize_tags([" Mean-Reversion ", "MOMENTUM", "momentum"]) == [
        "mean-reversion", "momentum"
    ]


def test_canonicalize_rejects_empty():
    assert canonicalize_tags(["", "  ", "x"]) == ["x"]


def test_dump_then_load_roundtrips():
    assert load_tags(dump_tags(["b", "a"])) == ["a", "b"]


def test_load_handles_null_and_garbage():
    assert load_tags(None) == []
    assert load_tags("not json") == []
    assert load_tags("[]") == []
