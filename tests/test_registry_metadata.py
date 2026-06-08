from algua.contracts.registry_metadata import Author, HypothesisStatus


def test_author_values():
    assert {a.value for a in Author} == {"agent", "human"}


def test_hypothesis_status_values():
    assert {h.value for h in HypothesisStatus} == {
        "untested", "supported", "refuted", "inconclusive"
    }


def test_enums_are_strenum():
    assert Author.AGENT == "agent"
    assert HypothesisStatus.UNTESTED == "untested"
