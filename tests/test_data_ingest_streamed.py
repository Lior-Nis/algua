import pytest

from algua.data.contracts import ImportRequest
from algua.data.importers import get_importer, register_importer


def test_get_importer_unknown_raises():
    with pytest.raises(ValueError, match="unsupported bar importer: nope"):
        get_importer("nope")


def test_register_and_get_importer_roundtrip():
    class _Dummy:
        name = "dummy"

        def import_bars(self, request):
            return iter(())

    register_importer("dummy", lambda: _Dummy())
    try:
        assert get_importer("dummy").name == "dummy"
    finally:
        from algua.data.importers import _REGISTRY

        del _REGISTRY["dummy"]


def test_import_request_defaults(tmp_path):
    req = ImportRequest(raw_dir=tmp_path / "raw", adjusted_dir=tmp_path / "adj")
    assert req.timeframe == "1d"
    assert req.adjustment == "split_div"
    assert req.symbols is None
    assert req.as_of is None
