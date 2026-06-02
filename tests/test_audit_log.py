from algua.audit.log import append, read
from algua.registry.db import connect, migrate


def test_append_and_read_roundtrip(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    append(conn, actor="agent", action="paper_run", reason="2 orders", strategy="s")
    rows = read(conn, strategy="s")
    assert len(rows) == 1
    assert rows[0]["actor"] == "agent"
    assert rows[0]["action"] == "paper_run"
    assert rows[0]["strategy"] == "s"
