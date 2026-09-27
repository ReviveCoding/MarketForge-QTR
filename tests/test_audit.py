from pathlib import Path

from marketforge.audit import audit_mbo, read_quality


def test_mbo_audit_fixture(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "sample.csv"
    raw.write_text(
        "ts_recv,ts_event,rtype,publisher_id,instrument_id,action,side,price,size,channel_id,order_id,flags,ts_in_delta,sequence,symbol\n"
        "2025-01-01T00:00:00.000000100Z,2025-01-01T00:00:00.000000000Z,160,1,2,A,B,100.0,1,0,3,0,100,1,ESZ5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("marketforge.audit.data_root", lambda: tmp_path)
    report = audit_mbo(raw)
    result = read_quality(report)
    assert result["metrics"]["row_count"] == 1
    assert result["quality_status"] == "PASS"
