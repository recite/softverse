"""A long collection must stop when every deposit fails, and only then.

`collect()` records a failure and moves on. Against a host that has started
refusing, that walks the whole frame sending requests it knows will fail, so
the runner counts a streak and stops. The two tests pin both sides: a block
stops the run, and scattered failures in otherwise healthy traffic do not.
"""

from __future__ import annotations

import csv
import sys

import collect_dataverse as runner

from softverse.acquire.state import DatasetRecord
from softverse.model.enums import CollectionState


class _Client:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _setup(tmp_path, monkeypatch, fails):
    frame = tmp_path / "deposits.csv"
    with frame.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protocol", "authority", "identifier"])
        for i in range(40):
            writer.writerow(["doi", "10.7910", f"DVN/X{i:03d}"])
    calls: list[str] = []

    def fake_collect(dois, files_root, raw_root, ledger, client):
        rows = []
        for doi in dois:
            calls.append(doi)
            failed = fails(len(calls))
            state = CollectionState.FAILED if failed else CollectionState.COMPLETE
            ledger.finish(DatasetRecord(dataset_doi=doi, state=state.value))
            if not failed:
                rows.append({"dataset_doi": doi})
        return rows

    monkeypatch.setattr(runner, "FRAME", frame)
    monkeypatch.setattr(runner, "OUT", tmp_path / "out")
    monkeypatch.setattr(runner, "collect", fake_collect)
    monkeypatch.setattr(runner, "PoliteClient", _Client)
    monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["collect_dataverse.py"])
    return calls


def test_a_block_stops_the_run(tmp_path, monkeypatch):
    calls = _setup(tmp_path, monkeypatch, fails=lambda n: n > 5)
    assert runner.main() == 1
    assert len(calls) < 40


def test_scattered_failures_do_not(tmp_path, monkeypatch):
    calls = _setup(tmp_path, monkeypatch, fails=lambda n: n % 3 == 0)
    assert runner.main() == 0
    assert len(calls) == 40
    lines = (tmp_path / "out" / "files.jsonl").read_text().splitlines()
    assert len(lines) == 40 - 40 // 3
