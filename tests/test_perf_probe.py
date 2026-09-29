"""Tests for CRYOVIT_PERF stage probes."""

import json
import os

import pytest

from cryovit import perf_probe as pp


def test_probe_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CRYOVIT_PERF", raising=False)
    pp._ENABLED = None
    pp.probe_reset()
    with pp.probe_stage("x"):
        pass
    assert pp.probe_snapshot() == {}


def test_probe_records_stage(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYOVIT_PERF", "1")
    pp._ENABLED = None
    pp.probe_reset()
    with pp.probe_stage("dino.forward"):
        pass
    with pp.probe_stage("dino.forward"):
        pass
    snap = pp.probe_snapshot()
    assert snap["dino.forward"]["count"] == 2
    assert snap["dino.forward"]["total_s"] >= 0
    path = pp.probe_dump(tmp_path / "perf.jsonl", case="dino")
    row = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert row["case"] == "dino"
    assert "dino.forward" in row["stages"]
    assert pp.probe_snapshot() == {}


def test_probe_nested_stages(monkeypatch):
    monkeypatch.setenv("CRYOVIT_PERF", "1")
    pp._ENABLED = None
    pp.probe_reset()
    with pp.probe_stage("a"):
        with pp.probe_stage("b"):
            pass
    snap = pp.probe_snapshot()
    assert set(snap) == {"a", "b"}
