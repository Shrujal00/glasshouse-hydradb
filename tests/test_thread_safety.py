"""One Asker serves every request, and FastAPI answers on a worker pool.

A SQLite connection belongs to the thread that opened it, so a lazily opened
index worked for whichever request arrived first and raised "SQLite objects
created in a thread can only be used in that same thread" for the next one
that landed on a different worker. The failure is intermittent by nature,
which is the worst kind to meet during a demo.
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

from glasshouse.ask import Asker
from glasshouse.priors import Priors
from glasshouse.recall import LocalRecall


def test_recall_index_is_usable_from_several_threads(tmp_path):
    recall = LocalRecall(tmp_path / "recall.sqlite3")
    recall.create()
    recall.add([("d1", "slack", "Retention", "retention policy notes", "", "", "")])
    # Open the connection on this thread first, exactly as a first request does.
    assert recall.count() == 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(lambda _: recall.count(), range(8)))
        found = list(pool.map(lambda _: len(recall.search("retention", limit=5)), range(8)))
    assert counts == [1] * 8
    assert all(n >= 0 for n in found)


def test_ontology_lookup_is_usable_from_several_threads(tmp_path):
    path = tmp_path / "ontology.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE alias (surface TEXT, kind TEXT, eid TEXT, node_id INTEGER, "
        "canonical_name TEXT, confidence REAL, alias_count INTEGER)"
    )
    conn.execute("INSERT INTO alias VALUES ('maya chen','name','maya',1,'Maya Chen',1.0,2)")
    conn.commit()
    conn.close()

    asker = Asker.__new__(Asker)
    asker._lookup_path = path
    asker._lookup = None
    asker.priors = Priors()
    assert asker.lookup.execute("SELECT count(*) FROM alias").fetchone()[0] == 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(
            lambda _: asker.lookup.execute("SELECT count(*) FROM alias").fetchone()[0],
            range(8),
        ))
    assert rows == [1] * 8
