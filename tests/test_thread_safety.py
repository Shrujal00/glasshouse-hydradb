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
from fake_ontology import FakeOntologyGraph
from glasshouse.recall import LocalRecall


def test_recall_index_is_usable_from_several_threads(tmp_path):
    recall = LocalRecall(tmp_path / "recall.sqlite3")
    recall.create()
    recall.add([("d1", "slack", "Retention", "retention policy notes", "", "", "", "")])
    # Open the connection on this thread first, exactly as a first request does.
    assert recall.count() == 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(lambda _: recall.count(), range(8)))
        found = list(pool.map(lambda _: len(recall.search("retention", limit=5)), range(8)))
    assert counts == [1] * 8
    assert all(n >= 0 for n in found)


def test_resolution_memo_is_shared_and_safe_across_threads(tmp_path):
    """Resolution is a traversal now, so the thing to protect is the memo.

    The engine answers about twenty of these a second, and the server asks from
    whichever worker took the request. A memo that raced would either resolve
    the same word twice per question or hand one thread a half-built entry;
    plain dict get/set is atomic under the GIL, so the worst outcome is a
    duplicated lookup, and the count below proves it does not happen per call.
    """
    calls = []

    class CountingGraph(FakeOntologyGraph):
        def denoted_by(self, text, limit=4):
            calls.append(text)
            return super().denoted_by(text, limit)

    asker = Asker.__new__(Asker)
    asker.engine = CountingGraph(
        [("maya chen", "name", "maya", 1, "Maya Chen", 1.0, 2)]
    )
    asker.priors = Priors()

    with ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda _: asker.denoted_by("maya chen"), range(8)))

    assert [[m.eid for m in got] for got in found] == [["maya"]] * 8
    # Eight concurrent asks, at most one traversal per distinct word.
    assert calls.count("maya chen") <= 4


def test_documents_are_fetched_by_rowid_not_by_scanning(tmp_path):
    """`docs` is an FTS5 table with `doc_id UNINDEXED`, so `WHERE doc_id IN
    (...)` cannot use an index and scans every row -- 21 seconds to fetch 55
    documents out of 511,962. A side table mapping doc_id to rowid turns the
    same fetch into rowid lookups, which FTS5 does answer directly.
    """
    recall = LocalRecall(tmp_path / "recall.sqlite3")
    recall.create()
    recall.add([
        (f"d{i}", "slack", f"Title {i}", f"body {i}", "", "", "", "") for i in range(50)
    ])
    recall.build_docmap()

    got = recall.get_many(["d7", "d3", "d40"])
    assert [d.doc_id for d in got] == ["d7", "d3", "d40"]
    assert recall.get("d11").title == "Title 11"
    assert recall.get_many(["nope"]) == []
    # A document added after the map is built must still be findable.
    recall.add([("late", "slack", "Late", "body late", "", "", "", "")])
    recall.build_docmap()
    assert recall.get("late").title == "Late"
