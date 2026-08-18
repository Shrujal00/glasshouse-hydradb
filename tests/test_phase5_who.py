"""Entering the graph from the evidence rather than from a person.

`documents_for_entities` needs the question to name somebody, so it opens for
21 of 570 benchmark questions and never for the ones that matter most: "who
owns the audit-log shipper sidecar?" asks for a person as the *answer*. Read
the same edge backwards -- from the documents retrieval already found to the
people connected to them -- and the graph finally has something to say. An
entity connected to four of the six documents about a component is a different
claim from one connected to a single document, and counting that is a graph
operation, not a keyword one.
"""

from __future__ import annotations

from glasshouse.graph import GraphEngine, Row, node_id


def _path(document_node: int, entity_node: int, eid: str, name: str) -> Row:
    return Row(
        {
            "path": {
                "nodes": [
                    {
                        "id": document_node,
                        "labels": ["Document"],
                        "properties": {"doc_id": "d"},
                    },
                    {
                        "id": entity_node,
                        "labels": ["Entity"],
                        "properties": {"eid": eid, "canonical_name": name},
                    },
                ],
                "relationships": [
                    {"edge_type": "MENTIONED_IN", "src": entity_node, "dst": document_node}
                ],
            }
        }
    )


class Reverse(GraphEngine):
    """Maya is in every document, Devon in one."""

    def __init__(self):
        self.calls = []

    def query(self, cypher, parameters=None, **kwargs):
        self.calls.append(cypher)
        node = int(cypher.split("sourceNode: ")[1].split(",")[0])
        rows = [_path(node, 1, "maya", "Maya Chen")]
        if node == node_id("doc:a"):
            rows.append(_path(node, 2, "devon", "Devon Ray"))
        return rows


def test_entities_are_reached_backwards_from_retrieved_documents():
    engine = Reverse()
    found = engine.entities_for_documents(["a", "b", "c"], limit=10)

    assert all("relDirection: 'incoming'" in call for call in engine.calls)
    assert all("algo.SSpaths" in call for call in engine.calls)
    # Ranked by how much of the evidence each person is connected to.
    assert [e.eid for e in found] == ["maya", "devon"]
    assert found[0].name == "Maya Chen"
    assert found[0].doc_ids == ("a", "b", "c")
    assert found[0].documents == 3
    assert found[1].documents == 1


def test_no_documents_means_no_traversal():
    engine = Reverse()
    assert engine.entities_for_documents([], limit=10) == []
    assert engine.calls == []


def test_document_scan_is_bounded_by_the_page_it_was_given():
    engine = Reverse()
    engine.entities_for_documents([f"d{i}" for i in range(40)], limit=10, documents=6)
    assert len(engine.calls) == 6
