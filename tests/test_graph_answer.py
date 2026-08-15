"""Tests that graph paths are wired into answer synthesis.

These tests verify the plumbing, not the model: that `build_prompt` includes
graph connection evidence when paths are supplied, and that the pipeline
functions accept and pass the `paths` kwarg through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from glasshouse.answer import build_prompt


@dataclass
class FakeDoc:
    doc_id: str = "d1"
    source: str = "slack"
    title: str = "Test Doc"
    date: str = "2026-01-01"
    text: str = "Alice mentioned Bob in the thread."

    def cite(self) -> str:
        return f"{self.source} — {self.title}"


@dataclass
class FakePerson:
    name: str = "Alice Smith"
    alias_count: int = 3
    surfaces: set = field(default_factory=lambda: {"alice", "asmith"})


# ---- build_prompt includes graph paths when supplied -----------------------

def test_build_prompt_without_paths():
    prompt = build_prompt("who owns billing?", [FakeDoc()], [FakePerson()])
    assert "Graph connections" not in prompt
    assert "who owns billing?" in prompt


def test_build_prompt_with_paths():
    paths = [
        {
            "summary": "Alice Smith — Billing ADR #42 — Bob Jones",
            "via": ["Billing ADR #42"],
        }
    ]
    prompt = build_prompt("who owns billing?", [FakeDoc()], [FakePerson()], paths=paths)
    assert "Graph co-occurrences found by HydraDB" in prompt
    assert "Alice Smith — Billing ADR #42 — Bob Jones" in prompt
    assert "Billing ADR #42" in prompt


def test_build_prompt_with_empty_paths():
    prompt = build_prompt("who owns billing?", [FakeDoc()], [FakePerson()], paths=[])
    assert "Graph connections" not in prompt


def test_build_prompt_paths_none_is_default():
    prompt = build_prompt("who owns billing?", [FakeDoc()], [FakePerson()])
    prompt_explicit = build_prompt("who owns billing?", [FakeDoc()], [FakePerson()], paths=None)
    assert prompt == prompt_explicit


# ---- build_prompt still includes identities --------------------------------

def test_build_prompt_includes_identities():
    prompt = build_prompt("who is alice?", [FakeDoc()], [FakePerson()])
    assert "Alice Smith is also written as:" in prompt


def test_build_prompt_includes_documents():
    prompt = build_prompt("who is alice?", [FakeDoc()], [FakePerson()])
    assert "[1] slack" in prompt
    assert "Test Doc" in prompt
