"""Turning evidence into an answer — the last step, and the one with teeth.

Everything before this narrows half a million documents to a handful and works
out who the people in them are. This writes the sentence, and it is where a
system of this kind usually goes wrong: asked a question the corpus cannot
answer, a language model will produce a confident, fluent, invented answer, and
20 of the benchmark's 500 questions exist purely to catch that.

So the model is given three jobs in a fixed order, and refusing is one of them:

1. Answer only from the supplied documents.
2. Cite which document each claim came from.
3. Say plainly when the documents do not contain the answer — while still
   surfacing what *is* there, because a bare "I don't know" is less useful and
   scores worse than a caveated partial answer.

The resolved identities are handed over with the documents, which is where the
ontology repays itself: the model is told outright that `sam h`, `@soham` and
`S. Ratnaparkhi` are one person, so a question about any of those names can be
answered from a document that uses a different one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator

import ollama

from .config import ADJUDICATION_MODEL, get

# Enough of each document to carry the answer without burying it. The recall
# layer already ranked them, so the useful part is near the top.
DOC_CHARS = 2600
MAX_DOCS = 6

# The model is told to open with this exact token when the corpus comes up
# short, so abstention is detectable in code rather than inferred from wording.
NOT_FOUND = "NOT_IN_CORPUS"

SYSTEM = """You answer questions about a company's internal documents.

Rules, in order of importance:
1. Use ONLY the documents provided. Never use outside knowledge. Never guess.
2. If the documents do not contain the answer, begin your reply with the exact
   token NOT_IN_CORPUS, then say in one sentence what is missing, then briefly
   describe what related information the documents DO contain.
3. Otherwise answer directly in 1-3 sentences. Lead with the specific fact —
   the number, name, threshold, or endpoint — not with preamble.
4. Cite the documents you used by their bracketed number, like [2].
5. Never invent a citation. Never cite a document you did not use.
6. If graph connections are provided, use them to reason about how people relate
   to each other and to documents. A graph path connecting two people through a
   document is evidence that they collaborate on or co-own whatever that
   document concerns.

Be terse. No preamble, no restating the question, no "based on the documents"."""


@dataclass(slots=True)
class Written:
    text: str
    abstained: bool
    cited: list[int]


def _client() -> ollama.Client:
    key = get("OLLAMA_API_KEY")
    return ollama.Client(
        host=get("OLLAMA_HOST", "https://ollama.com"),
        headers={"Authorization": f"Bearer {key}"} if key else None,
    )


def build_prompt(
    question: str,
    docs: Iterable,
    people: Iterable,
    paths: Iterable[dict] | None = None,
) -> str:
    """Lay out the evidence, identities first, graph connections last.

    When `paths` is supplied the model sees the multi-hop connections
    HydraDB found between the people this question reached.  That
    evidence can change the answer: a path linking two people through
    a shared document is proof of collaboration that a keyword search
    would miss.
    """
    lines: list[str] = []

    known = [p for p in people if getattr(p, "alias_count", 1) > 1]
    if known:
        lines.append("Identities already resolved (the same person, written several ways):")
        for p in known[:8]:
            lines.append(f"  {p.name} is also written as: {', '.join(sorted(p.surfaces))}")
        lines.append("")

    lines.append("Documents:")
    for i, d in enumerate(list(docs)[:MAX_DOCS], start=1):
        body = (d.text or "")[:DOC_CHARS].strip()
        lines.append(f"\n[{i}] {d.source} — {d.title or d.doc_id}"
                     + (f" ({d.date})" if d.date else ""))
        lines.append(body)

    path_list = list(paths or ())
    if path_list:
        lines.append("\nGraph connections found by HydraDB (entity relationships across documents):")
        for p in path_list[:6]:
            lines.append(f"  {p.get('summary', '')}")
            via = p.get("via") or []
            if via:
                lines.append(f"    via: {', '.join(str(v)[:60] for v in via[:3])}")

    lines.append(f"\n\nQuestion: {question}")
    return "\n".join(lines)


# The model reaches for CJK brackets — 【2】 — as readily as ASCII ones, so
# citations are normalised rather than demanded. Insisting on one form in the
# prompt and then silently dropping the other loses every citation the answer
# actually made.
_CITE = re.compile(r"[\[【](\d{1,2})[\]】]")


def normalise_citations(text: str) -> str:
    return _CITE.sub(lambda m: f"[{m.group(1)}]", text)


def _finish(text: str) -> Written:
    text = normalise_citations(text.strip())
    abstained = text.startswith(NOT_FOUND)
    if abstained:
        text = text[len(NOT_FOUND) :].lstrip(" :.—-")
    return Written(
        text=text,
        abstained=abstained,
        cited=sorted({int(n) for n in _CITE.findall(text)}),
    )


def write(
    question: str, docs, people, model: str | None = None,
    paths: Iterable[dict] | None = None,
) -> Written:
    """Answer in one shot."""
    response = _client().chat(
        model=model or ADJUDICATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, docs, people, paths=paths)},
        ],
        options={"temperature": 0},
    )
    return _finish(response["message"]["content"])


def write_streaming(
    question: str, docs, people, model: str | None = None,
    paths: Iterable[dict] | None = None,
) -> Iterator[dict]:
    """Answer token by token, so the interface can show it being written.

    Yields `{"chunk": ...}` as text arrives and a final `{"done": Written}`.
    The abstention token is stripped from the visible stream: it is a control
    signal for us, not something to show the reader.
    """
    buffer: list[str] = []
    stream = _client().chat(
        model=model or ADJUDICATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, docs, people, paths=paths)},
        ],
        options={"temperature": 0},
        stream=True,
    )
    for part in stream:
        piece = part.get("message", {}).get("content", "")
        if not piece:
            continue
        buffer.append(piece)
        # Hold the opening back until it is clear whether it is the abstention
        # marker, so the token never flashes on screen.
        joined = "".join(buffer)
        if len(joined) < len(NOT_FOUND) and NOT_FOUND.startswith(joined.strip()):
            continue
        yield {"chunk": piece}
    yield {"done": _finish("".join(buffer))}
