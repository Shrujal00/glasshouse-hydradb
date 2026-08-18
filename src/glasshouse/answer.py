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

import math
import re
from dataclasses import dataclass
from typing import Iterable, Iterator

import ollama

from .config import ADJUDICATION_MODEL, get

# How much of each document reaches the model. The budget is spent on the
# passages that match the question rather than on the opening characters:
# corpus documents run to 5-7k characters and the sentence that answers the
# question is routinely past the halfway mark, so taking the head silently
# dropped the answer and left behind similar-looking numbers that did not
# answer anything.
DOC_CHARS = 2600
MAX_DOCS = 6

# The unit passages are scored and spent in. Wide enough that a matched term
# arrives with the sentence it belongs to, narrow enough that several separate
# regions of a long document fit inside one document's allowance.
_PASSAGE_WINDOW = 520

# The model is told to open with this exact token when the corpus comes up
# short, so abstention is detectable in code rather than inferred from wording.
NOT_FOUND = "NOT_IN_CORPUS"

_STOPWORDS = frozenset(
    "the and for are was were what which who whom whose when where why how "
    "did does do done should would could will can may might must have has had "
    "that this these those there their them they from with without into onto "
    "about above after again against all any because been before being below "
    "between both during each few more most other over same some such than "
    "then they too under until very you your our its his her not but".split()
)

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
6. Graph connections show only that entities co-occur in a document. They do
   not prove collaboration, ownership, agreement, or responsibility.

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


def select_passages(body: str, question: str, budget: int = DOC_CHARS) -> str:
    """The parts of `body` that mention the question's own terms.

    Ranked retrieval decides which documents are worth reading; this decides
    which part of one gets read. The document is scored in fixed windows and
    the budget is spent on the best ones that do not overlap, so a dense
    opening cannot swallow the allowance and hide a later passage -- which is
    exactly how the answer to the burst-credit question went missing.

    Rare terms count for more than common ones. A pool identifier that appears
    twice locates the answer; the word "pool", appearing forty times, does not.
    """
    body = body.strip()
    if len(body) <= budget:
        return body

    terms = {t for t in re.findall(r"[a-z0-9][a-z0-9._/-]{2,}", question.lower())}
    terms -= _STOPWORDS
    lowered = body.lower()
    weights = {}
    for term in terms:
        seen = lowered.count(term)
        if seen:
            weights[term] = 1.0 / math.sqrt(seen)
    if not weights:
        return body[:budget].strip()

    step = _PASSAGE_WINDOW // 2
    scored: list[tuple[float, int]] = []
    for begin in range(0, len(body), step):
        chunk = lowered[begin : begin + _PASSAGE_WINDOW]
        score = sum(weight for term, weight in weights.items() if term in chunk)
        if score:
            scored.append((score, begin))
    if not scored:
        return body[:budget].strip()

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    picked: list[tuple[int, int]] = []
    spent = 0
    for _, begin in scored:
        end = min(len(body), begin + _PASSAGE_WINDOW)
        if any(begin < other_end and other_begin < end for other_begin, other_end in picked):
            continue
        if spent + (end - begin) > budget:
            continue
        picked.append((begin, end))
        spent += end - begin
    if not picked:
        return body[:budget].strip()

    picked.sort()
    parts = [body[begin:end].strip() for begin, end in picked]
    joined = " … ".join(parts)
    return "… " + joined if picked[0][0] > 0 else joined


def build_prompt(
    question: str,
    docs: Iterable,
    people: Iterable,
    paths: Iterable[dict] | None = None,
) -> str:
    """Lay out the evidence, identities first, graph connections last.

    When `paths` is supplied the model sees the multi-hop connections
    HydraDB found between the people this question reached. A shared document
    establishes co-occurrence only, not collaboration, ownership, or agreement.
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
        body = select_passages(d.text or "", question, budget=DOC_CHARS)
        lines.append(f"\n[{i}] {d.source} — {d.title or d.doc_id}"
                     + (f" ({d.date})" if d.date else ""))
        lines.append(body)

    path_list = list(paths or ())
    if path_list:
        lines.append("\nGraph co-occurrences found by HydraDB (shared documents only):")
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
    full: list[str] = []
    pending = ""
    checking_marker = True
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
        full.append(piece)
        # Hold the opening back until it is clear whether it is the abstention
        # marker, so the token never flashes on screen.
        if checking_marker:
            pending += piece
            if NOT_FOUND.startswith(pending):
                continue
            if pending.startswith(NOT_FOUND):
                checking_marker = False
                visible = pending[len(NOT_FOUND) :].lstrip(" :.—-")
                pending = ""
                if visible:
                    yield {"chunk": visible}
                continue
            checking_marker = False
            yield {"chunk": pending}
            pending = ""
            continue
        yield {"chunk": piece}
    if pending:
        yield {"chunk": pending}
    yield {"done": _finish("".join(full))}
