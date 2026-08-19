# Demo script — Glasshouse

Five beats, roughly four minutes. Each one shows a different thing, and they
are ordered so the strongest moment lands first while attention is highest.

Server: `.venv/bin/python -m uvicorn glasshouse.server:app --host 127.0.0.1 --port 8080 --app-dir src`

---

## 1 · The system refuses to guess  (~50s)

> **INC-9821: was the degraded GPU node an OOM or intermittent driver/kernel
> launch stalls?**

**Watch for:** the *competing claims* panel. Sixteen claims extracted, three
genuine conflicts found. One is decided on corroboration. Two are **not**:

```
NO ACCEPTED VALUE — trust is too close to choose (0.67 against 0.67)
```

**Say:** "Two documents disagree about who owns this incident. The system scores
both on source authority, recency, explicitness and corroboration — and when the
margin is too thin to justify a choice, it says so and hands both values to the
model rather than inventing a winner. Most RAG systems pick. This one is allowed
to decline."

This is the single most distinctive thing in the project. Lead with it.

---

## 2 · Three ways into the graph  (~40s)

> **In the internal customer success and support knowledge space, which
> published page contains escalation templates?**

**Watch for:** the trace line `hydradb reached from 1 container`.

**Say:** "That question names no person, and its words appear nowhere in the
page's body. It names a *place*. Folders and channels are nodes in HydraDB, so
this is one anchored hop — not a keyword guess. It's the third entrance: the
first needs the question to name someone, the second reads the same edge
backwards from whatever retrieval found, and this one enters from the place."

Worth saying out loud: **the obvious graph entrance is the one that doesn't
work.** Person-seeded retrieval opens for 21 of 570 benchmark questions. We
measured it, and built two more entrances because of what the measurement said.

---

## 3 · Identity resolution  (~30s)

> **In the customer success shared drive, which draft spreadsheet owned by
> Jordan Reyes was last modified most recently?**

**Watch for:** the alias expansion — one person, several spellings — and the
*connections found by hydradb* panel.

**Say:** "209,388 surface forms across nine tools collapse to 166,429 people.
39,847 of them were written more than one way; one person had 22. `sam h`,
`@soham` and `S. Ratnaparkhi` are one node, so a question using any of those
names answers from a document using a different one. This is the part that is
genuinely awkward to do with a vector store."

---

## 4 · Knowing when the answer isn't there  (~25s)

> *(any `info_not_found` question)*

**Watch for:** the answer beginning with a refusal, while still describing what
the documents *do* contain.

**Say:** "Twenty of the benchmark's questions have no answer in the corpus. This
category scores 100%, and it held at 100% through every prompt change we made —
including the ones that made it more willing to answer elsewhere."

---

## 5 · The disagreement map  (~60s)

Switch to the **disagreements** view in the header. Nothing has been asked.

**Watch for:** a ranked list of things the organisation states two ways, each
with both values, both sources, and whether the system settled it.

**Say:** "Everything so far answered a question somebody had. This answers one
nobody asked. Contradiction is stored as an edge in HydraDB, not recomputed per
query — so 'what does this company disagree with itself about' is a ranked read
of a label, not a search. No vector store can do this: you cannot embed your way
to *two documents that disagree*, because they are about the same thing and sit
next to each other in the embedding space. That similarity is the problem, not
the solution."

Then click into one, and use the two buttons on a claim:

- **history** — walks `SUPERSEDES` backwards. "This is what the value used to
  be, and the document that corrected it, at every step. The edge is only
  written where recency is what actually settled the conflict — an unresolved
  disagreement is not a history, and drawing it as one would be a claim about
  time the evidence does not support."
- **who read this** — claim → document → the people on that document. "This is
  the alarming one. These are the people who sent, spoke in, or are named in the
  document asserting the version that lost. Three hops."

Then hit the **refuses to decide** filter.

**Say:** "These are the ones it will not settle. That is the same abstention
you saw in the first question, except now it is a property of the whole corpus
rather than of one answer — a list of the things this company has not actually
decided yet."

**Be precise about scale.** This is built over the work items the most tools
quote, not all 511,962 documents — extraction is a model call. Say that
plainly; the selection is the corpus narrowing itself by cross-quotation, which
is a better story than pretending it is exhaustive.

---

## Showing the benchmark

Run it live or show the output of:

```bash
.venv/bin/python scripts/grade.py --limit 220
```

Say what it is: every answer graded against the benchmark's own `answer_facts`
rubric by an LLM judge, one fact at a time, stratified across all eleven
question types. The answer key lives outside the repository and only the two
scoring scripts can read it — the pipeline cannot see what it is marked against.

**Be honest about the numbers.** The credible version of this story is not "we
scored X". It is:

- what we measured,
- what the measurement told us to build,
- and what we tried that *didn't* work.

The negative results are worth showing: person-seeded graph retrieval (opens on
4% of questions, never added a correct document), and LLM query expansion for
paraphrased questions, which made retrieval measurably **worse** — 9/30 to 5/30
— because the model invents plausible jargon like `safest_numeric_mode` that
exists in no document.

## What not to claim

- Do not claim a graph-native *retrieval* win. The measurement does not support
  it. The graph earns its place on identity and arbitration; say that instead.
- Do not describe the container entrance as graph-only. It runs on HydraDB, and
  the local facet table serves the identical scope as a fallback — the trace
  names which one answered, and so should the narration.
- Do not imply the contradiction graph covers the whole corpus. It covers the
  work items the most tools quote. The honest framing is that cross-quotation
  is what makes two documents *able* to disagree, so that is where the
  disagreements are — not that we ran out of budget.
- Do not call a single-document contradiction a disagreement. One transcript
  listing three thresholds is one text read three times; the loader drops
  those deliberately and says how many it dropped.
- `semantic` questions are weak, and the reason is worth stating plainly:
  they paraphrase deliberately ("too many requests errors" for `429`), and
  keyword retrieval cannot bridge that. The fix is dense retrieval, which this
  stack does not have.
