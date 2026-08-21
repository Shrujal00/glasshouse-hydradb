# Demo script — Glasshouse · 3 minutes

Hack Hydra Track 01. The rules say the video must cover four things, so this is
built as those four, in order:

1. the problem you are trying to solve
2. what you actually built
3. a demo of the project working
4. how you used the HydraDB repo and why it matters

**Hard rule: stop at 3:00.** Anything past the mark may not be reviewed.

Before you record: `docker compose up -d`, then

```bash
.venv/bin/python -m uvicorn glasshouse.server:app --host 127.0.0.1 --port 8080 --app-dir src
```

Open `http://127.0.0.1:8080`. It lands on **disagreements**. Leave it there.

---

## 0:00 — 0:25 · The problem

**On screen:** the disagreements list, already loaded. Don't touch anything yet.

> "A company keeps the same fact in nine different apps. Slack, Gmail, Jira,
> Confluence, GitHub, Drive, HubSpot, Linear, Fireflies.
>
> Over time those facts drift apart, and nobody notices — because nobody is
> looking. You only find out when someone acts on the wrong one."

*Don't say "RAG". Don't say "knowledge graph" yet. Say the problem.*

---

## 0:25 — 0:55 · What it is

**On screen:** scroll the list slowly — the scroll itself is the point, there
are 173 of these. **ENG-4824 · Who owns this?** is about 30 rows down. Click it.

*(If you would rather not scroll on camera: row 7, `PR-28644`, is answered with
short values — linear says "Landed PR + hotfix", github says "Approved".)*

> "Glasshouse read half a million documents across all nine tools and found
> every place the company wrote the same thing down two different ways.
>
> Nobody asked it a question. This list already existed when I opened the page.
>
> Here — who owns ENG-4824? Confluence says Benji Okafor and Sofia Ivanova.
> Slack says Liam and Maria. Three tools, three answers.
>
> It picked Confluence, and it says why: a reviewed page outranks a chat
> message."

---

## 0:55 — 1:40 · The demo — walk the graph

**On screen:** the middle panel now shows the disagreement drawn.

> "And this is the part I actually care about. That's not a picture of the
> data — it *is* the data."

**Click a position circle.** Documents fan out.

> "One click, one hop. These are the documents that assert that answer."

**Click a document.** People fan out.

> "Another hop — and now I know who read it."

**Go back to the list. Click the `refuses to decide` filter.**

> "And these are the ones it won't settle — forty-eight of a hundred and
> seventy-three. Two sources, equally credible, and the honest answer is that
> this company hasn't actually decided yet.
>
> Most systems pick. This one is allowed to say it doesn't know — and it tells
> you why it can't."

*This is the strongest 20 seconds in the video. Slow down here.*

---

## 1:40 — 2:15 · The ontology

**On screen:** click **THE ONTOLOGY**.

> "None of that works unless you know who everybody is.
>
> Four hundred thousand written forms across nine tools become a hundred and
> sixty-six thousand people. But look at the number that matters —"

**Point at `163,262 merges refused`.**

> "— it refused nearly four times more merges than it made. That's the
> difference between an ontology and string matching."

**Type `irene choi`, press resolve.**

> "Twenty-two ways of writing one person, including typo'd domains — and every
> single one carries the evidence that merged it. Hover any of them and it
> tells you which rule fired."

---

## 2:15 — 2:50 · Why HydraDB

**On screen:** stay on the ontology, or go back to a drawn disagreement.

> "Everything you just saw is HydraDB doing the work, not storing the result.
>
> Who somebody is, is a traversal — one hop, ninety milliseconds. What
> contradicts what, is an edge. So 'what does this company contradict itself
> about' is one query over a graph, not a search. You cannot ask that of a
> search box, because nobody typed a question.
>
> I measured it: turn HydraDB off and every one of those four questions returns
> zero. It isn't sitting in the README. It's the product."

---

## 2:50 — 3:00 · Land it

> "Half a million documents, all nine tools, graded fact by fact against a key
> the pipeline can't see — including the things that didn't work.
>
> That's Glasshouse."

**Stop recording.**

---

## Numbers you may be asked, all measured

| | |
|---|---|
| Documents loaded | 511,962 across 9 tools |
| Written forms → people | 401,163 → 166,429 |
| Merges accepted / refused | 42,959 / **163,262** |
| Ontology build time | 22.6s |
| Disagreements found | 173, of which 48 it refuses to settle |
| Claims in the graph | 3,356 |
| CONTRADICTS / SUPERSEDES edges | 816 / 75 |
| Identity resolution | one hop, ~0.09s |
| Container hop | 159,030 documents → 6 |
| Disagreement map | 0.06s |
| Tests | 183 passing |
| Fact recall | ~44% weighted; 100% on knowing the answer is absent |

## Say these if asked

- **"Is it deterministic?"** — Arbitration makes no model call. It's arithmetic
  on source authority, recency, corroboration and hedging. Ten runs with the
  input order shuffled give byte-identical verdicts.
- **"Why is search still in SQLite?"** — Because we measured it. BM25 over
  511,962 documents is the unanchored scan the engine rejects; moving it made
  the product slower and worse. The graph earns its place on identity and
  arbitration.
- **"How much of the corpus did you extract claims from?"** — 1,406 documents
  of 511,962, about a quarter of one percent, and it found 173 disagreements.
  Reading all of it is a few hours of compute, not a research problem.
- **"Why do some rows say one tool contradicts itself?"** — Because it does:
  Linear says "Done" in one document and "merged" in another. 41 of the 173 are
  a single tool disagreeing with itself across two documents, and those are
  labelled rather than hidden.

## Do not claim

- **Do not claim a graph-native *retrieval* win.** Person-seeded graph
  retrieval opens on 21 of 570 questions and never added a correct document.
  We measured it, and we built two other entrances because of what it said.
- **Do not imply the contradiction graph covers the whole corpus.** It covers
  the work items the most tools quote. Cross-quotation is what makes two
  documents *able* to disagree, so that is where the disagreements are.
- **Do not call a single-document contradiction a disagreement.** One
  transcript listing three thresholds is one text read three times; the loader
  drops those and reports how many it dropped.
