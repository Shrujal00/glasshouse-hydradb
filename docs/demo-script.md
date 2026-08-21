# Demo script — 3 minutes

**Five clicks. Two paste-ins. No typing, no scrolling.**

Have these two lines copied somewhere you can paste from — a notes app, a
second window. Never type on camera.

---

## Before you hit record

Server is already running on **http://127.0.0.1:8080**. It opens on the
disagreements screen. Leave it there.

Both questions below are already cached, so they answer in about 5 seconds and
still stream visibly.

---

## 0:00 — The problem

**Do nothing. The list is already on screen.**

> "A company writes the same fact down in nine different tools. Slack, Gmail,
> Jira, Confluence, GitHub, Drive, HubSpot, Linear, Fireflies.
>
> Over time those copies drift apart, and nobody notices — because nobody is
> looking. You find out when somebody acts on the wrong one."

---

## 0:20 — What I built

**Still nothing. Just point at the list.**

> "This read half a million documents across all nine tools and found every
> place the company contradicts itself.
>
> Nobody asked a question. This list already existed when I opened the page.
> There are a hundred and seventy-three of them."

---

## 0:40 — Show one

**Click the third row — `Predictive mode`.**

> "HubSpot says this feature is disabled. HubSpot also says it's enabled. Same
> tool, two answers, and both look equally trustworthy.
>
> So it refuses to pick. Forty-eight of the hundred and seventy-three end this
> way — and it tells you exactly why it can't decide, in one line you can
> check."

*Read the refusal line off the screen if you want. It names the numbers.*

---

## 1:05 — Ask it something

**Click `ASK`. Paste this. Press enter.**

```
In the internal customer success and support knowledge space, which published page by Elliot Price contains copy-paste templates for customer-safe escalation and incident updates?
```

**Talk while it streams:**

> "Watch what happens on the left. It resolves Elliot Price into every way he's
> ever been written — six different forms, including two typo'd email domains.
>
> Then it walks into the Confluence folder the question named, and pulls the
> pages out of it. That folder holds two hundred and thirty-six documents out of
> half a million.
>
> The question doesn't share a single keyword with that page. Keyword search
> can't find it. It names a *place*, and places are things you can walk to."

**Point at the badge under the answer.**

> "And that's scored — five of six required facts, marked by a separate judge
> against an answer key this system cannot see."

---

## 1:50 — Ask it something it doesn't know

**Paste this. Press enter.**

```
For the admin activity chronicle's daily Merkle-root anchoring, which public blockchain network do we anchor to and what smart contract address is used, and how should an auditor verify the anchor end-to-end?
```

> "This one isn't in the corpus. Most systems would invent something plausible.
>
> It says it doesn't know. And that's scored too — one out of one. On the whole
> category of questions where the answer isn't there, it gets a hundred percent."

---

## 2:20 — Why HydraDB

**Click `THE ONTOLOGY`.**

> "All of that runs on HydraDB, used two ways.
>
> Four hundred thousand written names became a hundred and sixty-six thousand
> real people — and look at the number that matters: it *refused* a hundred and
> sixty-three thousand merges. Nearly four times more than it accepted.
>
> Knowing who someone is takes one hop, ninety milliseconds. And 'what does this
> company contradict itself about' is one query, because contradiction is stored
> as a connection between two claims.
>
> You cannot type that into a search box. There's no word to search for. Turn the
> engine off and it returns nothing."

---

## 2:50 — Land it

> "Half a million documents. Nine tools. Graded fact by fact against a key it
> can't see.
>
> That's Glasshouse."

**Stop.**

---

# Cheat sheet

## Numbers, if you're asked

| | |
|---|---|
| Documents | 511,962 across 9 tools |
| Names → people | 401,163 → 166,429 |
| Merges refused | **163,262** (vs 42,959 accepted) |
| Disagreements | 173, of which **48 it refuses to settle** |
| Claims in the graph | 3,356 |
| Who-is-this | one hop, **0.09s** |
| The disagreement map | **0.06s** |
| Folder hop | 159,030 documents → 6 |
| Tests | 183 passing |
| Fact recall | 199 of 424 — **47%** over 100 graded answers |
| Knowing the answer is absent | **100%**, 20 of 20 |

## If they ask

**"Is it deterministic?"**
Yes. Deciding who's right makes no model call — it's arithmetic on source
authority, recency and corroboration. Ten runs with the input shuffled give
byte-identical answers.

**"Why is search still in SQLite?"**
Because we measured it. Keyword search over 511,962 documents is the kind of
full sweep the engine is not built for. The graph earns its place on identity
and contradiction instead.

**"Why does one tool contradict itself?"**
Because it does. Linear says "Done" in one document and "merged" in another.
Those get labelled, not hidden.

**"How much of the corpus has claims extracted?"**
1,406 documents — about a quarter of one percent — and it found 173
disagreements. Reading all of it is compute, not a research problem.

## Don't say

- Don't claim the graph beats keyword search at **finding documents**. It doesn't,
  and it isn't the claim. It wins on the questions keyword search cannot ask.
- Don't imply every document has claims extracted. It's the work items the most
  tools quote.
