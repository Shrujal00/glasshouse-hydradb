"""Entity resolution — deciding that 'Sam', '@soham' and 'S. Ratnaparkhi' are one person.

This is the hardest scored thing in the track, and it is done here without an
LLM. The reason it can be is that identity in an enterprise corpus is mostly
*structured*: `Maya Chen <maya.chen@example.com>` is an explicit binding the
corpus states outright, and handles are derived from names by a small number
of conventions (first, first.last, flast, firstlast).

The pipeline is four stages:

1. **Mine** every identity surface out of the normalized corpus, with counts.
2. **Block** surfaces into small candidate groups, so we never compare all
   pairs. Blocking keys are name-shaped: last token, first token, the compacted
   full name, and first-initial+last.
3. **Score** every candidate pair against a set of named signals. Each signal
   is a separate piece of evidence with its own weight, and every one that
   fires is recorded - a merge that cannot say why it happened is not
   auditable, and auditability is the whole point.
4. **Cluster** by union-find, applying merges strongest-evidence-first and
   refusing any merge that would violate a hard constraint.

Two constraints do most of the precision work. One person has one mailbox
name, and one person has one employer — so a cluster holding two distinct
mailboxes, or addresses at two different companies, is a false merge by
construction.

Nothing in this module knows anything about the company it is pointed at.
Which domains are the employer's, which words are job titles rather than
surnames, and which mailboxes belong to departments are all learned from the
corpus itself in `priors.py`, and arrive here as `Priors`.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .priors import Priors, domain_label

# Blocks bigger than this are dropped rather than compared pairwise - a block
# of 500 first-name matches is 125k comparisons that yield nothing but noise.
# Dropped blocks are reported, never silently skipped.
MAX_BLOCK = 240

# Surfaces rarer than this are ignored. A handle seen twice carries no
# resolvable signal and adds a long tail of speculative merges.
MIN_OCCURRENCES = 5

MERGE_THRESHOLD = 0.70

_PUNCT = re.compile(r"[^\w\s'’-]+")
_DIGITS = re.compile(r"\d+")
_SPLIT = re.compile(r"[._\-\s]+")

# "Maya Chen (Redwood AE)" — the qualifier describes the role, not the person,
# and leaving it in produces a surname of "AE" that then collides with every
# other account executive in the corpus.
_PAREN = re.compile(r"\([^)]*\)")

# A usable email address. The normalizer's `Name <...>` pattern is greedy
# enough to swallow an entire meeting transcript when a stray '<' appears, so
# the resolver validates rather than trusting its input.
_VALID_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)+$")

# Conversational and structural words that occupy the name slot without naming
# anybody: "Meeting Header", "Action Items", "The Team". This is the one list
# left in the source, and it is deliberately *language* vocabulary rather than
# *company* vocabulary — it would be identical for any English corpus, in the
# same way the stemmer knows English. Everything company-specific — which
# domains are the employer's, which words are job titles, which mailboxes
# belong to departments — is learned from the corpus in `priors.py`.
_NOT_A_PERSON = re.compile(
    r"^(?:the|team|group|all|everyone|here|channel|room|meeting|header|attendees|"
    r"participants|action|items?|notes?|agenda|summary|update|re|fwd|hi|hey|"
    r"thanks|regards|best|cheers|unknown|tbd|n/?a)$",
    re.I,
)

# Given names whose bare use is too common to resolve on their own are still
# allowed through when exactly one candidate person carries them; the
# uniqueness test lives in `_given_name_owner`, not in a stoplist.


@dataclass(slots=True)
class Surface:
    """One observed identity string, with everything known about where it came from."""

    key: str  # "email:m@x.com" — kind-prefixed, unique
    kind: str  # email | handle | name
    value: str
    count: int = 0
    docs: int = 0
    sources: Counter = field(default_factory=Counter)
    orgs: Counter = field(default_factory=Counter)
    # Both are decided by learned priors at mining time rather than by a rule
    # in this module, so that pointing the resolver at a different company
    # changes the answers without changing the code.
    org: str = ""
    is_functional: bool = False
    # The prior-aware token reading of this surface, computed once when it is
    # first seen. A name reads as its name tokens; an address reads through
    # its local part, so `maya.chen@redwood.ai` and "Maya Chen" produce the
    # same tokens and can be compared directly.
    tokens: tuple[str, ...] = ()

    @classmethod
    def build(cls, kind: str, value: str, priors: Priors, count: int = 0) -> "Surface":
        """Create a surface, applying the learned priors once and storing the result.

        The single place a surface is interpreted, so the index and the tests
        cannot drift apart on what a given prior implies.
        """
        s = cls(key=f"{kind}:{value.lower()}", kind=kind, value=value.lower(), count=count, docs=count)
        if kind == "email":
            s.org = priors.org_of(s.domain)
            s.is_functional = priors.is_functional(s.localpart)
            s.tokens = tuple(handle_tokens(s.localpart, priors))
        elif kind == "name":
            s.tokens = tuple(name_tokens(s.value, priors))
        else:
            s.tokens = tuple(handle_tokens(s.value, priors))
        return s

    @property
    def domain(self) -> str:
        return self.value.rpartition("@")[2] if self.kind == "email" else ""

    @property
    def localpart(self) -> str:
        return self.value.partition("@")[0] if self.kind == "email" else ""

    def to_dict(self) -> dict:
        return {
            "surface": self.value,
            "kind": self.kind,
            "occurrences": self.count,
            "docs": self.docs,
            "sources": dict(self.sources.most_common()),
            "orgs": dict(self.orgs.most_common(3)),
        }


@dataclass(slots=True)
class Merge:
    """One accepted union, with the evidence that justified it."""

    a: str
    b: str
    score: float
    signals: list[str]

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "score": round(self.score, 3), "signals": self.signals}


# --- normalization ----------------------------------------------------------


def norm_name(raw: str, priors: Priors | None = None) -> str:
    """Lowercase a human name down to comparable form, dropping role qualifiers.

    Which words count as qualifiers is *learned* (`priors.qualifiers`), not
    listed here, so "Maya Chen (Redwood AE)", "Maya Chen Redwood AE" and
    "Maya Chen" collapse to one string at this company, and whatever the
    equivalent trailing job titles are collapse at the next one.
    """
    s = _PUNCT.sub(" ", _PAREN.sub(" ", raw)).replace("’", "'").strip().lower()
    toks = [t for t in re.sub(r"\s+", " ", s).split(" ") if t]
    if priors is None:
        return " ".join(toks)
    stripped = [t for t in toks if not priors.is_qualifier(t)]
    # Only drop qualifiers when something name-shaped survives — a person
    # genuinely surnamed "Head" should not normalize away to nothing.
    return " ".join(stripped if len(stripped) >= 2 else toks)


def name_tokens(raw: str, priors: Priors | None = None) -> list[str]:
    """Name tokens, with initials kept as single characters."""
    return [t for t in norm_name(raw, priors).split(" ") if t]


def handle_tokens(raw: str, priors: Priors | None = None) -> list[str]:
    """Split a handle on its separators, dropping trailing disambiguation digits."""
    s = _DIGITS.sub("", raw.lower())
    toks = [t for t in _SPLIT.split(s) if t]
    if priors is None:
        return toks
    stripped = [t for t in toks if not priors.is_qualifier(t)]
    return stripped or toks


def is_email(value: str) -> bool:
    return bool(_VALID_EMAIL.match(value))


def looks_like_handle(raw: str) -> bool:
    """A single unspaced token, the shape Slack and @mention syntax produce.

    Must contain a letter. Version strings, ports and IP addresses turn up in
    the same positions as handles — `8.8.8.8` and `512` were both resolving to
    their own "people" — and no account name is purely numeric.
    """
    raw = raw.strip()
    if not raw or " " in raw or len(raw) > 40:
        return False
    return any(c.isalpha() for c in raw)


def looks_like_person_name(raw: str, priors: Priors | None = None) -> bool:
    """A full name: two-plus alphabetic tokens, none of them a stopword."""
    toks = name_tokens(raw, priors)
    if not 2 <= len(toks) <= 4:
        return False
    if any(_NOT_A_PERSON.match(t) for t in toks):
        return False
    # Accented and apostrophed names are people too ("Tomás", "O'Connell");
    # digits and symbols are not.
    return all(t[0].isalpha() and all(c.isalpha() or c in "'-" for c in t) for t in toks)


def compact(toks: Iterable[str]) -> str:
    return "".join(toks)


def initial_key(toks: list[str]) -> str:
    """first-initial + last token: 'maya chen' and 'm chen' agree here."""
    return f"{toks[0][0]}|{toks[-1]}" if len(toks) >= 2 else ""


# --- stage 1: mine ----------------------------------------------------------


class SurfaceIndex:
    """Every identity surface in the corpus, with occurrence counts."""

    def __init__(self, priors: Priors | None = None) -> None:
        self.priors = priors or Priors()
        self.surfaces: dict[str, Surface] = {}
        # Explicit "Name <email>" bindings the corpus states outright, counted.
        self.hard_links: Counter = Counter()
        self.docs_seen = 0
        # Malformed values turned away at the door, counted so a parser
        # regression shows up as a number rather than as silent data loss.
        self.rejected: Counter = Counter()
        # Which names each address was ever stated to belong to.
        self.binding_names: dict[str, set[str]] = defaultdict(set)

    def _bump(self, kind: str, value: str, source: str, org: str = "") -> str | None:
        value = value.strip()
        if not value:
            return None
        if kind == "email" and not is_email(value):
            self.rejected[kind] += 1
            return None
        if kind == "handle" and not looks_like_handle(value):
            self.rejected[kind] += 1
            return None
        key = f"{kind}:{value.lower()}"
        s = self.surfaces.get(key)
        if s is None:
            s = self.surfaces[key] = Surface.build(kind, value, self.priors)
        s.count += 1
        s.docs += 1
        s.sources[source] += 1
        if org:
            s.orgs[org] += 1
        return key

    def add_document(self, doc: dict) -> None:
        self.docs_seen += 1
        source = doc.get("source", "")

        for email in doc.get("emails") or ():
            self._bump("email", email, source)

        for pair in doc.get("named_emails") or ():
            name, email = pair.get("name", ""), pair.get("email", "")
            if not looks_like_person_name(name, self.priors):
                continue
            nk = self._bump("name", norm_name(name, self.priors), source)
            ek = self._bump("email", email, source)
            if nk and ek:
                self.hard_links[(nk, ek)] += 1
                self.binding_names[ek].add(nk)

        for speaker in doc.get("speakers") or ():
            # A speaker line that is neither a person nor handle-shaped is a
            # transcript artifact ("Meeting Header", "Action Items") and must
            # not become a candidate identity.
            if looks_like_person_name(speaker, self.priors):
                self._bump("name", norm_name(speaker, self.priors), source)
            else:
                self._bump("handle", speaker, source)

        for mention in doc.get("mentions") or ():
            self._bump("handle", mention, source)

        for att in doc.get("attendees") or ():
            name = att.get("name", "")
            if looks_like_person_name(name, self.priors):
                self._bump("name", norm_name(name, self.priors), source, org=att.get("org", ""))

    def shared_emails(self) -> set[str]:
        """Addresses that cannot stand for one person.

        Two ways to earn the label. The first is by name: `compliance@` is a
        department. The second is by behaviour, and matters far more — an
        address the corpus binds to several *different* people is a shared
        mailbox no matter what it is called, and `procurement@harbortech.com`
        signed by both Susan Park and Susan Lee proves it of itself.

        Getting this wrong is not a small error. Treating shared mailboxes as
        personal let stated bindings chain unrelated people together through
        them, and produced a single cluster holding 102,564 aliases.
        """
        return {
            k
            for k, s in self.surfaces.items()
            if s.kind == "email" and (s.is_functional or len(self.binding_names.get(k, ())) > 1)
        }

    def exclusive_links(self) -> Counter:
        """Stated bindings that survive the shared-mailbox test."""
        shared = self.shared_emails()
        return Counter(
            {pair: n for pair, n in self.hard_links.items() if pair[1] not in shared}
        )

    def working_set(self, min_occurrences: int = MIN_OCCURRENCES) -> dict[str, Surface]:
        """Surfaces worth resolving into people.

        Kept: anything seen often enough to carry signal, plus anything named
        in an exclusive stated binding regardless of frequency — the corpus
        said outright who it is, so rarity is not a reason to doubt it.

        Dropped: shared mailboxes. They are real and worth modelling later as
        their own node type, but they are not people and must not sit in the
        middle of a person graph acting as a bridge between them.
        """
        shared = self.shared_emails()
        keep = {k for pair in self.exclusive_links() for k in pair}
        return {
            k: s
            for k, s in self.surfaces.items()
            if k not in shared and (s.count >= min_occurrences or k in keep)
        }


def load_index(shards: Iterable, priors: Priors | None = None, on_progress=None) -> SurfaceIndex:
    """Stream normalized JSONL shards into a SurfaceIndex built with learned priors."""
    index = SurfaceIndex(priors)
    for path in shards:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                index.add_document(json.loads(line))
                if on_progress and index.docs_seen % 50_000 == 0:
                    on_progress(index.docs_seen)
    return index


# --- stage 2: block ---------------------------------------------------------


def block_keys(s: Surface) -> set[str]:
    """Blocking keys for one surface.

    Two surfaces are only ever compared if they share a key, so these must be
    generous enough that true pairs collide and narrow enough that the blocks
    stay small. Name-shaped keys do both: they collapse `maya.chen`,
    `Maya Chen` and `mchen` onto overlapping keys without pulling in the rest
    of the corpus.
    """
    keys: set[str] = set()
    toks = (
        list(s.tokens)
    )
    if not toks:
        # Nothing name-shaped survives — an all-digits or all-role-word handle.
        # It gets no keys at all rather than an empty one, which would
        # otherwise collect every such surface into one enormous block.
        return keys
    keys.add(f"local:{compact(toks)}")
    keys.add(f"compact:{compact(toks)}")
    keys.add(f"first:{toks[0]}")
    keys.add(f"last:{toks[-1]}")
    if len(toks) >= 2:
        keys.add(f"init:{initial_key(toks)}")
    elif len(toks) == 1 and len(toks[0]) > 2:
        # A single-token handle like "mchen" may be an initial+surname jam;
        # offer that reading as a key so it can meet "Maya Chen".
        keys.add(f"init:{toks[0][0]}|{toks[0][1:]}")
    return keys


def build_blocks(surfaces: dict[str, Surface]) -> tuple[dict[str, list[str]], list[tuple[str, int]]]:
    """Group surface keys by blocking key. Returns (blocks, oversized_dropped)."""
    blocks: dict[str, list[str]] = defaultdict(list)
    for key, s in surfaces.items():
        for bk in block_keys(s):
            blocks[bk].append(key)
    dropped = [(bk, len(m)) for bk, m in blocks.items() if len(m) > MAX_BLOCK]
    kept = {bk: m for bk, m in blocks.items() if 2 <= len(m) <= MAX_BLOCK}
    return kept, sorted(dropped, key=lambda x: -x[1])


def candidate_pairs(blocks: dict[str, list[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for members in blocks.values():
        members = sorted(set(members))
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                pairs.add((a, b))
    return pairs


# --- stage 3: score ---------------------------------------------------------

# Each signal is a distinct claim about why two surfaces are the same person.
# Weights are ordered by how hard the evidence is: an explicit binding stated
# by the corpus outranks any convention we infer.
WEIGHTS = {
    "STATED_BINDING": 0.99,  # "Maya Chen <maya.chen@redwood.com>" appears verbatim
    "LOCAL_EQUALS_NAME": 0.95,  # maya.chen@ ↔ Maya Chen
    "HANDLE_EQUALS_LOCAL": 0.90,  # @maya.chen ↔ maya.chen@
    "HANDLE_EQUALS_NAME": 0.88,  # @mayachen ↔ Maya Chen
    "NAME_EQUALS_NAME": 0.88,  # Maya Chen ↔ maya chen (different casing/source)
    "INITIAL_SURNAME": 0.72,  # @mchen ↔ Maya Chen
    "ABBREVIATED_NAME": 0.72,  # "M. Chen" ↔ "Maya Chen"
    # @soham ↔ Soham Ratnaparkhi, when no other Soham exists anywhere in the
    # corpus. Sits above the merge threshold on its own: this is the track's
    # own worked example, and the uniqueness test in front of it is strict.
    "UNIQUE_GIVEN_NAME": 0.75,
}


def _full_tokens(s: Surface) -> list[str]:
    """The two-plus token reading of a surface, or [] when it has none.

    Names read as names; an email reads through its local part, so
    `maya.chen@redwood.ai` and "Maya Chen" produce the same token list.
    """
    return list(s.tokens) if len(s.tokens) >= 2 else []


def identity_form(s: Surface) -> str:
    """The compacted full-name reading of a surface: both spellings of Maya Chen
    — the name and `maya.chen@` — reduce to "mayachen".

    Ambiguity has to be counted over identities rather than over surfaces. A
    person and their own email address are not two candidates competing for an
    abbreviation; they are one person written twice, and counting them as two
    would make every well-evidenced name look contested.
    """
    return compact(_full_tokens(s))


def _sole_owner(pairs: Iterable[tuple[str, str]]) -> dict[str, str | None]:
    """Collapse (key, identity) pairs to a sole identity per key, None if contested."""
    owners: dict[str, set[str]] = defaultdict(set)
    for key, owner in pairs:
        owners[key].add(owner)
    return {k: (next(iter(v)) if len(v) == 1 else None) for k, v in owners.items()}


class Scorer:
    """Turns a candidate pair into a confidence and the signals behind it.

    Two ambiguity maps do most of the precision work. Both answer the same
    question — *is this abbreviation unique?* — because an abbreviation that
    fits two people is evidence for neither:

    - `given_name_owner`: `@maya` is Maya Chen only if she is the only Maya.
      This corpus has eight distinct Priyas, so `@priya` resolves to nobody.
    - `initial_owner`: `@jae` is Jordan Ae only if no other J-surname-Ae
      exists. It did not: Jamie Ae did too, and without this guard both
      collapsed into one person through the shared handle.
    """

    def __init__(self, surfaces: dict[str, Surface], hard_links: Counter) -> None:
        self.surfaces = surfaces
        self.hard_links = hard_links
        self.given_name_owner = _sole_owner(
            (s.tokens[0], identity_form(s))
            for s in surfaces.values()
            if s.kind == "name" and len(s.tokens) >= 2
        )
        self.initial_owner = _sole_owner(
            (initial_key(toks), compact(toks))
            for s in surfaces.values()
            if (toks := _full_tokens(s))
        )

    def score(self, ka: str, kb: str) -> tuple[float, list[str]]:
        """Combined confidence and the list of signals that fired.

        Signals combine as independent evidence — the complement of all of them
        being wrong at once — so two medium signals can clear the bar that
        neither clears alone, without any single weak one doing it by itself.
        """
        a, b = self.surfaces[ka], self.surfaces[kb]
        signals = [s for s in self._signals(ka, kb, a, b)]
        if not signals:
            return 0.0, []
        residual = 1.0
        for sig in signals:
            residual *= 1.0 - WEIGHTS[sig]
        return 1.0 - residual, signals

    def _signals(self, ka: str, kb: str, a: Surface, b: Surface) -> Iterator[str]:
        kinds = {a.kind, b.kind}

        if self.hard_links.get((ka, kb)) or self.hard_links.get((kb, ka)):
            yield "STATED_BINDING"

        if kinds == {"email", "name"}:
            email, name = (a, b) if a.kind == "email" else (b, a)
            etoks, ntoks = list(email.tokens), list(name.tokens)
            if compact(etoks) == compact(ntoks) and len(ntoks) >= 2:
                yield "LOCAL_EQUALS_NAME"
            elif self._unique_initial(etoks, name):
                yield "INITIAL_SURNAME"
            elif len(etoks) == 1 and self._unique_given(etoks[0], name):
                yield "UNIQUE_GIVEN_NAME"

        elif kinds == {"handle", "email"}:
            handle, email = (a, b) if a.kind == "handle" else (b, a)
            htoks, etoks = list(handle.tokens), list(email.tokens)
            if compact(htoks) == compact(etoks):
                yield "HANDLE_EQUALS_LOCAL"
            elif self._unique_initial(htoks, email):
                yield "INITIAL_SURNAME"

        elif kinds == {"handle", "name"}:
            handle, name = (a, b) if a.kind == "handle" else (b, a)
            htoks, ntoks = list(handle.tokens), list(name.tokens)
            if len(ntoks) < 2:
                return
            if compact(htoks) == compact(ntoks):
                yield "HANDLE_EQUALS_NAME"
            elif self._unique_initial(htoks, name):
                yield "INITIAL_SURNAME"
            elif len(htoks) == 1 and self._unique_given(htoks[0], name):
                yield "UNIQUE_GIVEN_NAME"

        elif kinds == {"name"}:
            ta, tb = list(a.tokens), list(b.tokens)
            if ta == tb:
                yield "NAME_EQUALS_NAME"
            elif self._abbreviates(ta, tb) and self._unique_initial(ta, b):
                yield "ABBREVIATED_NAME"
            elif self._abbreviates(tb, ta) and self._unique_initial(tb, a):
                yield "ABBREVIATED_NAME"

    def _unique_given(self, token: str, name: Surface) -> bool:
        """True when `token` is a given name owned by exactly this one person."""
        return self.given_name_owner.get(token) == identity_form(name)

    def _unique_initial(self, abbrev: list[str], full: Surface) -> bool:
        """True when an initial+surname reading points at `full` and nothing else.

        The abbreviation may arrive already split (`m.chen`) or jammed together
        (`mchen`); both are tried, and both must land on a key that only `full`
        owns. Where two people share it, no signal fires and the pair is left
        for a human or an LLM to adjudicate.
        """
        keys = {initial_key(abbrev)} if len(abbrev) >= 2 else set()
        if len(abbrev) == 1 and len(abbrev[0]) > 2:
            keys.add(f"{abbrev[0][0]}|{abbrev[0][1:]}")
        toks = _full_tokens(full)
        target = initial_key(toks)
        return (
            bool(target)
            and target in keys
            and self.initial_owner.get(target) == compact(toks)
        )

    @staticmethod
    def _abbreviates(short: list[str], full: list[str]) -> bool:
        """'m chen' abbreviates 'maya chen': same surname, given name initialed."""
        if len(short) < 2 or len(full) < 2 or short[-1] != full[-1]:
            return False
        return len(short[0]) == 1 and full[0].startswith(short[0]) and len(full[0]) > 1


# --- stage 4: cluster -------------------------------------------------------


class Clusters:
    """Union-find over surfaces, refusing merges that break a hard constraint."""

    def __init__(self, keys: Iterable[str], surfaces: dict[str, Surface]) -> None:
        self.parent = {k: k for k in keys}
        self.surfaces = surfaces
        self.merges: dict[str, list[Merge]] = defaultdict(list)
        self.rejected: list[tuple[str, str, str]] = []
        # Mailbox names and employers held by each cluster, for the guards below.
        self.mailboxes: dict[str, set[str]] = {
            k: ({self._mailbox(surfaces[k])} if surfaces[k].kind == "email" else set())
            for k in self.parent
        }
        self.orgs: dict[str, set[str]] = {
            k: ({surfaces[k].org} if surfaces[k].org else set()) for k in self.parent
        }

    @staticmethod
    def _mailbox(s: Surface) -> str:
        """The local part, normalized past plus-tags.

        Deliberately domain-blind: this company writes the same people as
        @redwood.com, @redwood.ai and @redwoodinference.com, so keying the
        constraint on the domain would have caught almost nothing. What is
        actually one-per-person is the mailbox name.
        """
        return _DIGITS.sub("", s.localpart.partition("+")[0]).strip(".")

    def find(self, k: str) -> str:
        root = k
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[k] != root:  # path compression
            self.parent[k], k = root, self.parent[k]
        return root

    def union(self, merge: Merge) -> bool:
        ra, rb = self.find(merge.a), self.find(merge.b)
        if ra == rb:
            return False

        # One person, one employer. A shared full name is not a shared person:
        # Susan Park at Acme and Susan Park at Novushealth are two people, and
        # the domain is the only thing that says so. This holds even against a
        # stated binding, because the binding is evidence about one address,
        # never about the two addresses now being asked to coexist.
        orgs = self.orgs[ra] | self.orgs[rb]
        if len(orgs) > 1:
            self.rejected.append((merge.a, merge.b, f"two organizations: {sorted(orgs)}"))
            return False

        # One person, one mailbox name. Two distinct mailboxes in a single
        # cluster means the chain of inferences went wrong somewhere, so the
        # weakest link — this merge, since merges are applied best-first — is
        # refused. A stated binding is allowed through here, and only here: the
        # corpus asserted this address belongs to this person, and it has
        # already been checked for exclusivity upstream.
        mailboxes = self.mailboxes[ra] | self.mailboxes[rb]
        if len(mailboxes) > 1 and "STATED_BINDING" not in merge.signals:
            self.rejected.append((merge.a, merge.b, "two distinct mailboxes"))
            return False

        self.parent[rb] = ra
        self.mailboxes[ra] = mailboxes
        self.orgs[ra] = orgs
        self.merges[ra].extend(self.merges.pop(rb, []))
        self.merges[ra].append(merge)
        return True

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for k in self.parent:
            out[self.find(k)].append(k)
        return out


@dataclass(slots=True)
class Entity:
    eid: str
    canonical_name: str
    type: str
    confidence: float
    aliases: list[Surface]
    merges: list[Merge]

    @property
    def occurrences(self) -> int:
        return sum(a.count for a in self.aliases)

    def to_dict(self) -> dict:
        return {
            "eid": self.eid,
            "canonical_name": self.canonical_name,
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "occurrences": self.occurrences,
            "alias_count": len(self.aliases),
            "aliases": [a.to_dict() for a in sorted(self.aliases, key=lambda s: -s.count)],
            "merges": [m.to_dict() for m in sorted(self.merges, key=lambda m: -m.score)],
            "sources": dict(sum((a.sources for a in self.aliases), Counter()).most_common()),
        }


def _canonical_name(aliases: list[Surface]) -> str:
    """Best display name for a cluster.

    A real full name wins over an address, which wins over a bare handle -
    "Maya Chen" is what a judge should see, not "mchen". Among names, the most
    frequently written one wins.
    """
    names = [a for a in aliases if a.kind == "name" and len(a.tokens) >= 2]
    if names:
        best = max(names, key=lambda s: (s.count, len(s.value)))
        return best.value.title()
    emails = [a for a in aliases if a.kind == "email"]
    if emails:
        best = max(emails, key=lambda s: s.count)
        return " ".join(t.capitalize() for t in best.tokens) or best.localpart
    return max(aliases, key=lambda s: s.count).value


def resolve(
    surfaces: dict[str, Surface],
    hard_links: Counter,
    threshold: float = MERGE_THRESHOLD,
) -> tuple[list[Entity], dict]:
    """Run blocking, scoring and clustering. Returns (entities, diagnostics)."""
    blocks, dropped = build_blocks(surfaces)
    pairs = candidate_pairs(blocks)

    # Blocking exists to avoid comparing everything to everything, but a stated
    # binding is not an inference that needs a candidate — the corpus wrote
    # `Maya Chen <mc@redwood.ai>` down. Nothing about "mc" collides with "maya
    # chen" under any name-shaped key, so without this the strongest evidence
    # available would be the evidence most likely to be missed.
    pairs |= {
        (a, b) if a < b else (b, a)
        for a, b in hard_links
        if a in surfaces and b in surfaces and a != b
    }

    scorer = Scorer(surfaces, hard_links)
    scored: list[Merge] = []
    for ka, kb in pairs:
        score, signals = scorer.score(ka, kb)
        if score >= threshold:
            scored.append(Merge(a=ka, b=kb, score=score, signals=signals))

    # Strongest evidence first, so that when the corporate-address constraint
    # has to refuse something, it refuses the weakest inference rather than
    # whichever one happened to be considered first.
    scored.sort(key=lambda m: -m.score)

    clusters = Clusters(surfaces.keys(), surfaces)
    applied = sum(1 for m in scored if clusters.union(m))

    entities: list[Entity] = []
    for i, (root, members) in enumerate(
        sorted(clusters.groups().items(), key=lambda kv: -sum(surfaces[k].count for k in kv[1]))
    ):
        alias_list = [surfaces[k] for k in members]
        merges = clusters.merges.get(root, [])
        # A singleton is certain by default; a cluster is only as good as its
        # weakest accepted merge, which is the link most likely to be wrong.
        confidence = min((m.score for m in merges), default=1.0)
        entities.append(
            Entity(
                eid=f"ent_{i:06d}",
                canonical_name=_canonical_name(alias_list),
                type="Person",
                confidence=confidence,
                aliases=alias_list,
                merges=merges,
            )
        )

    diagnostics = {
        "surfaces": len(surfaces),
        "blocks": len(blocks),
        "blocks_dropped_oversized": len(dropped),
        "largest_dropped_blocks": dropped[:10],
        "candidate_pairs": len(pairs),
        "pairs_above_threshold": len(scored),
        "merges_applied": applied,
        "merges_refused_by_constraint": len(clusters.rejected),
        "entities": len(entities),
        "multi_alias_entities": sum(1 for e in entities if len(e.aliases) > 1),
        "signal_counts": dict(
            Counter(s for m in scored for s in m.signals).most_common()
        ),
        "threshold": threshold,
    }
    return entities, diagnostics
