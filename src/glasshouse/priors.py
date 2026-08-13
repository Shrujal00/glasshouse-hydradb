"""Corpus priors, learned rather than written down.

Entity resolution needs to know three things about a company before it can
resolve anyone in it:

1. Which email domains are the employer's, so that one person writing from
   two of them is still one person.
2. Which words in the name slot are job titles rather than surnames, so that
   "Jordan AE" and "Jamie AE" are two people rather than one family.
3. Which mailboxes belong to a department rather than a person, so that
   `procurement@` does not become a bridge joining everyone who ever signed
   from it.

Every one of those is company-specific, and the obvious way to supply them is
a list in the source: `HOME_DOMAIN = "redwood"`, a set of job titles, a set of
role mailbox names. That works exactly once. Point the same code at a second
company and all three are wrong, silently — and a resolver that only works on
the corpus it was written against has not solved entity resolution, it has
memorised an answer.

So all three are derived here from the corpus's own statistics. The signals
are deliberately structural, not lexical: none of them ask what a word means,
only how it is distributed.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

# Labels that name a mail host rather than the company that owns it. This is
# infrastructure vocabulary, not company vocabulary - it is the same list for
# every corpus on earth, which is why it is the one thing still written down.
_MAIL_HOST = frozenset("mail smtp email mx www m".split())

_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold(token: str) -> str:
    """Strip accents, so a name and the address built from it compare equal.

    Mail systems ASCII-fold: Anna Müller writes from `anna.mueller@` or
    `anna.muller@`, never `müller@`. Comparing the raw forms therefore finds
    a surname that never appears in any mailbox — which is the signature this
    module uses to identify job titles, so `müller` was classified as one and
    stripped out of her name.
    """
    decomposed = unicodedata.normalize("NFKD", token)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    # German folds ü->ue rather than ->u; accept both by also indexing the
    # bare form, which `_qualifiers` compares against.
    return stripped.replace("ß", "ss").lower()


def domain_label(domain: str) -> str:
    """The leading meaningful label of a domain.

    Read from the left rather than the right. Stripping TLDs from the right
    requires knowing every TLD present, and missing one is silent: with `.it`
    absent from such a list, `salushealth.it` and `novamed.it` both reduce to
    the "organisation" `it` and two unrelated people merge.
    """
    labels = [p for p in domain.lower().split(".") if p and p not in _MAIL_HOST]
    return labels[0] if labels else domain.lower()


@dataclass(slots=True)
class Priors:
    """What was learned about this particular company."""

    home_labels: frozenset[str] = frozenset()
    qualifiers: frozenset[str] = frozenset()
    functional: frozenset[str] = frozenset()
    evidence: dict = field(default_factory=dict)

    def org_of(self, domain: str) -> str:
        """The employer a domain belongs to, collapsing the home org's spellings."""
        label = domain_label(domain)
        return "\x00home" if label in self.home_labels else label

    def is_qualifier(self, token: str) -> bool:
        return token in self.qualifiers

    def is_functional(self, localpart: str) -> bool:
        return localpart.lower() in self.functional

    def to_dict(self) -> dict:
        return {
            "home_labels": sorted(self.home_labels),
            "qualifiers": sorted(self.qualifiers),
            "functional": sorted(self.functional),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Priors":
        return cls(
            home_labels=frozenset(raw.get("home_labels") or ()),
            qualifiers=frozenset(raw.get("qualifiers") or ()),
            functional=frozenset(raw.get("functional") or ()),
            evidence=raw.get("evidence") or {},
        )


def _home_labels(
    domains: Counter,
    localparts_by_label: dict[str, set[str]],
    distinctive: set[str],
    min_overlap: float = 0.25,
    min_shared: int = 3,
) -> tuple[frozenset[str], list]:
    """Find the employer's domain labels.

    The employer is whoever the most distinct people write from — counting
    mailboxes rather than messages, so one automated sender with a hundred
    thousand emails cannot outvote a real department.

    A company usually owns several spellings (`redwood.com`, `redwood.ai`,
    `redwood-inference.com`). The tempting way to recover them is by name:
    any label starting with the winner. That is wrong, and measurably so —
    on this corpus it claimed `redwoodbank`, `redwoodhealth` and
    `redwood-customer`, which are a bank, a hospital and a customer tenant,
    plus `red` because the winner starts with it.

    So the family is found by *who writes from it* instead. Two labels belong
    to one employer when the same mailboxes appear on both: Maya Chen writes
    from redwood.com and redwood.ai, but nobody at Redwood Bank does. Names
    can coincide; staff lists cannot.
    """
    if not localparts_by_label:
        return frozenset(), []
    ranked = sorted(localparts_by_label.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    winner = ranked[0][0]

    # Only *distinctive* mailboxes vote. `info@`, `security@` and `ops@` exist
    # at every company in the corpus, so counting them made any domain with a
    # single generic address look like a wholly-owned subsidiary — the first
    # attempt returned nine thousand "employer" domains including `acme-air`
    # and `1password`.
    home = localparts_by_label[winner] & distinctive
    family = {winner}
    for label, mailboxes in localparts_by_label.items():
        if label == winner:
            continue
        own = mailboxes & distinctive
        shared = len(own & home)
        # Absolute evidence as well as proportional: three people in common,
        # not one coincidence expressed as a percentage.
        if shared >= min_shared and shared / min(len(own), len(home)) >= min_overlap:
            family.add(label)

    top = [(label, len(mailboxes), domains[label]) for label, mailboxes in ranked[:8]]
    return frozenset(family), top


def _qualifiers(
    name_pairs: Counter,
    localpart_tokens: Counter,
    min_partners: int,
) -> tuple[frozenset[str], list]:
    """Find tokens that are job titles rather than surnames.

    Two structural signals, and neither needs to know what the word means:

    **Promiscuity.** A surname is selective — knowing it narrows who you are
    talking about, so it follows only a handful of given names. A job title is
    not: every account executive in the company is "<given name> AE", so the
    token follows dozens of different first names.

    **Absence from mailboxes.** People's addresses are built from their names,
    so a real surname turns up inside email local parts — `chen` appears in
    `maya.chen@`. Job titles do not: nobody's address is `maya.ae@`.

    A token needs both to be called a qualifier. Promiscuity alone would
    condemn genuinely common surnames; the mailbox test rescues them, because
    a common surname is common in addresses too.
    """
    partners: dict[str, set[str]] = defaultdict(set)
    for (before, token), _ in name_pairs.items():
        partners[token].add(before)

    flagged, rows = set(), []
    for token, befores in partners.items():
        n_partners = len(befores)
        if n_partners < min_partners:
            continue
        # Compared accent-folded: mail systems ASCII-fold names, so `müller`
        # would otherwise appear in zero mailboxes and be mistaken for a title.
        folded = fold(token)
        in_mailboxes = max(
            localpart_tokens.get(token, 0),
            localpart_tokens.get(folded, 0),
            localpart_tokens.get(folded.replace("u", "ue"), 0),
        )
        # A surname that follows N distinct given names should appear in a
        # comparable number of addresses. Well under that, and the token is
        # not naming a family.
        if in_mailboxes * 4 < n_partners:
            flagged.add(token)
            rows.append((token, n_partners, in_mailboxes))
    rows.sort(key=lambda r: -r[1])
    return frozenset(flagged), rows[:25]


def _names_someone(token: str, person_tokens: set[str]) -> bool:
    """Whether a mailbox token is built from somebody's name.

    Beyond the obvious match, the common compressed convention counts too:
    `mchen` is Maya Chen's mailbox, and reading it as an initial followed by a
    known surname is what stops it being filed as a department.
    """
    if token in person_tokens:
        return True
    return len(token) > 2 and token[1:] in person_tokens


def _functional(
    localpart_orgs: dict[str, set[str]],
    person_tokens: set[str],
    min_orgs: int = 3,
) -> tuple[frozenset[str], list]:
    """Find departmental mailboxes.

    A departmental name recurs across companies: `events@`, `compliance@` and
    `procurement@` exist everywhere. That alone looked like enough, and it is
    not — it flagged `elena`, `samir` and `priya`, because common given names
    also recur across companies. Thirteen thousand "departments", most of them
    people.

    The second test is what separates them, and it is free: a given name turns
    up *inside somebody's full name* somewhere in the corpus. `elena` appears
    in "Elena Rossi"; `procurement` appears in nobody's name. So a local part
    is departmental when it recurs across organisations **and** never occurs
    as part of a person's name.
    """
    flagged, rows = set(), []
    for localpart, orgs in localpart_orgs.items():
        if len(orgs) < min_orgs:
            continue
        # Compare the local part's *tokens*, not the whole string. Matching
        # `samir.patel` against a set of single name words never hits, so
        # seven thousand people — every common name that happens to exist at
        # three companies — were classified as departments and dropped out of
        # resolution entirely.
        tokens = [t for t in _TOKEN.findall(localpart) if t]
        if tokens and all(_names_someone(t, person_tokens) for t in tokens):
            continue
        flagged.add(localpart)
        rows.append((localpart, len(orgs)))
    rows.sort(key=lambda r: -r[1])
    return frozenset(flagged), rows[:25]


def derive(
    emails: Iterable[str],
    raw_names: Iterable[tuple[str, int]],
    min_partners: int = 8,
) -> Priors:
    """Learn the priors from observed emails and raw (unstripped) names.

    `raw_names` must be *unnormalised* — qualifier stripping is what we are
    trying to learn, so it cannot already have happened.
    """
    domains: Counter = Counter()
    localparts_by_label: dict[str, set[str]] = defaultdict(set)
    localpart_orgs: dict[str, set[str]] = defaultdict(set)
    localpart_tokens: Counter = Counter()

    for address in emails:
        local, _, domain = address.lower().partition("@")
        if not domain or not local:
            continue
        label = domain_label(domain)
        domains[label] += 1
        localparts_by_label[label].add(local)
        localpart_orgs[local].add(label)
        for token in _TOKEN.findall(local):
            localpart_tokens[token] += 1

    # A mailbox name seen at only a handful of organisations identifies a
    # person; one seen everywhere is a role. Computed before the employer is
    # known, so it can be used to find them.
    distinctive = {lp for lp, orgs in localpart_orgs.items() if len(orgs) <= 2}

    home, home_rows = _home_labels(domains, localparts_by_label, distinctive)

    # Collapse the employer's several domain spellings before counting how
    # many *organisations* a mailbox appears at, or every internal address
    # would look departmental for spanning redwood.com and redwood.ai.
    collapsed: dict[str, set[str]] = {
        lp: {("\x00home" if o in home else o) for o in orgs}
        for lp, orgs in localpart_orgs.items()
    }

    name_pairs: Counter = Counter()
    person_tokens: set[str] = set()
    for raw, count in raw_names:
        # Apostrophes are dropped rather than treated as separators: mail
        # systems write Liam O'Neill as `liam.oneill`, so splitting the name
        # into "o" and "neill" leaves nothing that matches his own address.
        tokens = [t for t in _TOKEN.findall(raw.lower().replace("'", "").replace("’", "")) if t]
        if len(tokens) >= 2:
            # Every word that has ever been part of somebody's full name.
            # This is what tells `elena` (a person) from `procurement` (a
            # department) when both appear as mailboxes at many companies.
            person_tokens.update(tokens)
            person_tokens.update(fold(t) for t in tokens)
        for before, token in zip(tokens, tokens[1:]):
            name_pairs[(before, token)] += count

    qualifiers, qual_rows = _qualifiers(name_pairs, localpart_tokens, min_partners)
    # A qualifier is a job title, not a person's name, so it must not also
    # protect a mailbox from being called departmental.
    functional, func_rows = _functional(collapsed, person_tokens - qualifiers)

    return Priors(
        home_labels=home,
        qualifiers=qualifiers,
        functional=functional,
        evidence={
            "home_org_candidates": home_rows,
            "home_labels": sorted(home),
            "qualifier_tokens": qual_rows,
            "qualifier_count": len(qualifiers),
            "functional_localparts": func_rows,
            "functional_count": len(functional),
            "distinct_domain_labels": len(domains),
        },
    )
