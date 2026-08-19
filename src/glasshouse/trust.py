"""Choosing between claims that contradict each other, deterministically.

Nine tools disagree. A Confluence page says a limit is 20%, a Slack thread four
months later says 30%, and both are in the same six documents synthesis reads.
The model cannot break that tie -- it has no notion of which source is a
reviewed artefact, which assertion is stale, or how many documents back a
value. So the tie is broken here, in code, before the model sees anything, and
the model is handed the decision along with the value it lost to.

Everything in this module is arithmetic on the claim records. There is no
second model call: an adjudication that changed its mind between two runs of
the same question would be worse than no adjudication at all, because the
rationale shown in the UI has to be the reason the answer actually came out
that way.

The one thing arbitration is allowed to do instead of deciding is refuse. Two
hedged Slack messages disagreeing about a threshold is not evidence for either
value, and crowning the marginally higher-scoring one would manufacture a
confident answer out of nothing -- exactly the failure the abstention path
elsewhere in the system exists to avoid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence

from .claims import Claim

# Source authority. A Confluence page and a merged GitHub PR have been through
# review by someone other than their author; a Jira or Linear ticket is a
# tracked record with a lifecycle; a Slack message is one person talking in the
# present tense. The band is narrow on purpose -- Slack is where this corpus
# says most true things first, so authority is a tiebreaker between comparable
# claims, not a licence to ignore chat.
AUTHORITY = {
    "confluence": 1.00,
    "github": 0.98,
    "jira": 0.94,
    "linear": 0.94,
    "google_drive": 0.90,
    "hubspot": 0.84,
    "gmail": 0.80,
    "fireflies": 0.74,
    "slack": 0.70,
}
DEFAULT_AUTHORITY = 0.75

# How much staleness costs, per predicate. A `status` or a `due_date` is a
# value that is *supposed* to change: the older assertion was true when it was
# written and is simply no longer the answer, so the newest claim in a group
# keeps its full trust and the oldest loses nearly half. `reports_to` is
# structural -- a newer mention of a reporting line is not evidence that it
# changed, so recency is given no weight at all and cannot be named as a reason.
# `owner` and `limit` sit between: they do change, but slowly enough that one
# chat message should not overturn a reviewed page on date alone.
STALENESS = {
    "status": 0.45,
    "due_date": 0.45,
    "limit": 0.20,
    "owner": 0.20,
    "reports_to": 0.00,
}
DEFAULT_STALENESS = 0.20

# What agreement is worth. Two documents stating the same value is the single
# most reliable signal available here, but it saturates fast: the third
# document repeating a figure from the first two adds almost nothing, and
# uncapped corroboration would let a quoted-everywhere wrong number win.
CORROBORATION_STEP = 0.15
CORROBORATION_CAP = 1.30

# Hedged values are worth less than asserted ones. "probably 20%" and "owner:
# TBD" are real sentences in this corpus and they are not claims about the
# world, they are claims about someone's uncertainty. `draft`, `pending` and
# `proposed` are deliberately absent: they read as hedges in English but they
# are legitimate values of `status`, and penalising them would make every
# ticket that is genuinely in draft lose to one that is not.
_HEDGES = re.compile(
    r"\b(tbd|tba|probably|maybe|perhaps|possibly|unclear|unknown|approx\w*|around|"
    r"roughly|about|likely|or)\b|[~?]"
)
HEDGED = 0.65

# How confident extraction was, folded in so that a claim read between the
# lines cannot outrank a plainly asserted one from the same source. Kept as a
# floor plus a slope rather than a raw multiplier: a 0.4-confidence claim is
# weaker evidence, not 40% of a claim.
CONFIDENCE_FLOOR = 0.60

# Below this, a contested value is not worth asserting. Two claims both under
# the floor means the corpus disagrees weakly with itself.
TRUST_FLOOR = 0.50

# And two claims within this of each other are a tie, whatever the arithmetic
# says. The third decimal place of a trust score is not a reason to prefer one
# document over another, and presenting it as one is how a system starts
# sounding certain about coin flips.
MARGIN = 0.06

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# What `_rationale` returns when nothing it can name separated the two claims.
# Compared against rather than matched on text elsewhere, so the sentence can
# be reworded without silently turning refusals back into verdicts.
NO_SIGNAL = "no signal separated these claims"


@dataclass(frozen=True)
class Conflict:
    subject: str
    predicate: str
    winner: Claim
    losers: tuple[Claim, ...]
    rationale: str
    # The work item both sides were talking about, empty when the question was
    # the scope. Two claims are only allowed to contradict each other inside
    # one scope: `owner` of "the API" in one ticket and `owner` of "the API" in
    # an unrelated one are two facts, and calling them a disagreement would be
    # the system inventing a conflict rather than finding one.
    scope: str = ""

    @property
    def decided(self) -> bool:
        """Whether arbitration actually chose, as opposed to declining to."""
        return self.winner.status == "accepted"


@dataclass(frozen=True)
class Arbitration:
    claims: tuple[Claim, ...]
    conflicts: tuple[Conflict, ...]

    @property
    def abstains(self) -> bool:
        return any(not conflict.decided for conflict in self.conflicts)

    def render(self) -> str:
        """What synthesis is handed: both values, where each came from, which
        one won, and why. The rejected value is shown deliberately -- an answer
        that names the accepted figure without acknowledging the disagreement
        reads as ignorance of the other document rather than a judgement about
        it."""
        if not self.claims:
            return ""
        lines = ["Claims extracted from the evidence above, already arbitrated:"]
        contested = {c.claim_id for conflict in self.conflicts
                     for c in (conflict.winner, *conflict.losers)}
        # Agreeing claims are collapsed to one line. Printing the same value
        # once per document reads as several separate facts and spends the
        # model's attention on the repetition rather than on the agreement.
        agreed: dict[tuple[str, str, str], list[Claim]] = {}
        for claim in self.claims:
            if claim.claim_id in contested:
                continue
            agreed.setdefault((_subject(claim), claim.predicate, _value(claim)), []).append(claim)
        for holders in agreed.values():
            lines.append(f"  {_describe(holders[0])}")
            if len(holders) > 1:
                where = ", ".join(sorted({other.source for other in holders[1:]}))
                lines.append(
                    f"      corroborated by {len(holders)} documents (also {where})"
                )
        for conflict in self.conflicts:
            lines.append(
                f"\nDISAGREEMENT — {conflict.predicate} of {conflict.winner.subject}, "
                f"{1 + len(conflict.losers)} competing claims:"
            )
            if conflict.decided:
                lines.append(f"  ACCEPTED   {_describe(conflict.winner)}")
            else:
                lines.append(
                    f"  NO ACCEPTED VALUE — {conflict.rationale}\n"
                    f"  candidate  {_describe(conflict.winner)}"
                )
            for loser in conflict.losers:
                lines.append(f"  {loser.status.upper():<10} {_describe(loser)}")
            if conflict.decided:
                lines.append(f"  why: {conflict.rationale}")
                lines.append(
                    "  Answer with the accepted value, and say plainly that the "
                    "sources disagree, naming the rejected value and its source."
                )
            else:
                lines.append(
                    "  Do not choose between these. Report that the documents "
                    "disagree and give both values with their sources."
                )
        return "\n".join(lines)


def _describe(claim: Claim) -> str:
    where = claim.source or "unknown source"
    if claim.asserted_at:
        where += f", {claim.asserted_at}"
    return f'{claim.predicate} of {claim.subject} = "{claim.object_value}"  [{where}]  trust {claim.trust:.2f}'


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def subject_key(subject: str) -> str:
    """Grouping key for a subject. Case, punctuation and a leading article are
    not distinctions: `The audit-log shipper` and `audit log shipper` are one
    subject, and treating them as two means the conflict is never found.

    Public because the offline loader has to name a disagreement with the same
    key arbitration grouped it under, and a second implementation of this that
    drifted would silently split one disagreement into two.
    """
    text = (subject or "").casefold().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _subject(claim: Claim) -> str:
    return subject_key(claim.subject)


def _value(claim: Claim) -> str:
    """Grouping key for a value. `30 percent`, `30%` and `30 %` are the same
    number, and a trailing full stop is punctuation rather than disagreement.

    The space before the unit has to go, and for a while it did not: spelling
    `percent` as `%` left `30 %`, which did not match `30%`, so two documents
    that agreed the limit was 30% were reported as contradicting each other.
    A manufactured disagreement is worse than a missed one -- it is the system
    inventing a conflict and then explaining, with a rationale, why it picked
    a side of it.
    """
    text = claim.object_value.casefold().strip()
    text = text.replace("per cent", "%").replace("percent", "%")
    text = re.sub(r"[^a-z0-9%]+", " ", text).strip()
    # `30 %` -> `30%`. Done after the sweep above rather than before, so that
    # `30 per cent` has already become `30 %` by the time it is closed up.
    return re.sub(r"\s+%", "%", text)


# Words that reverse a value. A sentence containing one of these is not a
# longer way of saying the phrase inside it, it is the opposite of it, and
# collapsing "not resolved" onto "resolved" would hide the one disagreement
# that matters most.
_NEGATION = re.compile(r"\b(not|no|never|without|isn t|wasn t|won t|cannot|can t|un\w+)\b")


def _restates(short: str, long: str) -> bool:
    """Whether `long` is the same assertion as `short`, said at more length.

    Extraction returns what the document said, and a document says `resolved`
    in one place and `SUP-3812 (transient p95 spike) is resolved` in another.
    Those are one position stated twice, and treating them as two is the
    system reporting a disagreement that nobody in the corpus is having --
    the most damaging thing a contradiction graph can do, because every part
    of it downstream then explains, with a rationale, why it picked a side of
    an argument that does not exist.

    Three guards, because over-merging hides real conflicts:

    - the shorter value must appear at word boundaries, so `20` does not
      collapse into `120`;
    - it must carry a letter, so two bare numbers are never merged -- numbers
      are exactly where a real disagreement usually lives;
    - neither may negate where the other does not.
    """
    if not short or short == long or len(short) >= len(long):
        return False
    if not any(ch.isalpha() for ch in short):
        return False
    if re.search(rf"(?:^|\s){re.escape(short)}(?:$|\s)", long) is None:
        return False
    return bool(_NEGATION.search(short)) == bool(_NEGATION.search(long))


def _fold(by_value: dict[str, list[Claim]]) -> dict[str, list[Claim]]:
    """Collapse values that restate each other onto the shortest form.

    The shortest wins because it is the one that is actually the value:
    `resolved` is a status, `resolved after identifying a transient spike` is
    a status plus a sentence about why.
    """
    keys = sorted(by_value, key=len)
    canonical: dict[str, str] = {}
    for key in keys:
        canonical[key] = next(
            (short for short in keys if short in canonical and short is not key
             and canonical[short] == short and _restates(short, key)),
            key,
        )
    folded: dict[str, list[Claim]] = {}
    for key, holders in by_value.items():
        folded.setdefault(canonical.get(key, key), []).extend(holders)
    return folded


def _explicitness(claim: Claim) -> float:
    return HEDGED if _HEDGES.search(claim.object_value.casefold()) else 1.0


def score(claim: Claim, *, corroboration: int = 1) -> float:
    """Trust in one claim, ignoring what else it is competing against.

    Recency deliberately is not in here: whether an assertion is stale is a
    fact about the group it sits in, not about the claim, and a per-claim
    recency score would need a "now" that makes the same corpus grade
    differently next month.
    """
    authority = AUTHORITY.get(claim.source, DEFAULT_AUTHORITY)
    confidence = CONFIDENCE_FLOOR + (1 - CONFIDENCE_FLOOR) * claim.extractor_confidence
    agreement = min(
        CORROBORATION_CAP, 1.0 + CORROBORATION_STEP * max(0, corroboration - 1)
    )
    return min(1.0, authority * _explicitness(claim) * confidence * agreement)


def _dated(claim: Claim) -> str | None:
    return claim.asserted_at if _DATE.match(claim.asserted_at or "") else None


def _newness(claim: Claim, dates: list[str]) -> float:
    """Where this claim sits between the oldest and newest dated claim in its
    group, 0 for the oldest and 1 for the newest. Undated claims score 0.5 --
    neither punished nor rewarded for a date the corpus never recorded."""
    date = _dated(claim)
    if not dates or dates[0] == dates[-1]:
        return 1.0
    if date is None:
        return 0.5
    span = [int(part) for part in date.split("-")]
    low = [int(part) for part in dates[0].split("-")]
    high = [int(part) for part in dates[-1].split("-")]
    days = _days(span) - _days(low)
    total = _days(high) - _days(low)
    return days / total if total else 1.0


def _days(parts: list[int]) -> int:
    """Ordinal-ish day count. Month lengths are approximated because only the
    ratio between two gaps matters, never the absolute number of days."""
    year, month, day = parts
    return year * 372 + month * 31 + day


def _rationale(
    winner: Claim,
    winner_size: int,
    losers: Sequence[Claim],
    loser_size: int,
    predicate: str,
    recency_moved: bool,
) -> str:
    """The sentence shown in the UI, naming only the signals that separated the
    two claims.

    An earlier draft asserted recency in every rationale, which was wrong twice
    over: sometimes the dates were equal, and sometimes recency pointed at the
    claim that lost. A rationale that lists signals which did not fire is a
    rationale nobody can check.
    """
    runner_up = losers[0]
    clauses: list[str] = []
    win_authority = AUTHORITY.get(winner.source, DEFAULT_AUTHORITY)
    lose_authority = AUTHORITY.get(runner_up.source, DEFAULT_AUTHORITY)
    if win_authority > lose_authority:
        clauses.append(f"{winner.source} carries more authority than {runner_up.source}")
    if winner_size > loser_size:
        clauses.append(f"corroborated by {winner_size} documents against {loser_size}")
    if recency_moved:
        clauses.append(
            f"asserted later ({winner.asserted_at} against {runner_up.asserted_at}) "
            f"and {predicate} is a value that changes over time"
        )
    elif _dated(winner) and not _dated(runner_up):
        # A real signal, and one that used to go unnamed: an undated claim is
        # scored as neither current nor stale, which costs it trust against a
        # dated rival. That gap decided plenty of conflicts while the rationale
        # said nothing had separated them.
        clauses.append(
            f"the alternative carries no date, so it cannot be shown to be "
            f"current, while this one is dated {winner.asserted_at}"
        )
    if _explicitness(winner) > _explicitness(runner_up):
        clauses.append(f'the alternative hedges its value ("{runner_up.object_value}")')
    if winner.extractor_confidence > runner_up.extractor_confidence:
        clauses.append(
            f"extraction confidence was higher ({winner.extractor_confidence:.2g} "
            f"against {runner_up.extractor_confidence:.2g})"
        )
    if not clauses:
        return NO_SIGNAL
    return "; ".join(clauses)


def arbitrate(claims: Sequence[Claim]) -> Arbitration:
    """Score every claim, group the ones that talk about the same thing, and
    decide -- or decline to.

    Same subject, same predicate, same value is corroboration and raises both
    claims' trust. Same subject and predicate with a different value is a
    conflict, and the losing claim is kept with its own trust and a status
    saying how it lost, so the UI can show what was rejected and the graph can
    store the contradiction rather than quietly discarding half of it.
    """
    groups: dict[tuple[str, str, str], list[Claim]] = {}
    for claim in claims:
        groups.setdefault(
            (claim.scope, _subject(claim), claim.predicate), []
        ).append(claim)

    settled: list[Claim] = []
    conflicts: list[Conflict] = []
    for (scope, subject, predicate), members in groups.items():
        by_value: dict[str, list[Claim]] = {}
        for claim in members:
            by_value.setdefault(_value(claim), []).append(claim)
        by_value = _fold(by_value)
        dates = sorted(d for d in (_dated(c) for c in members) if d)
        staleness = STALENESS.get(predicate, DEFAULT_STALENESS)

        scored: dict[str, list[Claim]] = {}
        for value, holders in by_value.items():
            scored[value] = [
                replace(
                    claim,
                    trust=round(
                        score(claim, corroboration=len(holders))
                        * (1 - staleness * (1 - _newness(claim, dates))),
                        4,
                    ),
                )
                for claim in holders
            ]
        # Deterministic order everywhere: trust, then the later assertion, then
        # the document id, so the same claims never arbitrate two ways. Three
        # stable sorts rather than one key, because the middle field wants
        # descending order and the others ascending.
        for holders in scored.values():
            holders.sort(key=lambda c: c.doc_id)
            holders.sort(key=lambda c: c.asserted_at, reverse=True)
            holders.sort(key=lambda c: c.trust, reverse=True)
        ranked = sorted(scored.values(), key=lambda hs: (-hs[0].trust, hs[0].doc_id))

        if len(ranked) == 1:
            settled.extend(ranked[0])
            continue

        winner, runner_up = ranked[0][0], ranked[1][0]
        losing = [claim for holders in ranked[1:] for claim in holders]
        # Recency only counts as a reason when it could have moved the result:
        # the predicate is one where staleness has weight, the dates differ,
        # and the winner is the later claim.
        recency_moved = bool(
            staleness
            and _dated(winner)
            and _dated(runner_up)
            and winner.asserted_at > runner_up.asserted_at
        )
        decided = winner.trust >= TRUST_FLOOR and winner.trust - runner_up.trust >= MARGIN
        if decided:
            rationale = _rationale(
                winner, len(ranked[0]), [runner_up], len(ranked[1]), predicate, recency_moved
            )
            # A decision that cannot be explained is not presented as one. The
            # arithmetic can separate two claims by more than the margin while
            # no individual signal is nameable, and "accepted -- no signal
            # separated these claims" is a verdict contradicting its own
            # reason. Refusing is the honest reading of that state, and it is
            # the same refusal the trust floor already performs.
            decided = rationale != NO_SIGNAL
        if decided:
            winner_status, loser_status = "accepted", "superseded" if recency_moved else "disputed"
        elif winner.trust < TRUST_FLOOR:
            rationale = (
                f"both values are weakly evidenced (trust {winner.trust:.2f} and "
                f"{runner_up.trust:.2f}, floor {TRUST_FLOOR:.2f})"
            )
            winner_status = loser_status = "disputed"
        else:
            rationale = (
                f"trust is too close to choose ({winner.trust:.2f} against "
                f"{runner_up.trust:.2f})"
            )
            winner_status = loser_status = "disputed"

        winners = [replace(c, status=winner_status, rationale=rationale) for c in ranked[0]]
        losers = [replace(c, status=loser_status, rationale=rationale) for c in losing]
        settled.extend(winners)
        settled.extend(losers)
        conflicts.append(
            Conflict(
                subject=subject,
                predicate=predicate,
                scope=scope,
                winner=winners[0],
                losers=tuple(losers),
                rationale=rationale,
            )
        )
    return Arbitration(claims=tuple(settled), conflicts=tuple(conflicts))
