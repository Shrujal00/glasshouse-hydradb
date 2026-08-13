"""Entity resolution tests.

Most of these encode a merge that actually went wrong against the real corpus
during the first run. Precision here is the top-scored thing in the track, and
the failures are not hypothetical: `@jae` really did bind Jordan Ae and Jamie Ae
into one person, and "Maya Chen (Redwood AE)" really did acquire the surname
"AE".
"""

from collections import Counter

import pytest

from glasshouse.priors import Priors
from glasshouse.resolve import (
    Surface,
    SurfaceIndex,
    handle_tokens,
    is_email,
    looks_like_person_name,
    name_tokens,
    norm_name,
    resolve,
)

# What the corpus in these tests would have taught us. Stated explicitly here
# rather than hidden in the module, because that is the whole point: the
# resolver knows nothing about any company until a corpus tells it.
PRIORS = Priors(
    home_labels=frozenset({"redwood", "redwoodinference"}),
    qualifiers=frozenset({"ae", "csm", "redwood", "am"}),
    functional=frozenset({"compliance", "procurement", "events"}),
)


def surface(kind: str, value: str, count: int = 10, priors: Priors = PRIORS) -> Surface:
    return Surface.build(kind, value, priors, count=count)


def index_of(*surfaces: Surface) -> dict[str, Surface]:
    return {s.key: s for s in surfaces}


def entity_for(entities, alias: str):
    """The resolved entity containing a given surface value."""
    return next(e for e in entities if any(a.value == alias for a in e.aliases))


# --- normalization ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Maya Chen", "maya chen"),
        ("Maya Chen (Redwood AE)", "maya chen"),
        ("Maya Chen Redwood AE", "maya chen"),
        ("Priya Nair (CSM)", "priya nair"),
        ("Tomás Alvarez", "tomás alvarez"),
        ("Liam O’Connell", "liam o'connell"),
    ],
)
def test_role_qualifiers_are_not_surnames(raw, expected):
    assert norm_name(raw, PRIORS) == expected


def test_a_name_made_only_of_role_words_is_left_alone():
    """Stripping must not erase the whole name; "Jordan AE" keeps something."""
    assert name_tokens("Jordan AE", PRIORS) == ["jordan", "ae"]


def test_accented_and_apostrophed_names_are_people():
    assert looks_like_person_name("Tomás Alvarez", PRIORS)
    assert looks_like_person_name("Liam O'Connell", PRIORS)


@pytest.mark.parametrize("junk", ["Meeting Header", "Action Items", "The Team", "hey"])
def test_transcript_artifacts_are_not_people(junk):
    assert not looks_like_person_name(junk, PRIORS)


def test_email_validation_rejects_a_swallowed_transcript():
    """The normalizer's `Name <...>` pattern can span newlines; this is the net."""
    assert is_email("maya.chen@redwood.ai")
    assert not is_email("meeting header")
    assert not is_email("2% on the golden set,\n[01:14] clara su: works for product")


def test_handle_tokens_drop_disambiguation_digits():
    assert handle_tokens("mchen2", PRIORS) == ["mchen"]
    assert handle_tokens("maya.chen", PRIORS) == ["maya", "chen"]


# --- mining -----------------------------------------------------------------


def test_stated_binding_is_recorded_and_survives_the_frequency_floor():
    idx = SurfaceIndex(PRIORS)
    idx.add_document(
        {
            "source": "gmail",
            "named_emails": [{"name": "Maya Chen", "email": "maya.chen@redwood.ai"}],
        }
    )
    assert idx.hard_links[("name:maya chen", "email:maya.chen@redwood.ai")] == 1
    # Seen once, far below MIN_OCCURRENCES, but explicitly stated.
    assert set(idx.working_set(min_occurrences=5)) == {
        "name:maya chen",
        "email:maya.chen@redwood.ai",
    }


def test_malformed_values_are_rejected_and_counted():
    idx = SurfaceIndex(PRIORS)
    idx.add_document(
        {"source": "fireflies", "emails": ["not an email"], "speakers": ["Meeting Header"]}
    )
    assert idx.surfaces == {}
    assert idx.rejected["email"] == 1
    assert idx.rejected["handle"] == 1


# --- scoring and clustering -------------------------------------------------


def test_handle_resolves_to_the_person_who_owns_the_initial():
    surfaces = index_of(
        surface("handle", "mchen"),
        surface("name", "maya chen"),
        surface("email", "maya.chen@redwood.ai"),
    )
    entities, _ = resolve(surfaces, Counter())
    assert len(entity_for(entities, "mchen").aliases) == 3


def test_an_ambiguous_initial_merges_nobody():
    """`@jae` fits Jordan Ae and Jamie Ae, so it is evidence for neither."""
    surfaces = index_of(
        surface("handle", "jae"),
        surface("name", "jordan ae"),
        surface("name", "jamie ae"),
    )
    entities, diag = resolve(surfaces, Counter())
    assert diag["merges_applied"] == 0
    assert len(entities) == 3


def test_an_ambiguous_given_name_merges_nobody():
    """Eight Priyas exist in this corpus; a bare `@priya` names none of them."""
    surfaces = index_of(
        surface("handle", "priya"),
        surface("name", "priya nair"),
        surface("name", "priya anand"),
    )
    entities, diag = resolve(surfaces, Counter())
    assert diag["merges_applied"] == 0


def test_a_unique_given_name_does_resolve():
    surfaces = index_of(surface("handle", "soham"), surface("name", "soham ratnaparkhi"))
    entities, _ = resolve(surfaces, Counter())
    assert len(entity_for(entities, "soham").aliases) == 2


def test_an_ambiguous_surname_initial_merges_nobody():
    """Jordan Chen and Jamie Chen both fit `jchen`, so it names neither."""
    surfaces = index_of(
        surface("handle", "jchen"),
        surface("email", "jordan.chen@redwood.ai"),
        surface("email", "jamie.chen@redwood.ai"),
    )
    entities, diag = resolve(surfaces, Counter())
    assert diag["merges_applied"] == 0
    assert len(entities) == 3


def test_two_mailboxes_never_land_in_one_cluster():
    """The precision guard: one person, one mailbox name.

    Both addresses have unambiguous evidence pulling them onto Maya Chen —
    `maya.chen@` by exact name match, `mchen@` by a uniquely-owned initial —
    yet they cannot both be her mailbox, so the weaker merge is refused.
    """
    surfaces = index_of(
        surface("name", "maya chen"),
        surface("email", "maya.chen@redwood.ai"),
        surface("email", "mchen@redwood.ai"),
    )
    entities, diag = resolve(surfaces, Counter())
    assert diag["merges_refused_by_constraint"] >= 1
    assert all(sum(1 for a in e.aliases if a.kind == "email") <= 1 for e in entities)
    # The stronger evidence is the one that survives.
    assert {a.value for a in entity_for(entities, "maya chen").aliases} == {
        "maya chen",
        "maya.chen@redwood.ai",
    }


def test_one_person_may_hold_the_same_mailbox_on_two_domains():
    surfaces = index_of(
        surface("email", "maya.chen@redwood.com"),
        surface("email", "maya.chen@redwood.ai"),
        surface("name", "maya chen"),
    )
    entities, _ = resolve(surfaces, Counter())
    assert len(entity_for(entities, "maya chen").aliases) == 3


def test_a_stated_binding_reaches_past_blocking():
    """`mc` shares no name-shaped key with "maya chen", so blocking never pairs
    them — but the corpus wrote the binding down, which outranks inference."""
    a, b = surface("name", "maya chen"), surface("email", "mc@redwood.ai")
    hard = Counter({(a.key, b.key): 3})
    entities, _ = resolve(index_of(a, b), hard)
    entity = entity_for(entities, "maya chen")
    assert {s.value for s in entity.aliases} == {"maya chen", "mc@redwood.ai"}
    assert entity.merges[0].signals == ["STATED_BINDING"]


def test_a_departmental_mailbox_is_not_a_person():
    idx = SurfaceIndex(PRIORS)
    for _ in range(9):
        idx.add_document({"source": "gmail", "emails": ["compliance@redwood.com"]})
    assert "email:compliance@redwood.com" in idx.shared_emails()
    assert "email:compliance@redwood.com" not in idx.working_set(min_occurrences=1)


def test_an_address_signed_by_two_people_is_a_shared_mailbox():
    """The behavioural test, which catches what no stoplist can.

    `procurement@harbortech.com` is not a departmental *name*, but the corpus
    binds it to both Susan Park and Susan Lee, so it belongs to neither.
    """
    idx = SurfaceIndex(PRIORS)
    for who in ("Susan Park", "Susan Lee"):
        idx.add_document(
            {
                "source": "gmail",
                "named_emails": [{"name": who, "email": "procurement@harbortech.com"}],
            }
        )
    assert idx.shared_emails() == {"email:procurement@harbortech.com"}
    assert idx.exclusive_links() == Counter()
    # And so the two women are never joined through it.
    entities, _ = resolve(idx.working_set(1), idx.exclusive_links())
    assert len(entities) == 2


def test_a_personal_mailbox_keeps_its_binding():
    idx = SurfaceIndex(PRIORS)
    idx.add_document(
        {"source": "gmail", "named_emails": [{"name": "Maya Chen", "email": "mc@redwood.ai"}]}
    )
    assert idx.shared_emails() == set()
    assert idx.exclusive_links()[("name:maya chen", "email:mc@redwood.ai")] == 1


def test_the_same_name_at_two_companies_is_two_people():
    """The org guard. Nothing but the domain separates these two."""
    surfaces = index_of(
        surface("name", "susan park"),
        surface("email", "susan.park@acme.com"),
        surface("email", "susan.park@novushealth.com"),
    )
    entities, diag = resolve(surfaces, Counter())
    assert diag["merges_refused_by_constraint"] >= 1
    assert len(entities) == 2
    assert all(len({a.org for a in e.aliases if a.org}) <= 1 for e in entities)


@pytest.mark.parametrize(
    "email,org",
    [
        ("elena.rossi@salushealth.it", "salushealth"),
        ("elena.rossi@novamed.it", "novamed"),
        ("s.park@acmecorp.co.uk", "acmecorp"),
        ("x@mail.harbortech.com", "harbortech"),
    ],
)
def test_org_is_read_from_the_leading_label(email, org):
    """A country TLD must not become the employer.

    Reading domains from the right requires knowing every TLD in the corpus;
    `.it` was missing, so two Elena Rossis at different Italian companies both
    reduced to the org "it" and merged into one person.
    """
    assert surface("email", email).org == org


@pytest.mark.parametrize(
    "email", ["maya@redwood.ai", "maya@redwood.inference.com", "maya@redwoodinference.com"]
)
def test_every_spelling_of_the_employer_is_one_employer(email):
    """The learned home labels collapse to a single value, whatever it is.

    The employer's identity is not a string we assert — it is whichever
    domain family the corpus taught us — so the test asserts that all its
    spellings agree with each other and disagree with an outsider.
    """
    home = surface("email", "maya@redwood.com").org
    assert surface("email", email).org == home
    assert surface("email", "s.park@acmecorp.co.uk").org != home


def test_two_people_at_different_italian_companies_stay_apart():
    surfaces = index_of(
        surface("name", "elena rossi"),
        surface("email", "elena.rossi@salushealth.it"),
        surface("email", "elena.rossi@novamed.it"),
    )
    entities, _ = resolve(surfaces, Counter())
    assert len(entities) == 2


def test_one_person_may_span_the_employers_own_domains():
    """redwood.com, redwood.ai and redwoodinference.com are one employer."""
    surfaces = index_of(
        surface("name", "maya chen"),
        surface("email", "maya.chen@redwood.com"),
        surface("email", "maya.chen@redwoodinference.com"),
    )
    entities, _ = resolve(surfaces, Counter())
    assert len(entity_for(entities, "maya chen").aliases) == 3


def test_a_stated_binding_cannot_override_the_org_guard():
    a = surface("name", "susan park")
    b = surface("email", "susan.park@novushealth.com")
    surfaces = index_of(a, b, surface("email", "susan.park@acme.com"))
    entities, diag = resolve(surfaces, Counter({(a.key, b.key): 5}))
    assert diag["merges_refused_by_constraint"] >= 1
    assert all(len({s.org for s in e.aliases if s.org}) <= 1 for e in entities)


def test_the_stated_mailbox_wins_the_slot_and_the_loser_is_recorded():
    """A known limitation, kept deliberately.

    One person really can hold two mailboxes, and here both `mc@` and
    `maya.chen@` have honest evidence pointing at Maya Chen. The constraint
    admits only the better-evidenced one — the stated binding — because
    relaxing it is what lets two *different* people chain into one cluster,
    and a false merge costs more than a missed alias. The refusal is counted,
    not silently dropped, so the review UI can surface it.
    """
    a, b = surface("name", "maya chen"), surface("email", "mc@redwood.ai")
    surfaces = index_of(a, b, surface("email", "maya.chen@redwood.ai"))
    entities, diag = resolve(surfaces, Counter({(a.key, b.key): 3}))
    assert {s.value for s in entity_for(entities, "maya chen").aliases} == {
        "maya chen",
        "mc@redwood.ai",
    }
    assert diag["merges_refused_by_constraint"] == 1


def test_every_merge_carries_its_reason():
    """A merge that cannot say why it happened is not auditable."""
    surfaces = index_of(surface("handle", "mchen"), surface("name", "maya chen"))
    entities, _ = resolve(surfaces, Counter())
    entity = entity_for(entities, "mchen")
    assert entity.merges
    assert all(m.signals and m.score > 0 for m in entity.merges)


def test_confidence_is_the_weakest_link_in_the_cluster():
    surfaces = index_of(
        surface("handle", "mchen"),
        surface("name", "maya chen"),
        surface("email", "maya.chen@redwood.ai"),
    )
    entities, _ = resolve(surfaces, Counter())
    entity = entity_for(entities, "mchen")
    assert entity.confidence == pytest.approx(min(m.score for m in entity.merges))


def test_canonical_name_prefers_a_real_name_over_a_handle():
    surfaces = index_of(
        surface("handle", "mchen", count=900),
        surface("name", "maya chen", count=12),
    )
    entities, _ = resolve(surfaces, Counter())
    assert entity_for(entities, "mchen").canonical_name == "Maya Chen"
