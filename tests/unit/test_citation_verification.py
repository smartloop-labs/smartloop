"""Citations credit only the sources the answer actually used.

Retrieval returning a document does not mean the answer used it: nearest
neighbour always returns *something*, so a project holding a single document
surfaces it for every question however unrelated.  That is how a `good-morning`
workout skill ended up citing an invoice — the invoice was the only document in
the project, so it was necessarily the closest match.

Matching is on document *content*.  An earlier revalidation matched on the
filename and silently dropped real sources whose names say nothing about their
contents ("sp46.pdf" for a Vanguard fund); see the comment at the emission site.

The evidence is a shared run of consecutive words, or a shared precise figure.
Single shared words were tried first and are far too weak: measured on the
transcripts below, a workout answer and an unrelated fund prospectus share
7-12 ordinary English words, which is exactly how five Vanguard PDFs came to
be credited to a `good-morning` routine.
"""

import pytest

from smartloop.api.routers.chat import (
    _citation_used,
    _figures,
    _phrases,
    _verified_citations,
)

INVOICE = (
    "=== Source: Invoice-Smartloop-Chris-Coveney.pdf ===\n"
    "Invoice from Smartloop Inc. to Chris Coveney. Amount $1,000.00 for "
    "contract work services. Invoice date August 2026."
)


def _cit(extract=INVOICE, doc_id="d1", name="Invoice-Smartloop-Chris-Coveney.pdf"):
    c = {"document_id": doc_id, "document_name": name}
    if extract is not None:
        c["_extract"] = extract
    return c


# --------------------------------------------------------------------------
# the reported bug
# --------------------------------------------------------------------------

def test_unrelated_answer_does_not_cite_the_document():
    answer = (
        "Good morning! Here are 10 minute workout routines focusing on upper "
        "body. Push-ups (3 sets of 8-12 reps), dumbbell rows, shoulder presses."
    )
    assert _verified_citations([_cit()], answer) == []


def test_answer_that_uses_the_document_still_cites_it():
    answer = ("The invoice is for $1,000.00 paid by Smartloop Inc. to "
              "Chris Coveney for contract work services.")
    out = _verified_citations([_cit()], answer)
    assert [c["document_name"] for c in out] == ["Invoice-Smartloop-Chris-Coveney.pdf"]


# --------------------------------------------------------------------------
# what still counts as use
# --------------------------------------------------------------------------

def test_paraphrase_keeping_the_wording_is_kept():
    """Reworded around the source's own phrase — still a real use."""
    assert _citation_used(_cit(), "Coveney was engaged for contract work.")


def test_numeric_match_survives_formatting():
    """'$1,000.00' in the source must match '1000' in the answer.

    A translated or fully reworded answer keeps none of the source's phrasing
    but still carries its figures, so a shared figure alone qualifies.
    """
    assert _citation_used(_cit(), "The total came to 1000 dollars.")


def test_small_numbers_are_not_evidence():
    """"3 sets of 10 reps" must not match every document containing a 3."""
    c = _cit(extract="The fund held 3 positions across 10 sectors last year.")
    assert not _citation_used(c, "Do 3 sets of 10 reps with a 5 second hold.")


def test_incidental_common_words_are_not_evidence():
    """The measured failure: unrelated texts share ordinary vocabulary."""
    c = _cit(extract=(
        "The Fund holds securities that may decline in value. Shares are "
        "available to investors since inception; performance is not otherwise "
        "guaranteed. Three different share classes are offered."
    ))
    answer = (
        "Since it is Wednesday, here are three different options. Hold each "
        "plank for 45 seconds and perform arm raises. Focus on full range."
    )
    assert not _citation_used(c, answer)


def test_citation_without_an_extract_is_always_kept():
    """Absence of evidence is not evidence of misuse."""
    assert _citation_used(_cit(extract=None), "anything at all")


def test_empty_response_keeps_the_citation():
    assert _citation_used(_cit(), "")


def test_extract_too_short_to_judge_keeps_the_citation():
    """Nothing to compare against is not proof of misuse."""
    assert _citation_used(_cit(extract="Rye flour"), "unrelated")


def test_filename_alone_does_not_justify_a_citation():
    """The regression that got the old filename check reverted, inverted.

    Echoing the filename is not evidence the *content* was used.
    """
    answer = "I could not read Invoice-Smartloop-Chris-Coveney.pdf."
    assert _verified_citations([_cit(extract="Vanguard fund S&P 500 holdings")], answer) == []


def test_opaque_filename_is_no_obstacle():
    """'sp46.pdf' says nothing about its contents; the content still matches."""
    c = _cit(extract="Vanguard S&P 500 index fund quarterly holdings report",
             name="sp46.pdf")
    assert _citation_used(c, "This is the Vanguard S&P 500 index fund.")


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

def test_private_extract_never_reaches_the_client():
    out = _verified_citations([_cit()], "Chris Coveney was paid $1,000.00")
    assert out and all(not k.startswith("_") for k in out[0])


def test_duplicates_are_collapsed():
    answer = "Chris Coveney was paid $1,000.00"
    out = _verified_citations([_cit(), _cit()], answer)
    assert len(out) == 1


def test_documents_are_judged_independently():
    used = _cit(extract="Quarterly revenue reached 4.25 million euros", doc_id="a", name="a.pdf")
    unused = _cit(extract="Recipe for sourdough bread with rye flour", doc_id="b", name="b.pdf")
    out = _verified_citations([used, unused], "Revenue reached 4.25 million euros.")
    assert [c["document_id"] for c in out] == ["a"]


def test_no_citations_in_no_citations_out():
    assert _verified_citations([], "anything") == []
    assert _verified_citations(None, "anything") == []


def test_dedup_falls_back_to_name_when_id_missing():
    c = {"document_name": "x.pdf", "_extract": "widget assembly torque specification is 12Nm"}
    out = _verified_citations([dict(c), dict(c)], "The widget assembly torque specification is 12Nm.")
    assert len(out) == 1


# --------------------------------------------------------------------------
# the matching primitives
# --------------------------------------------------------------------------

def test_phrases_are_runs_of_three_words():
    assert _phrases("alpha beta gamma delta") == {"alpha beta gamma", "beta gamma delta"}


def test_phrases_ignore_case_and_punctuation():
    assert "total came to" in _phrases("The total, came to: 1000")


def test_text_shorter_than_a_phrase_yields_nothing():
    assert _phrases("two words") == set()


@pytest.mark.parametrize("text,expected", [
    ("$1,000.00", {"1000"}),
    ("1000", {"1000"}),
    ("0.36%", {"0.36"}),
    ("3 sets of 10 reps", set()),      # too few digits to be evidence
    ("page 5 of 12", set()),
])
def test_figures_normalise_formatting(text, expected):
    assert _figures(text) == expected


def test_figures_survive_a_number_too_long_for_a_float():
    """A 400-digit run overflows float() to inf; int(inf) used to raise
    OverflowError inside the streaming generator and abort the response."""
    huge = "9" * 400
    assert _figures(huge) == {huge}
    assert _citation_used(_cit(extract="total " + huge), "The total is " + huge)


def test_verification_failure_keeps_sources_instead_of_raising(monkeypatch):
    monkeypatch.setattr(
        "smartloop.api.routers.chat._verify_citations",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    out = _verified_citations([_cit(), _cit()], "any answer")
    assert len(out) == 1
    assert "_extract" not in out[0]
