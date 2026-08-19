"""Citations credit only the sources the answer actually used.

Retrieval returning a document does not mean the answer used it: nearest
neighbour always returns *something*, so a project holding a single document
surfaces it for every question however unrelated.  That is how a `good-morning`
workout skill ended up citing an invoice — the invoice was the only document in
the project, so it was necessarily the closest match.

Matching is on document *content*.  An earlier revalidation matched on the
filename and silently dropped real sources whose names say nothing about their
contents ("sp46.pdf" for a Vanguard fund); see the comment at the emission site.
"""

import pytest

from smartloop.api.routers.chat import (
    _citation_used,
    _distinctive_tokens,
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
# conservative: only drop on *zero* shared content
# --------------------------------------------------------------------------

def test_paraphrase_without_figures_is_kept():
    """Genuine use echoes at least one entity even when reworded."""
    assert _citation_used(_cit(), "Coveney was engaged for contract work.")


def test_numeric_match_survives_formatting():
    """'$1,000.00' in the source must match '1000' in the answer."""
    assert _citation_used(_cit(), "The total came to 1000 dollars.")


def test_citation_without_an_extract_is_always_kept():
    """Absence of evidence is not evidence of misuse."""
    assert _citation_used(_cit(extract=None), "anything at all")


def test_empty_response_keeps_the_citation():
    assert _citation_used(_cit(), "")


def test_extract_of_only_stopwords_keeps_the_citation():
    assert _citation_used(_cit(extract="the and of for with that this"), "unrelated")


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
    assert _citation_used(c, "The Vanguard fund tracks the S&P 500 index.")


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
    used = _cit(extract="Quarterly revenue reached 4.2 million euros", doc_id="a", name="a.pdf")
    unused = _cit(extract="Recipe for sourdough bread with rye flour", doc_id="b", name="b.pdf")
    out = _verified_citations([used, unused], "Revenue reached 4.2 million euros.")
    assert [c["document_id"] for c in out] == ["a"]


def test_no_citations_in_no_citations_out():
    assert _verified_citations([], "anything") == []
    assert _verified_citations(None, "anything") == []


def test_dedup_falls_back_to_name_when_id_missing():
    c = {"document_name": "x.pdf", "_extract": "widget assembly torque specification"}
    out = _verified_citations([dict(c), dict(c)], "The widget torque specification is 12Nm.")
    assert len(out) == 1


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("common", ["the", "document", "page", "summary", "information"])
def test_stopwords_are_not_distinctive(common):
    assert common not in _distinctive_tokens(f"the {common} of that report")


def test_short_tokens_are_ignored():
    assert _distinctive_tokens("a to of an it") == set()


@pytest.mark.parametrize("written,expected", [
    ("$1,000.00", "1000"),
    ("1,000", "1000"),
    ("1000.00", "1000"),
    ("4.2", "4.2"),
])
def test_numbers_normalise_to_a_comparable_form(written, expected):
    """Stripping to digits alone would turn '1,000.00' into '100000'."""
    assert expected in _distinctive_tokens(written)
