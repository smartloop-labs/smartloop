"""Citations credit only the sources the answer actually used.

Retrieval returning a document does not mean the answer used it: nearest
neighbour always returns *something*, so a project holding a single document
surfaces it for every question however unrelated.  That is how a `good-morning`
workout skill ended up citing an invoice — the invoice was the only document in
the project, so it was necessarily the closest match.

Matching is on document *content*.  An earlier revalidation matched on the
filename and silently dropped real sources whose names say nothing about their
contents ("sp46.pdf" for an index fund); see the comment at the emission site.

The evidence is a shared run of consecutive words, or a shared precise figure.
Single shared words were tried first and are far too weak: measured on the
transcripts below, a workout answer and an unrelated fund prospectus share
7-12 ordinary English words, which is exactly how five fund PDFs came to be
credited to a `good-morning` routine.

Every fixture below is invented.  The behaviour under test came from real
transcripts, but a fixture is read far more often than the transcript it came
from, so no real person, employer, document or product name appears here.
"""

import pytest

from smartloop.api.routers.chat import (
    _citation_used,
    _figures,
    _phrases,
    _verified_citations,
)

INVOICE_DOC = "invoice-contoso-jordan-rivera.pdf"
INVOICE = (
    f"=== Source: {INVOICE_DOC} ===\n"
    "Invoice from Contoso Ltd. to Jordan Rivera. Amount $1,000.00 for "
    "contract work services. Invoice date August 2026."
)


def _cit(extract=INVOICE, doc_id="d1", name=INVOICE_DOC):
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
    answer = ("The invoice is for $1,000.00 paid by Contoso Ltd. to "
              "Jordan Rivera for contract work services.")
    out = _verified_citations([_cit()], answer)
    assert [c["document_name"] for c in out] == [INVOICE_DOC]


# --------------------------------------------------------------------------
# what still counts as use
# --------------------------------------------------------------------------

def test_paraphrase_keeping_the_wording_is_kept():
    """Reworded around the source's own phrase — still a real use."""
    assert _citation_used(_cit(), "Rivera was engaged for contract work.")


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
    answer = f"I could not read {INVOICE_DOC}."
    assert _verified_citations(
        [_cit(extract="Northwind Growth Fund quarterly holdings")], answer,
    ) == []


def test_opaque_filename_is_no_obstacle():
    """'sp46.pdf' says nothing about its contents; the content still matches."""
    c = _cit(extract="Northwind Growth Fund quarterly holdings report",
             name="sp46.pdf")
    assert _citation_used(c, "This is the Northwind Growth Fund.")


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

def test_private_extract_never_reaches_the_client():
    out = _verified_citations([_cit()], "Jordan Rivera was paid $1,000.00")
    assert out and all(not k.startswith("_") for k in out[0])


def test_duplicates_are_collapsed():
    answer = "Jordan Rivera was paid $1,000.00"
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


# --------------------------------------------------------------------------
# grammar is not attribution
# --------------------------------------------------------------------------
#
# A citation exists so a reader can trace a statement back to the text it came
# from, so the check is a text comparison and nothing else — no model call,
# nothing for the user to wait through once the answer is on screen.
#
# That puts the whole weight on what counts as a shared run.  Every English
# text contains "here are the" and "in the last"; a run built only out of
# grammar is a coincidence of the language, not a trace of one text in
# another, and each such run used to be enough to credit a document the answer
# had never opened.

def test_a_run_of_pure_grammar_is_not_evidence():
    assert _phrases("here are the") == set()
    assert _phrases("as well as") == set()
    assert _phrases("one of the") == set()


def test_a_run_with_one_content_word_is_not_evidence():
    """The shape of an incidental collision: grammar around a single word."""
    assert _phrases("here are the") == set()
    assert _phrases("in the last") == set()


def test_a_run_with_two_content_words_is_evidence():
    assert "for contract work" in _phrases("paid for contract work")
    assert "total came to" in _phrases("The total, came to: 1000")


def test_grammar_only_overlap_no_longer_credits_a_document():
    """Two texts about nothing in common, sharing only sentence scaffolding."""
    c = _cit(extract=(
        "Here are the terms. As well as the above, one of the parties shall "
        "in the last instance be liable."
    ))
    answer = (
        "Here are the steps. As well as the setup, one of the options is to "
        "in the last resort restart."
    )
    assert not _citation_used(c, answer)


def test_shared_phrasing_still_credits_the_document():
    """The filter must not cost a real trace."""
    answer = "Jordan Rivera was invoiced $1,000.00 for contract work services."
    out = _verified_citations([_cit()], answer)
    assert [c["document_name"] for c in out] == [INVOICE_DOC]


# --------------------------------------------------------------------------
# the reported cases
# --------------------------------------------------------------------------

# A stand-in for the indexed personal document in the reported cases.  Real
# names and employers are deliberately absent — a fixture is read far more
# often than the transcript it came from.
RESUME = (
    "Principal Widget Architect at Acme Corp. Designed and led "
    "production-grade widget assembly lines for industrial customers. "
    "Tech Lead and Co-Founder, Example Industries."
)


def _resume_cit(extract=RESUME):
    return _cit(doc_id="r1", name="candidate-cv.pdf", extract=extract)


def test_an_unrelated_document_is_not_a_source_for_a_camera_answer():
    """A camera answer, with a CV as the only indexed document.

    The text comparison always got this one right — the answer and the CV have
    nothing whatever in common.  It was an LLM judge, since removed, that put
    the CV back: asked whether the answer had drawn on it, the small model
    said yes.
    """
    answer = (
        "Based on the provided search results, here are the key features for "
        "the Zenith QX-9 camera: it is equipped with a medium format BSI CMOS "
        "sensor featuring 100 megapixels and 5-axis stabilisation."
    )
    assert _verified_citations([_resume_cit()], answer) == []


def test_a_document_the_answer_was_written_from_is_kept():
    """The other half: when the answer really is the document, say so.

    The same CV, on the request whose answer genuinely was built from it — the
    citation is the only thing telling the reader where those job titles came
    from.
    """
    answer = (
        "Based on the provided documents, here are the best jobs these days: "
        "Principal Widget Architect, designing and leading production-grade "
        "widget assembly lines for industrial customers."
    )
    out = _verified_citations([_resume_cit()], answer)
    assert [c["document_name"] for c in out] == ["candidate-cv.pdf"]


# --------------------------------------------------------------------------
# the check has to work in the language the answer is written in
# --------------------------------------------------------------------------
#
# The tokenizer used to be ``[a-z0-9][a-z0-9.,%/-]*``.  Outside English that
# did not weaken the check, it switched it off: no ASCII letters meant no
# words, no words meant no runs, no runs meant no evidence either way — and
# "no evidence" is deliberately read as "keep the source".  Every citation on
# every Arabic, Chinese, Hebrew, Russian, Hindi or Thai answer was credited
# unconditionally.  Accented Latin was worse than useless: "über" tokenised to
# "ber", "für" to "f" and "r", and those fragments appear in no stopword list,
# so mangled German counted as three content words.

def _src(text):
    return {"document_id": "s1", "document_name": "s.pdf", "_extract": text}


def test_accented_words_survive_tokenisation():
    """Not "ber" and not "f" + "r"."""
    phrases = _phrases("Die Rechnung über tausend Euro für Vertragsarbeit")
    assert "rechnung über tausend" in phrases
    assert "euro für vertragsarbeit" in phrases


def test_a_script_without_latin_letters_still_yields_runs():
    assert _phrases("الفاتورة من شركة كونتوسو بمبلغ ألف دولار")


def test_an_unrelated_arabic_source_is_dropped():
    assert not _citation_used(
        _src("السيرة الذاتية: مهندس معماري رئيسي في شركة أكمي"),
        "الميزات الرئيسية لكاميرا زينيث: مستشعر بدقة 100 ميجابكسل",
    )


def test_an_arabic_source_the_answer_used_is_kept():
    assert _citation_used(
        _src("الفاتورة من شركة كونتوسو بمبلغ ألف دولار مقابل أعمال تعاقدية"),
        "دفعت شركة كونتوسو بمبلغ ألف دولار مقابل أعمال تعاقدية.",
    )


# Japanese, Chinese and Thai put no spaces between words, so splitting on
# whitespace yields one enormous token and no runs — the same hole, reopened.
# Their runs are counted in characters instead.

def test_a_space_less_script_yields_character_runs():
    assert _phrases("发票金额为一千美元的合同工作服务")


def test_an_unrelated_chinese_source_is_dropped():
    assert not _citation_used(
        _src("简历：阿克米公司首席架构师，负责设计生产级系统"),
        "泽尼斯相机的主要特点是一亿像素传感器",
    )


def test_a_chinese_source_the_answer_used_is_kept():
    assert _citation_used(
        _src("简历：阿克米公司首席架构师，负责设计生产级系统"),
        "他是阿克米公司首席架构师，负责设计生产级系统。",
    )


def test_a_japanese_source_the_answer_used_is_kept():
    assert _citation_used(
        _src("請求書：コントソ社への契約作業に対する千ドルの支払い"),
        "コントソ社への契約作業に対する千ドルが支払われました。",
    )
