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


# ---------------------------------------------------------------------------
# The embedding path.
#
# Word overlap is the wrong test for a citation: an answer is a rewrite of its
# source, not a quotation of it, so a paraphrase, a translation or a summary
# into a table keeps none of the source's phrasing while being entirely about
# it.  Verification therefore embeds both sides and compares them the way
# retrieval compares a chunk to the user's question — `encode_texts` for the
# vectors, `cosine_distances` for the comparison, the retrieval quality floor
# for the cutoff.  The lexical checks above remain as the fallback for installs
# with no embedding model, which is why they are still exercised.
#
# The stub embedder below is a bag of words over a fixed vocabulary.  It is not
# a language model and knows no synonyms, so these tests pin the plumbing --
# batching, windowing, per-source scoring, thresholding, degradation -- and not
# how well bge-m3 happens to score any particular pair of texts.

import numpy as np

from smartloop.api.routers.chat import (
    _CITATION_CUTOFF_MARGIN,
    _CITATION_MAX_DISTANCE,
    _MAX_WINDOWS,
    _WINDOW_CHARS,
    _citation_sources,
    _semantic_distances,
    _windows,
)

_VOCAB = ["invoice", "contoso", "rivera", "camera", "sensor", "hasselblad"]


def _bag(text: str) -> list:
    lowered = text.lower()
    return [float(lowered.count(term)) for term in _VOCAB]


def _stub_encoder(monkeypatch, calls=None):
    """Install a deterministic bag-of-words embedder in place of the model."""
    def encode_texts(texts, base_dir=None):
        if calls is not None:
            calls.append(list(texts))
        # A text with none of the vocabulary still needs a non-zero vector:
        # the zero vector normalises to itself and is equidistant from
        # everything, which would make the threshold meaningless.
        rows = [_bag(t) or None for t in texts]
        rows = [r if r and any(r) else [1e-6] * len(_VOCAB) for r in rows]
        return np.asarray(rows, dtype=np.float32)

    monkeypatch.setattr("smartloop.db.embeddings.encode_texts", encode_texts)


def _distance(monkeypatch, source: str, answer: str) -> float:
    _stub_encoder(monkeypatch)
    return _semantic_distances([{"_extract": source}], answer, None)[0]


def test_a_source_the_answer_is_about_scores_close(monkeypatch):
    d = _distance(monkeypatch, INVOICE, "The Contoso invoice is for Rivera.")
    assert d <= _CITATION_MAX_DISTANCE


def test_an_unrelated_source_scores_far(monkeypatch):
    d = _distance(monkeypatch, INVOICE, "The camera sensor is a Hasselblad.")
    assert d > _CITATION_MAX_DISTANCE


def test_the_unrelated_source_is_dropped(monkeypatch):
    _stub_encoder(monkeypatch)
    assert _verified_citations([_cit()], "The camera sensor is a Hasselblad.") == []


def test_the_used_source_survives_with_private_keys_stripped(monkeypatch):
    _stub_encoder(monkeypatch)
    out = _verified_citations([_cit()], "The Contoso invoice is for Rivera.")
    assert len(out) == 1
    assert not any(k.startswith("_") for k in out[0])


def test_every_source_is_embedded_in_one_batched_call(monkeypatch):
    # Verification runs after the answer has streamed, so its cost is latency
    # the user waits through.  One call for the answer and every source, not
    # one call per source.
    calls = []
    _stub_encoder(monkeypatch, calls)
    _verified_citations(
        [_cit(name=f"doc-{i}.pdf", extract=INVOICE) for i in range(5)],
        "The Contoso invoice is for Rivera.",
    )
    assert len(calls) == 1


def test_duplicates_are_collapsed_before_embedding(monkeypatch):
    calls = []
    _stub_encoder(monkeypatch, calls)
    out = _verified_citations([_cit(), _cit(), _cit()],
                              "The Contoso invoice is for Rivera.")
    assert len(out) == 1
    # One answer window plus one source window -- the same document is not
    # embedded three times.
    assert len(calls[0]) == 2


def test_a_source_matching_only_in_its_tail_is_kept(monkeypatch):
    # A citation's extract is the whole document.  Scoring it as one blob
    # would average the matching passage away, and truncating to the first
    # window would never reach it -- so a document is scored on its closest
    # window, as retrieval scores one on its closest chunk.
    _stub_encoder(monkeypatch)
    buried = ("filler text. " * 400) + INVOICE
    out = _verified_citations([_cit(extract=buried)],
                              "The Contoso invoice is for Rivera.")
    assert len(out) == 1


def test_a_long_document_is_capped_at_max_windows(monkeypatch):
    calls = []
    _stub_encoder(monkeypatch, calls)
    _verified_citations([_cit(extract="filler text. " * 5000)], "any answer")
    assert len(calls[0]) - 1 <= _MAX_WINDOWS


def test_windows_sample_across_a_long_text_not_just_the_head():
    text = "".join(f"[{i:04d}]" + "x" * 96 for i in range(200))
    windows = _windows(text)
    assert len(windows) <= _MAX_WINDOWS
    assert all(len(w) <= _WINDOW_CHARS for w in windows)
    # The last window reaches material the first windows never see.
    assert windows[-1] not in text[:_WINDOW_CHARS * 2]


def test_a_source_with_no_extract_cannot_be_judged_and_is_kept(monkeypatch):
    _stub_encoder(monkeypatch)
    assert len(_verified_citations([_cit(extract=None)], "anything at all")) == 1


def test_verification_falls_back_to_the_lexical_check_without_an_embedder(monkeypatch):
    # No embedding model on this install: degrade to the weaker text check
    # rather than switching verification off and crediting everything.
    monkeypatch.setattr("smartloop.db.embeddings.encode_texts",
                        lambda texts, base_dir=None: None)
    assert _verified_citations([_cit()], "The camera sensor is a Hasselblad.") == []
    assert len(_verified_citations([_cit()], "Contoso paid Jordan Rivera $1,000.00")) == 1


def test_an_embedder_that_raises_never_costs_the_answer_its_sources(monkeypatch):
    def boom(texts, base_dir=None):
        raise RuntimeError("llama context died")

    monkeypatch.setattr("smartloop.db.embeddings.encode_texts", boom)
    out = _verified_citations([_cit(), _cit()], "any answer")
    assert len(out) == 1
    assert not any(k.startswith("_") for k in out[0])


# ---------------------------------------------------------------------------
# Which text a citation is judged on, and how many survive.
#
# `_extract` is the whole document: retrieval expands every match to the
# complete file before building the prompt, so a citation for a 40-page report
# carries all 40 pages.  Scoring that credits almost anything -- a long file
# covers enough ground that some page of it is related to any answer.  The
# chunks the document was retrieved *for* are the passages that made it a
# match, and each already carries the document reference the citation is built
# from, so those are what the answer is checked against.

CAMERA_CHUNK = "The Hasselblad camera uses a 100-megapixel sensor."
INVOICE_CHUNK = "Invoice from Contoso Ltd. to Jordan Rivera."


def _chunked(name="report.pdf", chunks=(), extract=INVOICE):
    return {
        "document_id": name,
        "document_name": name,
        "_extract": extract,
        "_chunks": list(chunks),
    }


def test_a_citation_is_judged_on_its_chunks_not_the_whole_document():
    assert _citation_sources(_chunked(chunks=[CAMERA_CHUNK])) == ([], [CAMERA_CHUNK])


def test_a_citation_without_chunks_falls_back_to_its_extract():
    assert _citation_sources(_chunked(chunks=[])) == ([], _windows(INVOICE))


def test_a_long_document_no_longer_rides_along_on_an_unrelated_passage(monkeypatch):
    # The document contains an invoice somewhere, but it was retrieved for its
    # camera passage.  Judged whole it looks like an invoice source; judged on
    # the chunk it matched on, it is not one.
    _stub_encoder(monkeypatch)
    answer = "The Contoso invoice is for Rivera."
    whole_doc = _chunked(chunks=[], extract=CAMERA_CHUNK + " " + INVOICE)
    assert len(_verified_citations([whole_doc], answer)) == 1
    by_chunk = _chunked(chunks=[CAMERA_CHUNK], extract=CAMERA_CHUNK + " " + INVOICE)
    assert _verified_citations([by_chunk], answer) == []


# Three sources under the ceiling at spread-out distances, plus one further
# out.  This is the shape of a real run: a camera question answered from three
# web pages at 0.123 / 0.248 / 0.354, with two unrelated fund PDFs at 0.563 and
# 0.569 also retrieved.  All three pages were used.
_ANSWER = "Invoice invoice invoice from Contoso to Rivera, invoice."
_NEAR = _chunked(name="near.pdf", chunks=[INVOICE_CHUNK])                    # .18
_MID = _chunked(name="mid.pdf", chunks=["Invoice for a camera sensor."])     # .46
_FAR = _chunked(name="far.pdf", chunks=["Invoice for a camera sensor hasselblad."])


def test_the_margin_is_wide_enough_for_sources_an_answer_spread_across(monkeypatch):
    # The failure this pins: at retrieval's own 0.15 the cutoff came out at
    # 0.273 on the run above and threw away the PCMag review of the exact
    # camera the question was about.  Sources an answer genuinely used spread
    # wider than chunks matching a query do.
    _stub_encoder(monkeypatch)
    distances = _semantic_distances([_NEAR, _MID], _ANSWER, None)
    assert distances[1] - distances[0] > 0.15, "fixture no longer spans the old margin"
    out = _verified_citations([_NEAR, _MID], _ANSWER)
    assert [c["document_name"] for c in out] == ["near.pdf", "mid.pdf"]


def test_the_margin_still_drops_what_trails_the_closest_source(monkeypatch):
    # It is a real cutoff, not a formality: the ceiling alone is not a robust
    # separator — the unrelated PDFs in that run scored 0.563, only just above
    # it — so a source that trails the best by more than the margin goes, even
    # though it clears the ceiling on its own.
    _stub_encoder(monkeypatch)
    distances = _semantic_distances([_NEAR, _FAR], _ANSWER, None)
    assert all(d <= _CITATION_MAX_DISTANCE for d in distances)
    assert distances[1] - distances[0] > _CITATION_CUTOFF_MARGIN

    out = _verified_citations([_NEAR, _MID, _FAR], _ANSWER)
    assert [c["document_name"] for c in out] == ["near.pdf", "mid.pdf"]


def test_sources_within_the_margin_of_the_best_are_all_kept(monkeypatch):
    _stub_encoder(monkeypatch)
    answer = "The Contoso invoice is for Rivera."
    a = _chunked(name="a.pdf", chunks=[INVOICE_CHUNK])
    b = _chunked(name="b.pdf", chunks=["Contoso invoice, Rivera."])
    out = _verified_citations([a, b], answer)
    assert [c["document_name"] for c in out] == ["a.pdf", "b.pdf"]


def test_the_chunk_text_never_leaks_to_the_client(monkeypatch):
    _stub_encoder(monkeypatch)
    out = _verified_citations([_chunked(chunks=[INVOICE_CHUNK])],
                              "The Contoso invoice is for Rivera.")
    assert out and not any(k.startswith("_") for k in out[0])


def test_the_lexical_fallback_also_judges_on_chunks(monkeypatch):
    monkeypatch.setattr("smartloop.db.embeddings.encode_texts",
                        lambda texts, base_dir=None: None)
    answer = "Contoso paid Jordan Rivera $1,000.00 for contract work services."
    by_chunk = _chunked(chunks=[CAMERA_CHUNK], extract=INVOICE)
    assert _verified_citations([by_chunk], answer) == []
    matching = _chunked(chunks=[INVOICE], extract=CAMERA_CHUNK)
    assert len(_verified_citations([matching], answer)) == 1


# ---------------------------------------------------------------------------
# What verification costs.
#
# Nothing on the source side.  A chunk's vector was computed when the document
# was indexed and read back out of the index to rank the search that produced
# this citation, so retrieval carries it along (`ChunkResult.embedding` ->
# `_vectors`).  Re-encoding that text here would be paying twice for the same
# numbers, after the answer has streamed, which is latency the user watches.
# The answer is the only thing that has to be embedded.

from smartloop.db.embeddings import embedding_to_blob


def _vec(text: str) -> bytes:
    return embedding_to_blob(np.asarray(_bag(text) or [1e-6] * len(_VOCAB),
                                        dtype=np.float32))


def _indexed(name="report.pdf", chunks=()):
    """A citation as retrieval builds it: chunk text and its stored vector."""
    return {
        "document_id": name,
        "document_name": name,
        "_extract": INVOICE,
        "_chunks": list(chunks),
        "_vectors": [_vec(c) for c in chunks],
    }


def test_only_the_answer_is_embedded_when_retrieval_carried_the_vectors(monkeypatch):
    calls = []
    _stub_encoder(monkeypatch, calls)
    answer = "The Contoso invoice is for Rivera."
    _verified_citations(
        [_indexed(name=f"doc-{i}.pdf", chunks=[INVOICE_CHUNK, CAMERA_CHUNK])
         for i in range(5)],
        answer,
    )
    assert calls == [[answer]]


def test_a_carried_vector_scores_the_same_as_encoding_its_text(monkeypatch):
    _stub_encoder(monkeypatch)
    answer = "The Contoso invoice is for Rivera."
    carried = _semantic_distances([_indexed(chunks=[CAMERA_CHUNK])], answer, None)
    encoded = _semantic_distances([_chunked(chunks=[CAMERA_CHUNK])], answer, None)
    assert carried[0] == pytest.approx(encoded[0])


def test_a_chunk_whose_vector_is_missing_is_encoded(monkeypatch):
    # The TF-IDF fallback path ranks without vectors and stores none, so the
    # text still has to be encoded — per chunk, not per citation.
    calls = []
    _stub_encoder(monkeypatch, calls)
    c = _indexed(chunks=[INVOICE_CHUNK, CAMERA_CHUNK])
    c["_vectors"] = [None, _vec(CAMERA_CHUNK)]
    answer = "The Contoso invoice is for Rivera."
    _verified_citations([c], answer)
    assert calls == [[answer, INVOICE_CHUNK]]


def test_a_vector_from_a_different_embedding_model_is_not_judged(monkeypatch):
    # Wrong dimensionality — the index was written by another model.  Decline
    # to score it rather than crash the response it belongs to; an unjudgeable
    # source is kept, since not measuring is not evidence of misuse.
    _stub_encoder(monkeypatch)
    c = _indexed(chunks=[CAMERA_CHUNK])
    c["_vectors"] = [embedding_to_blob(np.ones(3, dtype=np.float32))]
    assert _semantic_distances([c], "The Contoso invoice is for Rivera.", None) == [None]
    assert len(_verified_citations([c], "The camera sensor is a Hasselblad.")) == 1


def test_carried_vectors_never_leak_to_the_client(monkeypatch):
    _stub_encoder(monkeypatch)
    out = _verified_citations([_indexed(chunks=[INVOICE_CHUNK])],
                              "The Contoso invoice is for Rivera.")
    assert out and set(out[0]) == {"document_id", "document_name"}
